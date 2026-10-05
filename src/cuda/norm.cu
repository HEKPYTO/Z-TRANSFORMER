// RMSNorm on the GPU, checked against the CPU twin in src/norm.zig.
//
// The specification is src/norm.zig and nothing else. Per row of a [rows, cols]
// f32 input, with a length-cols weight:
//
//     sum_sq = sum over the row of (f64)v * (f64)v     accumulated in f64
//     rms    = f32(sqrt(sum_sq / cols + 1e-5))
//     y[i]   = v[i] / rms * w[i]                       two f32 operations
//
// eps, the f64 accumulator, the narrowing to f32 before rms is used, and the
// order of the divide and the multiply are all load-bearing and all reproduced
// here. See the two blocks below that explain the one place this cannot
// reproduce the CPU exactly.
//
// There is no CPU implementation of RMSNorm in this file, on purpose. AGENTS.md
// is explicit that a second implementation we wrote is not a reference and that
// agreeing with it proves less than it appears to, so the only thing this can be
// compared against is `norm.forward` itself, run by src/cuda/norm_twin.zig on the
// same machine and over the same input bytes. That is why the shape table lives
// in the twin and is read here as a manifest.

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

// One block per row, 256 threads.
//
// 256 is a choice with two parts. It is a whole number of warps, which the
// reduction below needs, and it is at least as wide as every row this model
// builds (d_model 128, ffn 512), so a row of the shipped model is covered by a
// single stride of the loop and no thread in the block is idle. Wider blocks
// only help rows wider than this model has, and cost occupancy.
//
// The kernel carries no bounds check on the row, for the reason probe.cu gives:
// a guard here would turn an under-sized grid into rows that were never written,
// which reads as a wrong number rather than as an error. `gpuRun` refuses a bad
// shape on the host instead, where refusing is loud.
#define THREADS 256
#define WARPS (THREADS / 32)

static_assert(THREADS % 32 == 0, "the block reduction folds whole warps");
static_assert(WARPS <= 32, "blockReduceSum reads every partial with no bound of its own");

// src/norm.zig's eps. Written as a double here because the value the CPU adds is
// a f64 literal; see rmsOf for why this one really is computed in double.
#define RMS_EPS 1e-5

// The parity gate: worst absolute difference over every element of every shape,
// which has to be under this or the run fails. It is a real number and not a
// formality, because `main` also runs two deliberately broken kernels through
// the same measurement and asserts that this gate rejects them. A check that
// has never been seen to fail is not known to work.
#define PARITY_GATE 1e-5

// Broken variants, as template arguments rather than runtime flags so that
// `if constexpr` folds them away and the correct kernel carries no trace of them.
//
//   0  correct
//   1  one weight element moved by 1e-3   the smallest change that must be caught
//   2  mean instead of mean of squares    the classic "forgot the multiply"
//   3  weight indexed from x               "read the wrong tensor at the right index"
//
// Variant 3 exists because the other two could not catch that fault. Both mutate a
// value the kernel already holds in a register, so a kernel reading the wrong
// TENSOR would still be caught by them -- but nothing tested the tensor choice
// itself. It is a fault this file could make silently: `norm_twin.zig` drew the
// weight from the same PRNG seed as the input, which made `w.data[i]` bit-identical
// to `x.data[i]` for every `i < cols`, so a kernel reading `x_row[i]` where it meant
// `weight[i]` agreed with the CPU twin on every shape's first row and on the whole
// of the 1x1 and 1x4 shapes. The twin now draws from a different stream, and this
// variant is what proves the gate would notice.
//
// Variant 1 is deliberately a small perturbation and not garbage. A gate that
// rejects a kernel returning NaN has proved very little; a gate that rejects a
// 1e-3 error on one element out of 16 million has proved the 1e-5 it claims.
#define BROKEN_WEIGHT 1
#define BROKEN_MEAN 2
#define BROKEN_WRONG_TENSOR 3

// How many deliberately broken kernels the parity gate runs, and therefore the
// width of its result arrays. Named so adding a variant cannot leave a stale
// `2` behind in a bound -- which is how the fifth variant would silently vanish.
#define BROKEN_VARIANTS 3

__device__ __forceinline__ float warpReduceSum(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

// Sums across the block and gives every thread the total.
//
// The tree matters and is not decoration. Summing a row's squares serially in f32
// accumulates an error that grows with the row length: src/norm_test.zig measures
// exactly that, at 512 elements, and finds the result lands 1.667e-6 from the
// f64 answer, past the 1e-6 that file holds the CPU implementation to. A serial
// f32 reduction in this kernel would put that same drift on the GPU side and the
// difference between the two implementations would stop being about anything.
//
// The tree costs log2(n) instead of n, so the error grows with the depth of the
// reduction rather than with the width of the row. At 4096 columns a thread holds
// 16 partials and the block folds them through 8 warps, which is a chain of
// about 24 additions instead of 4096. That is why the wide shapes below land
// inside the gate rather than near its edge.
__device__ float blockReduceSum(float v) {
    __shared__ float partial[WARPS];
    const int lane = (int)(threadIdx.x % 32u);
    const int warp = (int)(threadIdx.x / 32u);

    v = warpReduceSum(v);
    if (lane == 0) partial[warp] = v;
    __syncthreads();

    // Every thread reads every partial rather than only thread 0 broadcasting
    // afterwards. WARPS is 8, so this is 8 broadcast shared loads in place of a
    // second barrier and a broadcast, and it needs no __syncthreads after it
    // because `partial` is not written again.
    float total = 0.0f;
#pragma unroll
    for (int i = 0; i < WARPS; ++i) total += partial[i];
    return total;
}

// rms exactly as src/norm.zig computes it, from an f32 total.
//
// The double here is not the accumulator. It is the last three operations of
// norm.zig's line 24, which run in f64 there: divide the f64 sum by the row
// length, add the f64 eps, take the f64 square root, then narrow the result to
// f32. Reproducing those in f64 and narrowing the same way means the only thing
// left that can differ between this kernel and the CPU is the sum itself.
//
// It is free. This runs once per row, not once per element, so a 4096-row tensor
// asks for 4096 double divisions and square roots against 16.7 million f32 loads,
// and the kernel is bandwidth bound long before that. The same reasoning rules
// out accumulating the squares in f64: that would be one double add per element,
// and on a GeForce part f64 runs at a sixtieth of f32, which is a real cost for
// a reduction that is already accurate to a part in ten million by being a tree.
__device__ __forceinline__ float rmsOf(float total, int cols) {
    const double mean = (double)total / (double)cols;
    const double rms = sqrt(mean + RMS_EPS);
    return (float)rms;
}

template <int BROKEN>
__global__ void rmsNormKernel(const float *__restrict__ x, const float *__restrict__ weight,
                              float *__restrict__ y, int cols) {
    // One row per block. The index is size_t because rows * cols is an element
    // count and 4096 x 4096 is already 16.7 million, which fits an int, but the
    // next shape up does not have to care whether this one does.
    const size_t base = (size_t)blockIdx.x * (size_t)cols;
    const float *x_row = x + base;
    float *y_row = y + base;

    // First pass: the sum of squares. The strided loop is what lets one block
    // cover a row of any length with a fixed block size, and it is the loop that
    // has to be right for a 3 x 7 tensor where 250 of the 256 threads read
    // nothing. Those threads contribute zero and still take part in the
    // reduction, which is why the result does not depend on the row length.
    float acc = 0.0f;
    for (int i = (int)threadIdx.x; i < cols; i += THREADS) {
        const float v = x_row[i];
        if constexpr (BROKEN == BROKEN_MEAN) {
            acc += v;
        } else {
            acc += v * v;
        }
    }

    const float rms = rmsOf(blockReduceSum(acc), cols);

    // Second pass: normalise and apply the weight.
    //
    // The division is a real division and not a multiply by a reciprocal, which
    // is the one place this file could have taken the fast option and did not.
    // `x / rms` is correctly rounded in f32 on both this device and the host, so
    // the two agree to the bit at this step; folding it into a reciprocal would
    // add a second rounding and buy back at most a few percent of a kernel that
    // is bound by memory. A row is read twice rather than cached in shared
    // memory for the same reason: a shared-memory copy needs a dynamic shared
    // size that caps the row width, and the second read is an L2 hit.
    for (int i = (int)threadIdx.x; i < cols; i += THREADS) {
        float w = weight[i];
        if constexpr (BROKEN == BROKEN_WEIGHT) {
            if (i == 0) w += 1e-3f;
        }
        if constexpr (BROKEN == BROKEN_WRONG_TENSOR) {
            // The index is unchanged; the TENSOR is not. This is the fault the other
            // two variants cannot express, and with `norm_twin.zig`'s old shared seed
            // it was invisible: `weight[i]` and `x_row[i]` were the same number.
            w = x_row[i];
        }
        y_row[i] = x_row[i] / rms * w;
    }
}

static void report(const char *call, cudaError_t err, int line) {
    fprintf(stderr, "norm: FAIL %s at line %d: %s (%d)\n", call, line,
            cudaGetErrorString(err), (int)err);
    fflush(stderr);
}

// Every CUDA call goes through this. `ok` is set before the jump so `cleanup`
// knows whether it is leaving with a result or without one, and `#expr` gives
// the failing expression verbatim, which is the part that is missing when a bare
// cudaError turns into an hour of guessing.
#define CUDA_GO(expr)                                    \
    do {                                                  \
        err = (expr);                                     \
        if (err != cudaSuccess) {                         \
            report(#expr, err, __LINE__);                 \
            ok = 0;                                       \
            goto cleanup;                                 \
        }                                                 \
    } while (0)

// Returns 0 when a kernel was launched and 1 when none was. The return value is the
// whole point of this signature: as `void`, a refusal that launched nothing was
// indistinguishable at the call site from a successful launch, and because no launch
// means no error, the caller's `cudaGetLastError()` came back clean and the follow-on
// `cudaMemcpy` copied whatever was in `d_y` -- uninitialised. `verify` then measured
// that garbage, recorded "caught", and exited 0. A gate that cannot tell a refusal from
// a result cannot be trusted to have run.
static int launchRmsNorm(int broken, const float *d_x, const float *d_w, float *d_y, int rows,
                         int cols) {
    switch (broken) {
        case 0:
            rmsNormKernel<0><<<rows, THREADS>>>(d_x, d_w, d_y, cols);
            break;
        case BROKEN_WEIGHT:
            rmsNormKernel<BROKEN_WEIGHT><<<rows, THREADS>>>(d_x, d_w, d_y, cols);
            break;
        case BROKEN_WRONG_TENSOR:
            rmsNormKernel<BROKEN_WRONG_TENSOR><<<rows, THREADS>>>(d_x, d_w, d_y, cols);
            break;
        case BROKEN_MEAN:
            rmsNormKernel<BROKEN_MEAN><<<rows, THREADS>>>(d_x, d_w, d_y, cols);
            break;
        default:
            // REFUSE rather than guess, and refuse LOUDLY because the alternative is
            // worse than it looks. This arm used to launch BROKEN_MEAN for anything it
            // did not recognise. Adding BROKEN_WRONG_TENSOR's kernel branch without
            // adding a case for it made variant 3 silently run the MEAN kernel: the
            // table printed "weight read from x" over a run that had never read x.
            // On the `_flat` shapes the two rows printed an identical 1.874910e+00,
            // which is the tell.
            //
            // The second trap is the one worth writing down. Replacing that default
            // with a bare `return` WITHOUT first restoring `case BROKEN_MEAN` was worse:
            // variant 2 then launched nothing at all, and because no launch means no
            // error, `cudaGetLastError()` came back clean, the follow-on `cudaMemcpy`
            // copied uninitialised device memory, and `verify` measured that garbage,
            // recorded "caught", and exited 0. Eighteen lines on stderr, "norm: OK" on
            // stdout, exit 0. Every variant now has its own case, and this arm cannot
            // be reached without a #define that has no case.
            fprintf(stderr, "norm: FAIL unknown variant %d\n", broken);
            return 1;
    }
    return 0;
}

// Runs one RMSNorm over `h_x` and `h_w` and returns a malloc'd host copy of the
// result, which the caller frees. NULL means a CUDA call failed or the shape was
// refused, and `report` has already said which.
//
// `device_bytes`, when not NULL, receives the size of the device buffers the run
// needed, which is what the memory comparison in `bench` reports.
static float *gpuRun(const float *h_x, const float *h_w, int rows, int cols, int broken,
                     size_t *device_bytes) {
    const size_t n = (size_t)rows * (size_t)cols;
    const size_t xb = n * sizeof(float);
    const size_t wb = (size_t)cols * sizeof(float);
    float *d_x = NULL;
    float *d_w = NULL;
    float *d_y = NULL;
    float *h_y = NULL;
    cudaError_t err = cudaSuccess;
    int ok = 1;

    // Refused here rather than clamped in the kernel: see the THREADS comment.
    if (rows < 1 || cols < 1) {
        fprintf(stderr, "norm: FAIL refusing shape %d x %d\n", rows, cols);
        return NULL;
    }
    if (broken < 0 || broken > BROKEN_VARIANTS) {
        fprintf(stderr, "norm: FAIL unknown variant %d\n", broken);
        return NULL;
    }

    h_y = (float *)malloc(xb);
    if (h_y == NULL) {
        fprintf(stderr, "norm: FAIL cannot allocate %zu host bytes\n", xb);
        return NULL;
    }

    CUDA_GO(cudaMalloc((void **)&d_x, xb));
    CUDA_GO(cudaMalloc((void **)&d_w, wb));
    CUDA_GO(cudaMalloc((void **)&d_y, xb));
    CUDA_GO(cudaMemcpy(d_x, h_x, xb, cudaMemcpyHostToDevice));
    CUDA_GO(cudaMemcpy(d_w, h_w, wb, cudaMemcpyHostToDevice));
    if (launchRmsNorm(broken, d_x, d_w, d_y, rows, cols) != 0) {
        // The kernel was NOT launched, so `d_y` still holds whatever the allocator
        // gave it. Falling through here would copy that back and measure it, which is
        // how an undispatched variant reported itself caught. Refuse instead.
        ok = 0;
        goto cleanup;
    }
    // The launch call returning success says the grid was well formed. It does
    // not say a kernel started.
    CUDA_GO(cudaGetLastError());
    CUDA_GO(cudaMemcpy(h_y, d_y, xb, cudaMemcpyDeviceToHost));

    if (device_bytes != NULL) *device_bytes = 2 * xb + wb;

cleanup:
    // cudaFree(NULL) is a documented no-op, so one sweep covers the paths that
    // failed before a buffer existed as well as the one that succeeded.
    if (cudaFree(d_x) != cudaSuccess) ok = 0;
    if (cudaFree(d_w) != cudaSuccess) ok = 0;
    if (cudaFree(d_y) != cudaSuccess) ok = 0;
    if (!ok) {
        free(h_y);
        return NULL;
    }
    return h_y;
}

// Worst absolute difference between two tensors, and where it is.
//
// The comparison is against the CPU answer read back off the device and the CPU
// answer from `norm.zig`, both f32, so the diff is a plain f64 subtraction of two
// f32 values and is not itself a source of error.
static double maxAbsDiff(const float *got, const float *want, size_t n, size_t *argmax) {
    double worst = 0.0;
    size_t at = 0;
    for (size_t i = 0; i < n; ++i) {
        double d = fabs((double)got[i] - (double)want[i]);
        // A NaN is the worst result there is and has to be counted as one.
        //
        // `d > worst` is false for a NaN, because every comparison against a NaN
        // is false, so an unguarded maximum silently reports a kernel that
        // returned NaN on every element as a perfect match. That is precisely
        // the failure this harness exists to catch: it is what the "mean instead
        // of mean of squares" kernel does on a row whose mean is negative, since
        // sqrt of a negative is a NaN and NaN never displaces the running
        // maximum. Found by running the gate against that kernel and watching it
        // report a difference of exactly zero.
        if (isnan(d)) d = INFINITY;
        if (d > worst) {
            worst = d;
            at = i;
        }
    }
    if (argmax != NULL) *argmax = at;
    return worst;
}

// One line of the manifest, which src/cuda/norm_twin.zig wrote.
#define TAG_CAP 40
#define KIND_CAP 16
#define MAX_SHAPES 64

typedef struct {
    char tag[TAG_CAP];
    char kind[KIND_CAP];
    int rows;
    int cols;
    int iters;
    double cpu_us;
    double alloc_us;
} Shape;

static int readManifest(const char *dir, Shape *out, int max) {
    char path[512];
    const int len = snprintf(path, sizeof path, "%s/manifest.tsv", dir);
    if (len < 0 || (size_t)len >= sizeof path) {
        fprintf(stderr, "norm: FAIL manifest path too long\n");
        return -1;
    }
    FILE *f = fopen(path, "r");
    if (f == NULL) {
        fprintf(stderr, "norm: FAIL cannot open %s. Run norm_twin first; this program compares, it does not generate.\n", path);
        return -1;
    }

    int n = 0;
    char line[256];
    while (n < max && fgets(line, (int)sizeof line, f) != NULL) {
        if (line[0] == '\n' || line[0] == '\0') continue;
        Shape s;
        const int got = sscanf(line, "%39s %d %d %15s %d %lf %lf", s.tag, &s.rows, &s.cols, s.kind,
                               &s.iters, &s.cpu_us, &s.alloc_us);
        if (got != 7) {
            fprintf(stderr, "norm: FAIL cannot parse manifest line: %s", line);
            fclose(f);
            return -1;
        }
        out[n] = s;
        n++;
    }
    // A shape past `max` was dropped by the loop condition and the caller then
    // measured `n` shapes and printed OK. One more line than the table holds is
    // exactly the case where the caller has to hear about it: the summary reports a
    // peak over the shapes it saw, and the one it did not see may be the largest.
    // `norm_twin.zig` writes 18 lines against a `MAX_SHAPES` of 64, so this is
    // unreachable today -- which is why it is a refusal rather than a resizable
    // table. Nothing would be gained by growing the table and a caller that outgrows
    // it has learned something real about the binary it is measuring.
    if (n == max && fgets(line, (int)sizeof line, f) != NULL) {
        fprintf(stderr,
                "norm: FAIL %s holds more than the %d shapes this binary was built for, so "
                "the one it dropped is not measured\n",
                path, max);
        fclose(f);
        return -1;
    }
    fclose(f);
    if (n == 0) {
        fprintf(stderr, "norm: FAIL %s is empty\n", path);
        return -1;
    }
    return n;
}

static float *readFloats(const char *dir, const char *tag, const char *suffix, size_t expect) {
    char path[512];
    const int len = snprintf(path, sizeof path, "%s/%s%s.bin", dir, tag, suffix);
    if (len < 0 || (size_t)len >= sizeof path) {
        fprintf(stderr, "norm: FAIL blob path too long for %s/%s%s.bin\n", dir, tag, suffix);
        return NULL;
    }
    FILE *f = fopen(path, "rb");
    if (f == NULL) {
        fprintf(stderr, "norm: FAIL cannot open %s\n", path);
        return NULL;
    }
    const size_t bytes = expect * sizeof(float);
    float *buf = (float *)malloc(bytes);
    if (buf == NULL) {
        fprintf(stderr, "norm: FAIL cannot allocate %zu bytes for %s\n", bytes, path);
        fclose(f);
        return NULL;
    }
    const size_t got = fread(buf, sizeof(float), expect, f);
    fclose(f);
    if (got != expect) {
        fprintf(stderr, "norm: FAIL %s holds %zu floats, expected %zu\n", path, got, expect);
        free(buf);
        return NULL;
    }
    return buf;
}

// Loads x, w and the CPU answer for one shape. Any of the three may be NULL.
static int loadShape(const char *dir, const Shape *s, float **hx, float **hw, float **hy) {
    const size_t n = (size_t)s->rows * (size_t)s->cols;
    *hx = readFloats(dir, s->tag, ".x", n);
    *hw = readFloats(dir, s->tag, ".w", (size_t)s->cols);
    *hy = readFloats(dir, s->tag, ".y", n);
    if (*hx == NULL || *hw == NULL || *hy == NULL) {
        free(*hx);
        free(*hw);
        free(*hy);
        *hx = NULL;
        *hw = NULL;
        *hy = NULL;
        return 0;
    }
    return 1;
}

static double nowUs(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return -1.0;
    // tv_nsec / 1000, not tv_nsec * 1e3. Nanoseconds to microseconds is a divide,
    // and writing it as a multiply inflates the sub-second part by a thousand
    // and then wraps, which produced an elapsed time that was both enormous and
    // occasionally negative before this line was correct.
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec / 1000.0;
}

// One timing of one shape, three ways.
//
// The three numbers are not three framings of one measurement and the
// difference between them is the whole result at the shipped model size, so all
// three are printed and none is left out:
//
//   kernel_us  launches back to back, one synchronize at the end. The per call
//              cost inside a training loop whose queue already has work in it.
//   sync_us    launch, synchronize, repeat. What a single isolated RMSNorm costs,
//              including everything the launch itself costs.
//   e2e_us     host to device, kernel, device to host, per call. What the op
//              costs when the tensor is not already resident, which is the only
//              state this repository is in today: there is no CUDA tensor type
//              yet, so every device buffer starts life in host memory.
static int benchShape(const Shape *s, const float *hx, const float *hw, double *kernel_us,
                      double *sync_us, double *e2e_us) {
    const size_t xb = (size_t)s->rows * (size_t)s->cols * sizeof(float);
    const size_t wb = (size_t)s->cols * sizeof(float);
    const int iters = s->iters;
    float *d_x = NULL;
    float *d_w = NULL;
    float *d_y = NULL;
    float *h_y = NULL;
    // NULL like every other handle in this function, and for the same reason: the
    // `cudaEventCreate` below is the FIRST CUDA call here, and `CUDA_GO` reaches
    // `cleanup` on its failure. An uninitialised `cudaEvent_t` was therefore handed
    // to `cudaEventDestroy` as an event handle made of stack bytes. The destroy is
    // also guarded rather than left to report `cudaErrorInvalidResourceHandle`, so
    // the failure that got us here is the one reported rather than the cleanup's.
    cudaEvent_t ev0 = NULL;
    cudaEvent_t ev1 = NULL;
    cudaError_t err = cudaSuccess;
    int ok = 1;
    float ms = 0.0f;
    // Declared here rather than next to the loops that use them: CUDA_GO jumps to
    // cleanup, and C++ refuses a jump that would bypass an initialization.
    double t0 = 0.0;
    double t1 = 0.0;

    CUDA_GO(cudaEventCreate(&ev0));
    CUDA_GO(cudaEventCreate(&ev1));
    CUDA_GO(cudaMalloc((void **)&d_x, xb));
    CUDA_GO(cudaMalloc((void **)&d_w, wb));
    CUDA_GO(cudaMalloc((void **)&d_y, xb));
    CUDA_GO(cudaMemcpy(d_x, hx, xb, cudaMemcpyHostToDevice));
    CUDA_GO(cudaMemcpy(d_w, hw, wb, cudaMemcpyHostToDevice));
    h_y = (float *)malloc(xb);
    if (h_y == NULL) {
        fprintf(stderr, "norm: FAIL cannot allocate %zu host bytes\n", xb);
        ok = 0;
        goto cleanup;
    }

    // One untimed launch first, so the module load and the first touch of the
    // buffers are not charged to the sample.
    launchRmsNorm(0, d_x, d_w, d_y, s->rows, s->cols);
    CUDA_GO(cudaDeviceSynchronize());

    CUDA_GO(cudaEventRecord(ev0, 0));
    for (int i = 0; i < iters; ++i) launchRmsNorm(0, d_x, d_w, d_y, s->rows, s->cols);
    CUDA_GO(cudaEventRecord(ev1, 0));
    CUDA_GO(cudaEventSynchronize(ev1));
    // An illegal access inside the kernel surfaces at the synchronize and nowhere
    // else, so this is the call that turns a faulting kernel into an exit code
    // rather than into a timing.
    CUDA_GO(cudaGetLastError());
    CUDA_GO(cudaEventElapsedTime(&ms, ev0, ev1));
    *kernel_us = (double)ms * 1000.0 / (double)iters;

    t0 = nowUs();
    for (int i = 0; i < iters; ++i) {
        launchRmsNorm(0, d_x, d_w, d_y, s->rows, s->cols);
        CUDA_GO(cudaDeviceSynchronize());
    }
    t1 = nowUs();
    *sync_us = (t1 - t0) / (double)iters;

    t0 = nowUs();
    for (int i = 0; i < iters; ++i) {
        CUDA_GO(cudaMemcpy(d_x, hx, xb, cudaMemcpyHostToDevice));
        launchRmsNorm(0, d_x, d_w, d_y, s->rows, s->cols);
        CUDA_GO(cudaMemcpy(h_y, d_y, xb, cudaMemcpyDeviceToHost));
    }
    CUDA_GO(cudaDeviceSynchronize());
    t1 = nowUs();
    *e2e_us = (t1 - t0) / (double)iters;

cleanup:
    if (cudaFree(d_x) != cudaSuccess) ok = 0;
    if (cudaFree(d_w) != cudaSuccess) ok = 0;
    if (cudaFree(d_y) != cudaSuccess) ok = 0;
    if (ev0 != NULL && cudaEventDestroy(ev0) != cudaSuccess) ok = 0;
    if (ev1 != NULL && cudaEventDestroy(ev1) != cudaSuccess) ok = 0;
    free(h_y);
    return ok;
}

// Checks every shape against the CPU twin and then checks that the gate rejects
// two broken kernels over the same shapes. Returns the number of failures.
static int verify(const char *dir, const Shape *shapes, int n) {
    printf("parity  %-12s %-12s %-9s %13s  %s\n", "shape", "size", "kind", "max_abs_diff", "gate");

    int failures = 0;
    double worst = 0.0;
    char worst_tag[TAG_CAP] = "";
    size_t worst_elems = 0;
    double gate_diff[MAX_SHAPES][BROKEN_VARIANTS];
    int gate_caught[MAX_SHAPES][BROKEN_VARIANTS];
    // Whether the variant actually launched. The third state of the gate column: a
    // row can read `caught`, `MISSED`, or neither, because the kernel behind it
    // never ran. `MISSED` means "the kernel ran and the gate failed to reject it",
    // which is a finding about the GATE; "DID NOT RUN" is a finding about the
    // HARNESS. Conflating them is how an undispatched variant reported itself caught.
    int gate_ran[MAX_SHAPES][BROKEN_VARIANTS];

    for (int i = 0; i < n; ++i) {
        const Shape *s = &shapes[i];
        float *hx = NULL;
        float *hw = NULL;
        float *hy = NULL;
        if (!loadShape(dir, s, &hx, &hw, &hy)) {
            fprintf(stderr, "norm: FAIL cannot load shape %s\n", s->tag);
            return 1;
        }
        const size_t count = (size_t)s->rows * (size_t)s->cols;
        char size[32];
        snprintf(size, sizeof size, "%dx%d", s->rows, s->cols);

        size_t arg = 0;
        size_t bytes = 0;
        float *got = gpuRun(hx, hw, s->rows, s->cols, 0, &bytes);
        if (got == NULL) {
            fprintf(stderr, "norm: FAIL kernel did not run for %s\n", s->tag);
            free(hx);
            free(hw);
            free(hy);
            return 1;
        }
        const double diff = maxAbsDiff(got, hy, count, &arg);
        const int pass = diff <= PARITY_GATE;
        if (!pass) failures++;
        if (diff > worst) {
            worst = diff;
            snprintf(worst_tag, sizeof worst_tag, "%s", s->tag);
            worst_elems = count;
        }
        printf("parity  %-12s %-12s %-9s %13.6e  %s\n", s->tag, size, s->kind, diff,
               pass ? "pass" : "FAIL");
        if (!pass) {
            printf("        worst at element %zu of %zu, row %zu column %zu\n", arg, count,
                   arg / (size_t)s->cols, arg % (size_t)s->cols);
        }
        free(got);

        // The gate, run against two broken kernels over this same shape. Both
        // must be rejected, and a run where either passes exits non-zero: that is
        // the run in which the parity number above cannot be trusted. The
        // numbers are kept rather than printed so the table can be printed once
        // at the end, in one block, instead of once per shape.
        for (int variant = 1; variant <= BROKEN_VARIANTS; ++variant) {
            float *bad = gpuRun(hx, hw, s->rows, s->cols, variant, NULL);
            if (bad == NULL) {
                fprintf(stderr, "norm: FAIL broken kernel did not run for %s (variant %d)\n",
                        s->tag, variant);
                gate_diff[i][variant - 1] = INFINITY;
                // NOT `caught`. A variant that never ran has not been caught; it has
                // not been tested, and printing `caught` for it is how the table ended
                // up claiming a wrong-tensor fault had been proven when the kernel
                // behind that row was the MEAN one. This state says neither caught
                // nor missed -- it says the row is not evidence of anything.
                gate_ran[i][variant - 1] = 0;
                gate_caught[i][variant - 1] = 0;
                failures++;
                continue;
            }
            gate_ran[i][variant - 1] = 1;
            const double bdiff = maxAbsDiff(bad, hy, count, NULL);
            gate_diff[i][variant - 1] = bdiff;
            gate_caught[i][variant - 1] = bdiff > PARITY_GATE;
            if (!gate_caught[i][variant - 1]) failures++;
            free(bad);
        }

        free(hx);
        free(hw);
        free(hy);
    }

    printf("parity: worst %.6e over %s (%zu elements), gate %.1e\n\n", worst, worst_tag, worst_elems,
           PARITY_GATE);

    printf("gate    the same measurement against %d deliberately broken kernels. A row that\n",
           BROKEN_VARIANTS);
    printf("gate    reads MISSED means this gate would not have caught that fault, and every\n");
    printf("gate    parity number above would then mean nothing.\n");
    printf("gate    %-12s %-12s %-24s %13s  %s\n", "shape", "size", "deliberate fault",
           "max_abs_diff", "caught?");
    for (int i = 0; i < n; ++i) {
        char size[32];
        snprintf(size, sizeof size, "%dx%d", shapes[i].rows, shapes[i].cols);
        for (int variant = 0; variant < BROKEN_VARIANTS; ++variant) {
            static const char *const names[BROKEN_VARIANTS] = {
                "weight[0] += 1e-3", "mean, not mean of squares", "weight read from x",
            };
            const char *what = names[variant];
            const char *verdict = !gate_ran[i][variant]     ? "DID NOT RUN"
                                  : gate_caught[i][variant] ? "caught"
                                                           : "MISSED";
            printf("gate    %-12s %-12s %-24s %13.6e  %s\n", shapes[i].tag, size, what,
                   gate_diff[i][variant], verdict);
        }
    }
    // Every pair of variants must differ on at least one shape, or two rows of that
    // table are the SAME defect under two names. This is the check `run-attn.sh`
    // has had for its four backward variants since it found the classifier problem,
    // and this file had none -- which is how BROKEN_WRONG_TENSOR went undispatched
    // and printed "weight read from x" over a run that had read the MEAN kernel. On
    // the `_flat` shapes the two rows printed an identical 1.874910e+00, and nothing
    // in the harness noticed. Comparing pairs rather than one fixed shape is
    // deliberate: two faults can coincide on a constant row, and that is a property
    // of the shape, not a duplicate variant.
    for (int a = 0; a < BROKEN_VARIANTS; ++a) {
        for (int b = a + 1; b < BROKEN_VARIANTS; ++b) {
            int differ = 0;
            for (int i = 0; i < n; ++i) {
                if (gate_diff[i][a] != gate_diff[i][b]) {
                    differ = 1;
                    break;
                }
            }
            if (!differ) {
                fprintf(stderr,
                        "norm: FAIL variants %d and %d produced IDENTICAL differences on all "
                        "%d shapes. They are the same fault under two names, and a row "
                        "reporting one as caught says nothing about the other.\n",
                        a + 1, b + 1, n);
                failures++;
            }
        }
    }
    if (failures == 0) {
        printf("gate    ...and every pair of variants differs on at least one shape, so these\n");
        printf("gate    are %d different faults rather than one fault wearing %d names.\n",
               BROKEN_VARIANTS, BROKEN_VARIANTS);
    }
    return failures;
}

static int bench(const char *dir, const Shape *shapes, int n) {
    printf("\nbench   timings in microseconds per call, on this host, both sides.\n");
    printf("bench   cpu        src/norm.zig forward as written, output allocation included\n");
    printf("bench   cpu_core   that minus the allocation, which is what the arithmetic costs\n");
    printf("bench   gpu        this kernel, launches back to back, one synchronize at the end\n");
    printf("bench   gpu_sync   this kernel, launch and synchronize every call\n");
    printf("bench   gpu_e2e    host to device, kernel, device to host, every call\n");
    printf("bench   win        cpu_core / gpu\n\n");
    printf("%-12s %-11s %10s %10s %10s %10s %10s %10s %8s\n", "shape", "size", "elements", "cpu", "cpu_core",
           "gpu", "gpu_sync", "gpu_e2e", "win");

    int failures = 0;
    const Shape *cross = NULL;
    size_t peak_bytes = 0;
    const Shape *peak_shape = NULL;

    for (int i = 0; i < n; ++i) {
        const Shape *s = &shapes[i];
        if (s->iters <= 0) continue;

        float *hx = NULL;
        float *hw = NULL;
        float *hy = NULL;
        if (!loadShape(dir, s, &hx, &hw, &hy)) {
            fprintf(stderr, "norm: FAIL cannot load shape %s\n", s->tag);
            return 1;
        }
        free(hy);

        double kernel_us = 0.0;
        double sync_us = 0.0;
        double e2e_us = 0.0;
        if (!benchShape(s, hx, hw, &kernel_us, &sync_us, &e2e_us)) {
            fprintf(stderr, "norm: FAIL benchmark did not run for %s\n", s->tag);
            free(hx);
            free(hw);
            return 1;
        }

        const double cpu_us = s->cpu_us;
        const double cpu_core = cpu_us - s->alloc_us;
        const size_t count = (size_t)s->rows * (size_t)s->cols;
        const size_t bytes = 2 * count * sizeof(float) + (size_t)s->cols * sizeof(float);
        if (bytes > peak_bytes) {
            peak_bytes = bytes;
            peak_shape = s;
        }
        char size[32];
        snprintf(size, sizeof size, "%dx%d", s->rows, s->cols);
        const double win = kernel_us > 0.0 ? cpu_core / kernel_us : 0.0;

        printf("%-12s %-11s %10zu %10.2f %10.2f %10.2f %10.2f %10.2f %7.1fx\n", s->tag, size, count,
               cpu_us, cpu_core, kernel_us, sync_us, e2e_us, win);

        // First shape in table order, going up in elements, where the kernel on
        // its own is cheaper than the CPU's arithmetic. The table is walked as
        // written, so this is a statement about these six shapes and not a claim
        // about a fitted curve.
        if (cross == NULL && kernel_us > 0.0 && kernel_us < cpu_core) cross = s;

        free(hx);
        free(hw);
    }

    if (cross != NULL) {
        printf("crossover: the kernel on its own is cheaper than src/norm.zig's arithmetic from "
               "%s (%dx%d) up, in the shapes measured here\n",
               cross->tag, cross->rows, cross->cols);
    } else {
        printf("crossover: no shape measured here makes the kernel on its own cheaper than "
               "src/norm.zig\n");
    }

    if (peak_shape != NULL) {
        printf("memory: %s needs %.1f MiB of device buffers for input, output and weight; the "
               "same tensor is %.1f MiB in host memory and is copied to and from on every call\n",
               peak_shape->tag, (double)peak_bytes / (1024.0 * 1024.0),
               (double)peak_bytes / (1024.0 * 1024.0));
    }
    return failures;
}

// Peak resident set of this process, in bytes, or 0 where there is no procfs.
// Zero rather than a guess: this number is supposed to be the memory half of the
// comparison and a fabricated one would be worse than a missing one.
static size_t peakRssBytes(void) {
    FILE *f = fopen("/proc/self/status", "r");
    if (f == NULL) return 0;
    char line[256];
    size_t out = 0;
    while (fgets(line, (int)sizeof line, f) != NULL) {
        if (strncmp(line, "VmHWM:", 6) != 0) continue;
        unsigned long long kb = 0;
        if (sscanf(line + 6, "%llu", &kb) == 1) out = (size_t)kb * 1024u;
        break;
    }
    fclose(f);
    return out;
}

int main(int argc, char **argv) {
    // The evidence lines and any FAIL reach different streams, and stderr is
    // unbuffered while a pipe buffers stdout in blocks. Without this a failure
    // that happened after a row was printed would arrive first.
    setvbuf(stdout, NULL, _IOLBF, 0);

    if (argc != 2) {
        fprintf(stderr, "usage: %s <scratch-dir>\n", argv[0]);
        fprintf(stderr, "  the scratch dir is written by src/cuda/norm_twin.zig\n");
        return 2;
    }
    const char *dir = argv[1];

    int devices = 0;
    cudaError_t err = cudaGetDeviceCount(&devices);
    if (err != cudaSuccess) {
        report("cudaGetDeviceCount", err, __LINE__);
        return 1;
    }
    if (devices < 1) {
        fprintf(stderr, "norm: FAIL no CUDA device visible to the runtime\n");
        return 1;
    }
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) {
        report("cudaGetDeviceProperties", err, __LINE__);
        return 1;
    }
    printf("device: %s, compute capability %d.%d, %d threads per block, %d warps per row\n", prop.name,
           prop.major, prop.minor, THREADS, WARPS);

    Shape shapes[MAX_SHAPES];
    const int n = readManifest(dir, shapes, MAX_SHAPES);
    if (n < 0) return 1;

    int failures = verify(dir, shapes, n);
    failures += bench(dir, shapes, n);

    const size_t rss = peakRssBytes();
    if (rss > 0) {
        printf("memory: this process peaked at %.1f MiB resident while holding the largest shape's "
               "host copies\n",
               (double)rss / (1024.0 * 1024.0));
    }

    if (failures != 0) {
        fprintf(stderr, "norm: FAIL %d check(s) failed\n", failures);
        return 1;
    }
    printf("norm: OK\n");
    return 0;
}
