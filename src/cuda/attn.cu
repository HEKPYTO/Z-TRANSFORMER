// Fused causal grouped-query attention on the GPU: a benchmark harness for the forward
// and backward kernels in `attn_kernels.cu`, which it `#include`s rather than links.
// Why this file exists: `zig build attn-bench` measures one call of
// `attention.forward` and one of `attentionBackward` at each context length and
// prints them beside the PCIe floor a GPU version has to clear. On the 32-core
// orders of magnitude below the CPU at the shipped window and three at 4096, so
// the arithmetic said the kernel was worth writing. It is, and the table this
// program prints is where "worth writing" turns into "here is what it does" --
// which is a smaller number, because this kernel is limited by its own arithmetic
// and re-reads the key and value prefix per block rather than by the bus.
//
// "Fused" is a specific claim and this is where it is cashed: the score matrix is
// never written to global memory. Each block owns one (query position, query
// head) pair and walks the causal prefix one tile at a time, keeping the running
// max, the running denominator and one output column per thread entirely in
// registers. Nothing of size T x T is ever allocated, on this side of the bus or
// the other.
//
// What this file does NOT do, and must not be read as doing: it is not a
// training step *by default*. `src/model.zig`'s `cuda_attn` is false as shipped;
// switched on, it routes a step's FORWARD through `attn_kernels.cu` -- this file
// only `#include`s that to benchmark it -- and `zig build cuda-attn-check`, or
// `zig build cuda-train` for a whole run, grades the result against
// `attention.forward` on a real step's tensors. The BACKWARD has no training
// path at all: nothing in the Zig tree calls `device.Attn.backward`, so this
// harness is the only thing that exercises it.
//
// A KV cache exists (`src/kv_cache.zig`) and `src/decode.zig` decodes through
// it, so the forward kernel's `q_offset` is reached at every real decode
// position, not only at 0.
//
// BOTH parity tables here are against the CPU implementations this would
// replace -- `attention.forward` and `attentionBackward` -- and NOT against a
// external reference, which still runs entirely on the CPU and is unaffected
// by anything in this file. Neither gate is one of the eighteen in
// `tools/removed/oracle.txt`; those grade a different implementation against a
// different reference and share no number with `ATTN_TOL` or `ATTN_BWD_TOL`.
//
// Tolerance: 1e-4 absolute, deliberately looser than the 1e-5 RMSNorm kernel
// gate. The CPU accumulates scores and the weighted sum in f64 and this
// accumulates them in f32, so some difference is arithmetic rather than defect.
// The table prints the worst difference next to the gate so a reader can see how
// much of it the kernel actually uses, instead of being asked to trust that it
// passes.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

// The gate, and the number it is next to in the table below.
#define ATTN_TOL 1e-4

// The backward's gate: 1e-5, set from the first measured run, and the derivation
// that sets it is recorded here because two earlier versions of this comment were
// wrong in ways a reader would have acted on.
//
// The FIRST error was arithmetic from a superseded kernel time. It claimed the
// forward runs at "roughly a third of one percent of f32 peak" and the backward
// error budget was dominated by a chain PERMUTATION between the reference's
// h-outer/t-inner order and the kernel's. There is no permutation:
// attnDkDvKernel keeps h-outer, t-inner, which IS the reference's order (its
// outer `for (0..cfg.n_heads)` and its unrolled `s` loop), deliberately, because
// k[s] and v[s] are staged once and stay live for the whole nest. The same
// comment also quoted sqrt(8192) * 5.96e-8 * 0.37 as "1.6e-6"; it is 2.0e-6.
//
// The SECOND error was worse, because it was a false claim about how this number
// was produced: it said the gate "is set from the first measured run at 10x
// whatever the worst row read". It was not. The gate was 1e-4 and the worst row
// reads 8.345e-07, which is 120x the gate, not 10x. A provenance claim that its own
// table contradicts is worse than no claim, so this one is replaced rather than
// amended.
//
// What the measurement actually is, at 1e-4 before this change and now tightened:
//
//              dq          dk          dv      % of 1e-5
//   ctx256     7.451e-09   1.537e-08   2.384e-07    2.4%
//   ctx4096    7.451e-09   1.863e-08   2.980e-07    3.0%
//   llama3-T512 2.980e-08  7.451e-08   8.345e-07    8.3%
//
// Eleven shapes, worst 8.345e-07, so 1e-5 leaves 12x over the worst row and 4x over
// the spread between shapes. 1e-4 would have been 120x, which is a gate that cannot
// detect anything this project has evidence to look for. The norm kernel's own gate
// is 1e-5 at 28.6% used, and the precedent for setting a gate from the measurement
// and recording why is that table.
//
// The ORDER of the three is the diagnostic, and it is not arbitrary:
//
//   dq  the reference accumulates it in f64 and narrows ONCE at the store
//       (`g.dq.set`), so the whole budget is a single f32 rounding of a
//       well-conditioned sum. It reads 7.451e-09.
//   dk  same shape as dq, reads within 2.5x of it.
//   dv  the reference accumulates it as an f32 READ-MODIFY-WRITE, one narrowing
//       per term, in a chain up to group * (T - s) long (its `dv_row[...] +=`).
//       At llama3-T512 that is 4 * 512 = 2048 roundings against dq's one, and dv
//       reads 28x worse than dq. The ratio is the chain length.
//
// So dv is the largest error in the table because the REFERENCE is least accurate
// there, not because the kernel is. A kernel that accumulated dk and dv in f64
// registers and narrowed once would be MORE accurate in absolute terms and would
// disagree with this reference by ~1e-6 -- worse, not better, against the thing
// being graded. Do not "fix" it.
//
// ONE CAVEAT THAT MAKES THIS GATE NARROWER THAN IT LOOKS. `attn_twin.zig` fills
// `dout` from its own seed, uniform on [-0.5, 0.5). The real `dout` at the
// attention boundary during training arrives from `dLossDLogits`, where a
// non-target entry is about (1/vocab)/T = 4e-6 and the target entry about 4e-3.
// So this gate is calibrated on a dout two to three orders of magnitude larger
// than the one a training step will supply, and applied unchanged to a real step it
// is a much looser RELATIVE gate than 1e-5 sounds. Every number above is a
// statement about the benchmark, and the gate a training step needs must be
// scale-relative -- `gradcheck.zig` already has the machinery for that and it has
// not been applied here yet. Recorded as a known limit rather than quietly left
// implied.
#define ATTN_BWD_TOL 1e-5


// The kernels, their shared-memory arithmetic, and the C entry points that launch
// them. `#include`d, not linked, so this benchmark and a training step compile the
// SAME kernels out of one translation unit and cannot drift. The file's own header
// explains why that matters more here than a clean object boundary would.
#include "attn_kernels.cu"


#define TAG_CAP 40
#define MAX_SHAPES 64

// One line of the manifest, which src/cuda/attn_twin.zig writes.
typedef struct {
    char tag[TAG_CAP];
    int T;
    int n_heads;
    int n_kv_heads;
    int head_dim;
    int iters;
    double cpu_us;
} Shape;

// Every distinct launch configuration the sweep used, recorded rather than
// assumed. A shape that reuses the previous one's numbers is recorded once.
struct ConfigUsed {
    int group_q;
    int block;
    int tile;
    unsigned blocks_x;
};

static ConfigUsed g_configs[16];
static int g_n_configs = 0;

static void noteConfig(int group_q, int block, int tile, unsigned blocks_x) {
    for (int i = 0; i < g_n_configs; ++i) {
        if (g_configs[i].group_q == group_q && g_configs[i].block == block && g_configs[i].tile == tile &&
            g_configs[i].blocks_x == blocks_x)
            return;
    }
    if (g_n_configs < (int)(sizeof(g_configs) / sizeof(g_configs[0]))) {
        g_configs[g_n_configs++] = ConfigUsed{group_q, block, tile, blocks_x};
    }
}


struct Row {
    double max_abs;
    double kernel_us;
    double cpu_us;
    size_t argmax;
    bool ok;
};

static float *readFloats(const char *dir, const char *tag, const char *suffix, size_t expect) {
    char path[512];
    const int len = snprintf(path, sizeof path, "%s/%s%s.bin", dir, tag, suffix);
    if (len < 0 || (size_t)len >= sizeof path) {
        fprintf(stderr, "attn: FAIL blob path too long for %s/%s%s.bin\n", dir, tag, suffix);
        return NULL;
    }
    FILE *f = fopen(path, "rb");
    if (f == NULL) {
        fprintf(stderr, "attn: FAIL cannot open %s\n", path);
        return NULL;
    }
    float *buf = (float *)malloc(expect * sizeof(float));
    if (buf == NULL) {
        fprintf(stderr, "attn: FAIL cannot allocate %zu floats for %s\n", expect, path);
        fclose(f);
        return NULL;
    }
    const size_t got = fread(buf, sizeof(float), expect, f);
    fclose(f);
    if (got != expect) {
        fprintf(stderr, "attn: FAIL %s holds %zu floats, expected %zu\n", path, got, expect);
        free(buf);
        return NULL;
    }
    return buf;
}

static int readManifest(const char *dir, Shape *out, int max) {
    char path[512];
    const int len = snprintf(path, sizeof path, "%s/manifest.tsv", dir);
    if (len < 0 || (size_t)len >= sizeof path) {
        fprintf(stderr, "attn: FAIL manifest path too long\n");
        return -1;
    }
    FILE *f = fopen(path, "r");
    if (f == NULL) {
        fprintf(stderr, "attn: FAIL cannot open %s. Run attn_twin first; this program compares, "
                        "it does not generate.\n",
                path);
        return -1;
    }
    int n = 0;
    while (n < max &&
           fscanf(f, "%39s %d %d %d %d %d %lf", out[n].tag, &out[n].T, &out[n].n_heads,
                  &out[n].n_kv_heads, &out[n].head_dim, &out[n].iters, &out[n].cpu_us) == 7) {
        n++;
    }
    fclose(f);
    if (n == 0) {
        fprintf(stderr, "attn: FAIL manifest.tsv holds no readable rows\n");
        return -1;
    }
    return n;
}

static double maxAbsDiff(const float *got, const float *want, size_t n, size_t *argmax) {
    double worst = 0.0;
    size_t where = 0;
    for (size_t i = 0; i < n; ++i) {
        double d = fabs((double)got[i] - (double)want[i]);
        // A NaN is the worst result there is and has to count as one. Every
        // comparison against NaN is false, so without this a kernel that returned
        // NaN in every element would report a worst difference of 0.0 and pass a
        // gate it failed in the most total way possible. norm.cu already carries
        // this line and the reason, and this file is the second kernel to need it,
        // which is the argument for putting it somewhere both can reach.
        if (isnan(d)) d = INFINITY;
        if (d > worst) {
            worst = d;
            where = i;
        }
    }
    *argmax = where;
    return worst;
}

template <int BROKEN>
static bool runShape(const Shape *s, const char *dir, size_t shared_limit, int max_tile, Row *row) {
    const int dim = s->head_dim;
    // `zt_attn_dim_ok` rather than a third copy of the predicate. The rule is the
    // one every gate in this repository follows: a check that exists in more than
    // one place is two checks that will disagree, and there is nothing to notice
    // when they do. The benchmark's message names the shape; the shared predicate
    // stays silent, as a predicate should.
    if (!zt_attn_dim_ok(dim)) {
        fprintf(stderr,
                "attn: FAIL %s has head_dim %d, and this kernel needs an EVEN head_dim in [%d, %d]. "
                "Even, because the staged K and V rows carry a stride of dim + 1 and that stride "
                "spreads a warp over 32 banks only while it is coprime with 32. A power of two is "
                "NOT required. Refusing rather than guessing.\n",
                s->tag, dim, MIN_DIM, MAX_DIM);
        return false;
    }
    if (s->n_heads <= 0 || s->n_kv_heads <= 0 || s->n_heads % s->n_kv_heads != 0) {
        // Zero heads is in this guard for a reason: the kernel computes
        // `h / (n_heads / n_kv_heads)`, so a zero divides on the device, and
        // grid.y of zero means the kernel never launches at all -- which would
        // leave the comparison grading uninitialised cudaMalloc memory.
        fprintf(stderr, "attn: FAIL %s has %d heads over %d kv heads, which is not a group.\n",
                s->tag, s->n_heads, s->n_kv_heads);
        return false;
    }

    const size_t qn = (size_t)s->T * (size_t)s->n_heads * (size_t)dim;
    const size_t kn = (size_t)s->T * (size_t)s->n_kv_heads * (size_t)dim;

    float *hq = readFloats(dir, s->tag, ".q", qn);
    float *hk = readFloats(dir, s->tag, ".k", kn);
    float *hv = readFloats(dir, s->tag, ".v", kn);
    float *ref = readFloats(dir, s->tag, ".y", qn); // the CPU answer, on the host
    float *got = (float *)malloc(qn * sizeof(float));
    if (hq == NULL || hk == NULL || hv == NULL || ref == NULL || got == NULL) {
        free(hq); free(hk); free(hv); free(ref); free(got);
        fprintf(stderr, "attn: FAIL %s could not be loaded from %s\n", s->tag, dir);
        return false;
    }

    float *dq = NULL, *dk = NULL, *dv = NULL, *dout = NULL;
    CUDA_GO(cudaMalloc((void **)&dq, qn * sizeof(float)));
    CUDA_GO(cudaMalloc((void **)&dk, kn * sizeof(float)));
    CUDA_GO(cudaMalloc((void **)&dv, kn * sizeof(float)));
    CUDA_GO(cudaMalloc((void **)&dout, qn * sizeof(float)));
    CUDA_GO(cudaMemcpy(dq, hq, qn * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_GO(cudaMemcpy(dk, hk, kn * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_GO(cudaMemcpy(dv, hv, kn * sizeof(float), cudaMemcpyHostToDevice));

    // TRIED AND REJECTED ON MEASUREMENT, and the reason is the useful part.
    //
    // The reasoning that motivated it was wrong, and the arithmetic that said so was
    // itself computed from a kernel time this file has since retired, so it is
    // redone here from the published table.
    //
    // Traffic at T = 4096: every block (t, h) reads 2 * (t + 1) * dim floats of its
    // own kv head, so n_heads * dim * 8 * T * (T + 1) / 2 = 8.59 GB. At the published
    // 18.625 ms that is 461 GB/s IF every one of those bytes missed L2, which is the
    // pessimistic reading and not the true one -- a 32 KB tile is re-read by
    // neighbouring blocks, so much of it is served from cache. 461 GB/s is 51% of the
    // 912 GB/s this card's spec arithmetic gives (384-bit GDDR6X at 19 Gbps), but no
    // DRAM counter was read, so that is a bound on the traffic rather than a
    // measurement of it, and it is not offered as a utilisation figure.
    //
    // Arithmetic: 2 * n_heads * dim * T * (T + 1) = 4.295 GFLOP in 18.625 ms is
    // 231 GFLOP/s. No fraction of f32 peak is quoted, because this card's core count
    // and clock are recorded nowhere in this repository and inventing them here would
    // be exactly the kind of number this file is trying not to contain.
    //
    // (Two earlier versions of this comment quoted 47% and then 24%, both derived from
    // a 39 ms figure that the padding work retired. The 17.2 GB that the 47% needed is
    // also retired: it counted a write of K and V which never happens.)
    //
    // Neither reading is a limit, and the obvious suspect was the barrier count:
    // 128 tile iterations and 385 barriers per block at the LONGEST published length,
    // T = 4096 -- not the shipped window, which is T = 256 and 8 iterations
    // and 25 barriers. So the tile was derived from the
    // device's shared-memory budget instead of from dim, because `min(dim, cap)`
    // is dim whenever the cap is generous and raising the cap alone changes
    // nothing. At dim 32 that gives a tile of 256 and cuts the iterations from 128
    // to 16.
    //
    // Measured on an idle host, twice, it is SLOWER at every dim-32 row:
    //
    //              this          tile 256
    //   ctx256     201.785 us     264.403 us   1.31x slower
    //   ctx1024   2439.148 us    3656.472 us   1.50x slower
    //   ctx4096  38218.906 us   56980.945 us   1.49x slower
    //
    // and neutral at dim 128, where the budget only moves the tile from 64 to 73.
    // The kernel is not barrier bound. It is also not OCCUPANCY bound, which is
    // what an earlier version of this comment concluded from this very
    // experiment. The real defect was a 32-way shared-memory bank conflict in the
    // QK phase, described where TILE_STRIDE is defined; fixing it was worth 1.43x to
    // 1.65x, several times what any change of tile reaches. Occupancy is still
    // worth something -- eleven warps per SM against a limit of thirty-two -- but
    // it was not what was stopping this kernel.
    //
    // A tile of 256 asks for
    // 68736 bytes of shared memory per block and an sm_86 block has about 100 KB
    // to give, so ONE block is resident per SM where 8704 bytes allowed about
    // eleven. With 32 threads there is nothing else to hide latency
    // behind. Fewer barriers per unit of work bought less than the residency it
    // cost, which is the usual bargain and not a surprise in hindsight.
    //
    // So the tile stays narrow. What would actually help is fewer threads doing
    // more each -- several queries per block, so one K and V load serves all of
    // them -- which is a different kernel and not a one-line change to this one.
    //
    // A SECOND padding was tried and could not be settled: dim + 4 instead of
    // dim + 1. The reasoning in its favour is real -- dim + 1 puts every row at
    // an odd word offset, so no row but the first is 16-byte aligned and nvcc
    // cannot emit vector accesses against K or V at all, and dim + 4 aligns
    // every row. The reasoning against it is that the QK inner loop is a
    // SEQUENTIAL reduction, which the compiler may not reassociate, so the
    // accesses stay scalar and the alignment buys nothing. And the bank arithmetic
    // then goes the wrong way:
    //
    //     stride 33 (dim + 1): 32 distinct banks, 1 thread each   conflict-free
    //     stride 36 (dim + 4):  8 distinct banks, 4 threads each   4-way conflict
    //     stride 40 (dim + 8):  4 distinct banks, 8 threads each   8-way conflict
    //
    // so the prediction is that dim + 1 wins for THIS loop and dim + 4 would win
    // for a loop that could be vectorised.
    //
    // MEASURED, and the prediction was right. Three interleaved rounds on an idle
    // GPU, one of which had both arms reading 0%:
    //
    //              dim + 1      dim + 4     dim + 1 is
    //   ctx256      105.656 us   116.495 us   1.10x faster
    //   ctx4096   18753.638 us  20327.527 us   1.08x faster
    //   llama3-T512 11913.148 us 12252.570 us  1.03x faster
    //
    // and the same direction in all three rounds and all three shapes, including
    // the round where dim + 4 carried 23% background load and still lost. So the
    // alignment dim + 4 buys is worth nothing to a sequential reduction, and its
    // four-way bank conflict is worth something. The shipped stride stays dim + 1,
    // which is also the configuration the published table was measured under.
    //
    // The occupancy reading above says where the headroom is: 8704 bytes of shared
    // memory allows about eleven blocks per SM and a block here is one warp, so
    // eleven warps against a hardware limit of thirty-two. An interleaved A/B of
    // tile 32 against tile 8 (2272 bytes, thirty-two warps) could not settle it:
    // the GPU was between 68% and 100% busy from another process throughout, and
    // the spread WITHIN the tile-32 arm at T=4096 was 9.8% against a 7.7%
    // difference between the arms, with the sign flipping at shorter contexts.
    // That is not a verdict and none is claimed. It needs a GPU that is actually
    // idle, which is the same requirement the rejected tile=256 numbers met.

    // How many consecutive queries one block owns. 1 is the configuration every
    // published row was measured at, so that is the default; ATTN_GROUP_Q exists
    // so the multi-query layout can be A/B'd without editing this file. It HAS
    // been: three rounds, min-of-3, and group_q 4 loses 1.20x at the shipped
    // geometry while winning 1.15x at 4096 and 1.34x at Llama-3's. It ships at 1
    // anyway -- see the kernel's own header for why.
    const char *gq_env = getenv("ATTN_GROUP_Q");
    int group_q = gq_env == NULL ? 1 : atoi(gq_env);
    if (group_q < 1 || group_q > 64) {
        fprintf(stderr, "attn: ATTN_GROUP_Q=%s is out of range [1, 64].\n",
                gq_env == NULL ? "(unset)" : gq_env);
        return false;
    }
    if ((size_t)dim * (size_t)group_q > 1024) {
        fprintf(stderr,
                "attn: FAIL %s asks for %zu threads per block (head_dim %d x %d queries), and a "
                "block cannot hold more than 1024.\n",
                s->tag, (size_t)dim * (size_t)group_q, dim, group_q);
        return false;
    }

    // From the kernels file, not a second copy of the arithmetic. This used to be
    // spelled out here, and the two agreeing was an assumption rather than a
    // guarantee; if they ever diverged the symptom would be a silent shared-memory
    // overrun rather than a compile error, which is the worst possible failure for
    // a number that decides how much memory a block gets.
    //
    // The TILE is chosen by the same helper the library's launcher uses, against
    // the same device ceiling, and for the same reason. It used to be
    // `min(dim, max_tile)` here and a refusal in the launcher, and at head_dim 256
    // that meant the library ended the process on a width its own predicate
    // admitted: 256 + 64 + 2*64*257 = 132864 bytes against an sm_86's 101376. A
    // benchmark that sized its own tile differently from the launcher could not
    // have caught that, and one that sized it the same way would have measured a
    // configuration a caller could not launch.
    size_t shmem = 0;
    const int tile = zt_attn_fit_tile(dim, max_tile, group_q, shared_limit, &shmem);
    if (tile == 0) {
        fprintf(stderr,
                "attn: FAIL %s needs %zu bytes of shared memory per block for head_dim %d and %d "
                "queries per block, even at a tile of 1, and this device will not give more than "
                "%zu. Refusing rather than launching something that fails later with a less useful "
                "message.\n",
                s->tag, shmem, dim, group_q, shared_limit);
        return false;
    }
    // Past 48 KB a block has to ask the device for the memory, and the ask is per
    // kernel. Skipping it does not degrade anything, it fails: the launch returns
    // cudaErrorInvalidValue from inside the runtime with no mention of shared
    // memory, which is the least useful message this program could produce. It is
    // inside the template because `fusedAttnForward<BROKEN>` is a distinct kernel
    // for every variant and each needs its own opt-in. The helper reports a refusal
    // rather than exiting, and this program checks it, because "the benchmark
    // would exit" is not a diagnostic.
    if (zt_attn_optin((const void *)fusedAttnForward<BROKEN>, shmem) != 0) return false;
    const unsigned blocks_x = (unsigned)((s->T + group_q - 1) / group_q);
    noteConfig(group_q, dim * group_q, tile, blocks_x);
    dim3 grid(blocks_x, (unsigned)s->n_heads);
    dim3 block((unsigned)(dim * group_q));

    for (int i = 0; i < 3; ++i) { // warm-up, untimed, so the timing is not the first touch
        fusedAttnForward<BROKEN><<<grid, block, shmem>>>(dq, dk, dv, dout, s->T, s->n_heads,
                                                          s->n_kv_heads, dim, tile, group_q,
                                                          0, s->T);
    }
    CUDA_GO(cudaGetLastError());
    CUDA_GO(cudaDeviceSynchronize());

    const int iters = s->iters > 0 ? s->iters : 1;
    cudaEvent_t ea, eb;
    CUDA_GO(cudaEventCreate(&ea));
    CUDA_GO(cudaEventCreate(&eb));
    CUDA_GO(cudaEventRecord(ea));
    for (int i = 0; i < iters; ++i) {
        fusedAttnForward<BROKEN><<<grid, block, shmem>>>(dq, dk, dv, dout, s->T, s->n_heads,
                                                          s->n_kv_heads, dim, tile, group_q,
                                                          0, s->T);
    }
    CUDA_GO(cudaEventRecord(eb));
    CUDA_GO(cudaEventSynchronize(eb));
    float ms = 0.0f;
    CUDA_GO(cudaEventElapsedTime(&ms, ea, eb));
    CUDA_GO(cudaEventDestroy(ea));
    CUDA_GO(cudaEventDestroy(eb));

    // The answer comes back before it is compared. Comparing a device pointer
    // against a host one is the bug this line exists to make impossible to write.
    CUDA_GO(cudaMemcpy(got, dout, qn * sizeof(float), cudaMemcpyDeviceToHost));

    size_t argmax = 0;
    const double diff = maxAbsDiff(got, ref, qn, &argmax);
    row->max_abs = diff;
    row->kernel_us = (double)ms * 1000.0 / (double)iters;
    row->cpu_us = s->cpu_us;
    row->argmax = argmax;
    row->ok = diff <= ATTN_TOL;

    CUDA_GO(cudaFree(dq));
    CUDA_GO(cudaFree(dk));
    CUDA_GO(cudaFree(dv));
    CUDA_GO(cudaFree(dout));
    free(hq); free(hk); free(hv); free(ref); free(got);
    return true;
}

struct BwdRow {
    double dq_abs;
    double dk_abs;
    double dv_abs;
    double kernel_us;
    size_t dq_arg;
    size_t dk_arg;
    size_t dv_arg;
    bool ok;
};

// Three gates, not one. A single worst difference across dq, dk and dv cannot say
// WHICH gradient is wrong, and "the backward matches" is a claim about three
// numbers; grading them together would let a broken dk hide behind a good dq at
// a length where dk happens to be small.
template <int BROKEN>
static bool runBwdShape(const Shape *s, const char *dir, size_t shared_limit, int max_tile,
                        BwdRow *row) {
    const int dim = s->head_dim;
    if (!zt_attn_dim_ok(dim)) { // the shared predicate, for the reason above
        fprintf(stderr, "attn: bwd FAIL %s has head_dim %d, which must be EVEN and in [%d, %d]. "
                        "Even, because the staged K and V rows carry a stride of dim + 1 and that "
                        "stride spreads a warp over 32 banks only while it is coprime with 32. A "
                        "power of two is NOT required.\n",
                s->tag, dim, MIN_DIM, MAX_DIM);
        return false;
    }
    if (s->n_heads <= 0 || s->n_kv_heads <= 0 || s->n_heads % s->n_kv_heads != 0) {
        fprintf(stderr, "attn: bwd FAIL %s has %d heads over %d kv heads, which is not a group.\n",
                s->tag, s->n_heads, s->n_kv_heads);
        return false;
    }

    const size_t qn = (size_t)s->T * (size_t)s->n_heads * (size_t)dim;
    const size_t kn = (size_t)s->T * (size_t)s->n_kv_heads * (size_t)dim;
    const size_t mn = (size_t)s->T * (size_t)s->n_heads;

    float *hq = readFloats(dir, s->tag, ".q", qn);
    float *hk = readFloats(dir, s->tag, ".k", kn);
    float *hv = readFloats(dir, s->tag, ".v", kn);
    float *hdout = readFloats(dir, s->tag, ".dout", qn);
    float *rdq = readFloats(dir, s->tag, ".dq", qn); // the CPU answers, on the host
    float *rdk = readFloats(dir, s->tag, ".dk", kn);
    float *rdv = readFloats(dir, s->tag, ".dv", kn);
    float *gdq = (float *)malloc(qn * sizeof(float));
    float *gdk = (float *)malloc(kn * sizeof(float));
    float *gdv = (float *)malloc(kn * sizeof(float));
    if (hq == NULL || hk == NULL || hv == NULL || hdout == NULL || rdq == NULL || rdk == NULL ||
        rdv == NULL || gdq == NULL || gdk == NULL || gdv == NULL) {
        free(hq); free(hk); free(hv); free(hdout); free(rdq); free(rdk); free(rdv);
        free(gdq); free(gdk); free(gdv);
        fprintf(stderr, "attn: bwd FAIL %s could not be loaded from %s\n", s->tag, dir);
        return false;
    }

    // SEVEN separate device allocations, and the names are the point. An earlier
    // version of this function reused the forward's scratch names, so q landed in
    // a buffer called `dq` and the kernels were called as
    //
    //     attnDqKernel<<<>>>(dq, dk, dv, dout, dq, ...)
    //
    // with the same pointer as both `q` and `dq`, and the same for k/dk and v/dv.
    // Every parameter is __restrict__, so the compiler was told three times over
    // that those pointers cannot alias -- which is exactly the licence it needs to
    // hoist a store past a load. The result was a backward that ran to completion
    // and reported max_abs of 2.9e-02 at t = 3 of every shape: kernel A wrote its
    // dq over its own q, so kernel B went on to read gradients where it expected
    // queries, every score collapsed toward zero, the softmax went uniform, and
    // the error that came back was in T-independent short rows.
    //
    // dq is now a separate allocation from q. The kernel bodies never rely on the
    // two being distinct, and the cost is one extra buffer per tensor.
    float *q = NULL, *k = NULL, *v = NULL, *dout = NULL;
    float *dq = NULL, *dk = NULL, *dv = NULL;
    float *rm = NULL, *rl = NULL, *rd = NULL;
    CUDA_GO(cudaMalloc((void **)&q, qn * sizeof(float)));
    CUDA_GO(cudaMalloc((void **)&k, kn * sizeof(float)));
    CUDA_GO(cudaMalloc((void **)&v, kn * sizeof(float)));
    CUDA_GO(cudaMalloc((void **)&dout, qn * sizeof(float)));
    CUDA_GO(cudaMalloc((void **)&dq, qn * sizeof(float)));
    CUDA_GO(cudaMalloc((void **)&dk, kn * sizeof(float)));
    CUDA_GO(cudaMalloc((void **)&dv, kn * sizeof(float)));
    CUDA_GO(cudaMalloc((void **)&rm, mn * sizeof(float)));
    CUDA_GO(cudaMalloc((void **)&rl, mn * sizeof(float)));
    CUDA_GO(cudaMalloc((void **)&rd, mn * sizeof(float)));
    CUDA_GO(cudaMemcpy(q, hq, qn * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_GO(cudaMemcpy(k, hk, kn * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_GO(cudaMemcpy(v, hv, kn * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_GO(cudaMemcpy(dout, hdout, qn * sizeof(float), cudaMemcpyHostToDevice));

    // The same helper the launcher uses, and the same device ceiling, for the
    // reason the forward's copy gives. Kernel B's ask is 2*dim floats with no tile
    // in it, so there is nothing to narrow there and the opt-in is the only thing
    // that can refuse it.
    size_t shmemA = 0;
    const int tile = zt_attn_fit_tile_dq(dim, max_tile, shared_limit, &shmemA);
    if (tile == 0) {
        fprintf(stderr,
                "attn: bwd FAIL %s needs %zu bytes of shared memory per block for head_dim %d, "
                "even at a tile of 1, and this device will not give more than %zu.\n",
                s->tag, shmemA, dim, shared_limit);
        return false;
    }
    const size_t shmemB = zt_attn_bwd_dkdv_shmem(dim);
    if (zt_attn_optin((const void *)attnDqKernel<BROKEN>, shmemA) != 0) return false;
    if (zt_attn_optin((const void *)attnDkDvKernel<BROKEN>, shmemB) != 0) return false;

    dim3 gridA((unsigned)s->T, (unsigned)s->n_heads);
    dim3 gridB((unsigned)s->T, (unsigned)s->n_kv_heads);
    dim3 blockA((unsigned)dim);
    dim3 blockB((unsigned)dim);

    // NO memset of dk and dv, and the reason is structural rather than an
    // oversight. Kernel B's grid is (T, n_kv_heads) blocks of `dim` threads, so
    // its threads are in one-to-one correspondence with dk's elements and every
    // one of them is assigned, not accumulated into. A kernel that accumulated
    // into dk from a query-outer grid would need zeroing and would need atomics;
    // that formulation is rejected in attnDkDvKernel's own header.
    for (int i = 0; i < 2; ++i) { // warm-up, untimed
        attnDqKernel<BROKEN><<<gridA, blockA, shmemA>>>(q, k, v, dout, dq, rm, rl, rd, s->T,
                                                         s->n_heads, s->n_kv_heads, dim, tile);
        attnDkDvKernel<BROKEN><<<gridB, blockB, shmemB>>>(q, k, v, dout, rm, rl, rd, dk, dv,
                                                           s->T, s->n_heads, s->n_kv_heads, dim);
    }
    CUDA_GO(cudaGetLastError());
    CUDA_GO(cudaDeviceSynchronize());

    // Three timed calls, and the MINIMUM is what is reported. The forward's
    // timing loop runs the manifest's own `iters`, which is calibrated to the
    // forward; the backward does about four times the work per call, and reusing
    // that count would spend twenty seconds at the longest published length to
    // produce a number this file does not publish a ratio for.
    double best_us = 0.0;
    for (int i = 0; i < 3; ++i) {
        cudaEvent_t ea, eb;
        CUDA_GO(cudaEventCreate(&ea));
        CUDA_GO(cudaEventCreate(&eb));
        CUDA_GO(cudaEventRecord(ea));
        attnDqKernel<BROKEN><<<gridA, blockA, shmemA>>>(q, k, v, dout, dq, rm, rl, rd, s->T,
                                                         s->n_heads, s->n_kv_heads, dim, tile);
        attnDkDvKernel<BROKEN><<<gridB, blockB, shmemB>>>(q, k, v, dout, rm, rl, rd, dk, dv,
                                                           s->T, s->n_heads, s->n_kv_heads, dim);
        CUDA_GO(cudaEventRecord(eb));
        CUDA_GO(cudaEventSynchronize(eb));
        float ms = 0.0f;
        CUDA_GO(cudaEventElapsedTime(&ms, ea, eb));
        CUDA_GO(cudaEventDestroy(ea));
        CUDA_GO(cudaEventDestroy(eb));
        const double us = (double)ms * 1000.0;
        if (i == 0 || us < best_us) best_us = us;
    }

    // The answers come back before they are compared. Comparing a device pointer
    // against a host one is the bug the forward's identical line exists to make
    // impossible to write.
    CUDA_GO(cudaMemcpy(gdq, dq, qn * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_GO(cudaMemcpy(gdk, dk, kn * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_GO(cudaMemcpy(gdv, dv, kn * sizeof(float), cudaMemcpyDeviceToHost));

    row->dq_abs = maxAbsDiff(gdq, rdq, qn, &row->dq_arg);
    row->dk_abs = maxAbsDiff(gdk, rdk, kn, &row->dk_arg);
    row->dv_abs = maxAbsDiff(gdv, rdv, kn, &row->dv_arg);
    row->kernel_us = best_us;
    row->ok = row->dq_abs <= ATTN_BWD_TOL && row->dk_abs <= ATTN_BWD_TOL &&
              row->dv_abs <= ATTN_BWD_TOL;

    CUDA_GO(cudaFree(q));
    CUDA_GO(cudaFree(k));
    CUDA_GO(cudaFree(v));
    CUDA_GO(cudaFree(dout));
    CUDA_GO(cudaFree(dq));
    CUDA_GO(cudaFree(dk));
    CUDA_GO(cudaFree(dv));
    CUDA_GO(cudaFree(rm));
    CUDA_GO(cudaFree(rl));
    CUDA_GO(cudaFree(rd));
    free(hq); free(hk); free(hv); free(hdout); free(rdq); free(rdk); free(rdv);
    free(gdq); free(gdk); free(gdv);
    return true;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <scratch-dir>\n", argv[0]);
        return 2;
    }
    Shape shapes[MAX_SHAPES];
    const int n = readManifest(argv[1], shapes, MAX_SHAPES);
    if (n < 0) return 1;

    // An unrecognised ATTN_BROKEN must not fall through to the correct kernel and
    // exit 0, because then "variant 9 passed the gate" and "there is no variant 9"
    // are indistinguishable and the second one reads like the first.
    const char *broken = getenv("ATTN_BROKEN");
    int b = 0;
    if (broken != NULL) {
        // Validated as a STRING, not with atoi. `atoi("abc")` is 0, so a range
        // check on its result sends a typo to the correct kernel and reports that
        // the gate held -- the exact false reading this check exists to prevent,
        // reached by a different road.
        char *end = NULL;
        const long parsed = strtol(broken, &end, 10);
        if (end == broken || *end != '\0' || parsed < 0 || parsed > 4) {
            fprintf(stderr, "attn: ATTN_BROKEN=%s is not a variant this file has. Variants are 0 "
                            "(correct) to 4. Refusing rather than running the correct kernel and "
                            "reporting that the gate held.\n",
                    broken);
            return 2;
        }
        b = (int)parsed;
    }

    // Same string validation and the same reason as ATTN_BROKEN above: an
    // unparseable value must not fall through to the correct kernel and exit 0.
    // It is a SEPARATE variable because the backward's variants are different
    // defects from the forward's, and one knob that silently drove both would let
    // "variant 3 was caught" mean two different things in one run.
    const char *bbroken = getenv("ATTN_BWD_BROKEN");
    int bb = 0;
    if (bbroken != NULL) {
        char *bend = NULL;
        const long parsed = strtol(bbroken, &bend, 10);
        if (bend == bbroken || *bend != '\0' || parsed < 0 || parsed > 4) {
            fprintf(stderr, "attn: ATTN_BWD_BROKEN=%s is not a variant this file has. Variants are "
                            "0 (correct) to 4. Refusing rather than running the correct kernel and "
                            "reporting that the gate held.\n",
                    bbroken);
            return 2;
        }
        bb = (int)parsed;
    }

    // Queried once, and used to refuse a launch rather than to discover it the
    // hard way. The default 48 KB is what a block gets without asking; the
    // opt-in limit is higher on every card this has run on and is what a tile of
    // dim at head_dim 128 would need.
    int device = 0;
    CUDA_GO(cudaGetDevice(&device));
    int shared_limit = 48 * 1024;
    CUDA_GO(cudaDeviceGetAttribute(&shared_limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    // How many keys one iteration stages. Kept at the value every published row was
    // measured at, and the comment above explains why widening it loses. A table
    // has to come from the source that ships.
    //
    // ATTN_MAX_TILE can lower it, and it exists because the tile is also the
    // variable that decides whether a WIDTH runs at all: at head_dim 256 the cap of
    // 64 does not fit a block and the launcher narrows the tile by itself. A knob
    // that never reached the binary once made six runs of the default read as an
    // A/B, so this one is validated as a STRING for the reason ATTN_BROKEN is, and
    // the `configurations used` block at the end prints the tile every row actually
    // reached -- a run in which the knob did not arrive shows up as one tile
    // repeated. Lowering it to 1 is also the only way to grade the claim in
    // zt_attn_fit_tile that a tile of 1 is correct rather than merely narrow, since
    // at 1 every key is its own tile and the online softmax rescale runs once per
    // key.
    const char *mt_env = getenv("ATTN_MAX_TILE");
    int max_tile = 64;
    if (mt_env != NULL) {
        char *end = NULL;
        const long parsed = strtol(mt_env, &end, 10);
        if (end == mt_env || *end != '\0' || parsed < 1 || parsed > 64) {
            fprintf(stderr, "attn: ATTN_MAX_TILE=%s is not a tile cap this file has. It is 1 to "
                            "64. Refusing rather than running the default and reporting that the "
                            "gate held.\n",
                    mt_env);
            return 2;
        }
        max_tile = (int)parsed;
    }
    printf("attn: ATTN_MAX_TILE=%d\n", max_tile);

    printf("shape          ctx     cpu_us   kernel_us   max_abs     gate   used    ratio  parity  argmax\n");
    int ok = 1;
    for (int i = 0; i < n; ++i) {
        const Shape *s = &shapes[i];
        Row row = {0, 0, 0, 0, false};
        bool ran = false;
        switch (b) {
        case 1: ran = runShape<1>(s, argv[1], (size_t)shared_limit, max_tile, &row); break;
        case 2: ran = runShape<2>(s, argv[1], (size_t)shared_limit, max_tile, &row); break;
        case 3: ran = runShape<3>(s, argv[1], (size_t)shared_limit, max_tile, &row); break;
        case 4: ran = runShape<4>(s, argv[1], (size_t)shared_limit, max_tile, &row); break;
        default: ran = runShape<0>(s, argv[1], (size_t)shared_limit, max_tile, &row); break;
        }
        if (!ran) {
            ok = 0;
            continue;
        }
        if (!row.ok) ok = 0;
        const double used = row.max_abs / ATTN_TOL * 100.0;
        const double ratio = row.kernel_us > 0.0 ? (row.cpu_us / row.kernel_us) : 0.0;
        printf("%-12s %6d %10.2f %11.3f %10.3e %9.0e %6.3f%% %8.1fx  %-4s argmax %zu\n", s->tag, s->T,
               row.cpu_us, row.kernel_us, row.max_abs, ATTN_TOL, used, ratio, row.ok ? "ok" : "FAIL",
               row.argmax);
    }
    // The backward table. Separate header, separate gate, separate verdict, and
    // no ratio column: the CPU backward is O(T^2 * H * dim) in f64 and this file
    // does not time it, so a ratio here would have no measured denominator.
    // The argmax indices are printed because "the backward is wrong by 2.9e-2" does
    // not say WHERE, and the index narrows it immediately: the row an index falls
    // in is the only clue that distinguishes "row 0 is wrong", which is the single
    // key whose softmax Jacobian vanishes identically, from "every row is wrong by
    // a constant factor", which is a scale or a missing term. The forward's table
    // prints its argmax for the same reason and it has earned its place twice.
    printf("\nshape          ctx     bwd_us     max_abs_dq     max_abs_dk     max_abs_dv     "
           "gate   worst   parity      worst_dq      worst_dk      worst_dv\n");
    for (int i = 0; i < n; ++i) {
        const Shape *s = &shapes[i];
        BwdRow brow = {0, 0, 0, 0, 0, 0, 0, false};
        bool ran = false;
        switch (bb) {
        case 1: ran = runBwdShape<1>(s, argv[1], (size_t)shared_limit, max_tile, &brow); break;
        case 2: ran = runBwdShape<2>(s, argv[1], (size_t)shared_limit, max_tile, &brow); break;
        case 3: ran = runBwdShape<3>(s, argv[1], (size_t)shared_limit, max_tile, &brow); break;
        case 4: ran = runBwdShape<4>(s, argv[1], (size_t)shared_limit, max_tile, &brow); break;
        default: ran = runBwdShape<0>(s, argv[1], (size_t)shared_limit, max_tile, &brow); break;
        }
        if (!ran) {
            ok = 0;
            continue;
        }
        if (!brow.ok) ok = 0;
        double worst = brow.dq_abs;
        if (brow.dk_abs > worst) worst = brow.dk_abs;
        if (brow.dv_abs > worst) worst = brow.dv_abs;
        printf("%-12s %6d %10.2f %14.3e %14.3e %14.3e %9.0e %6.2f%%  %-4s %12zu %12zu %12zu\n",
               s->tag, s->T, brow.kernel_us, brow.dq_abs, brow.dk_abs, brow.dv_abs,
               ATTN_BWD_TOL, worst / ATTN_BWD_TOL * 100.0, brow.ok ? "ok" : "FAIL", brow.dq_arg,
               brow.dk_arg, brow.dv_arg);
    }

    // THE LIBRARY'S OWN ENTRY POINTS, once per manifest shape. Everything above
    // launches the kernels directly, through the BROKEN templates, so the two
    // `zt_attn_*` functions a training step would call to do any work were never
    // executed by the only program in the repository with a GPU. That is how a
    // launcher could carry a `cudaFuncSetAttribute` that ends the process at
    // head_dim 256 -- a width its own predicate admits -- and still be described as
    // untested rather than broken. The other four of the six are pure and are read
    // directly above. Two launches per shape, the tile the launcher chose, and the
    // return code; a non-zero code fails the run.
    //
    // T is CLAMPED to 8, because this is a launch and not a measurement: the
    // question is whether the entry point accepts the configuration, and eight rows
    // cross every guard in it. The device memory is deliberately left uninitialised
    // -- no reference is loaded and nothing is compared, so a number that is never
    // read cannot be wrong, which is the same reason `cudaMalloc` alone is refused
    // for the sweeps above. What IS checked is the return code and a clean
    // synchronise afterwards, which is where a launch that was going to fault shows
    // up.
    printf("\nlibrary entry points, one launch per manifest shape at T=8:\n");
    {
        const int Tp = 8;
        size_t qn = 0, kn = 0, mn = 0;
        for (int i = 0; i < n; ++i) {
            const Shape *s = &shapes[i];
            const size_t a = (size_t)Tp * (size_t)s->n_heads * (size_t)s->head_dim;
            const size_t b = (size_t)Tp * (size_t)s->n_kv_heads * (size_t)s->head_dim;
            const size_t c = (size_t)Tp * (size_t)s->n_heads;
            if (a > qn) qn = a;
            if (b > kn) kn = b;
            if (c > mn) mn = c;
        }
        float *pq = NULL, *pk = NULL, *pv = NULL, *po = NULL, *pdout = NULL;
        float *pdq = NULL, *pdk = NULL, *pdv = NULL, *prm = NULL, *prl = NULL, *prd = NULL;
        CUDA_GO(cudaMalloc((void **)&pq, qn * sizeof(float)));
        CUDA_GO(cudaMalloc((void **)&pk, kn * sizeof(float)));
        CUDA_GO(cudaMalloc((void **)&pv, kn * sizeof(float)));
        CUDA_GO(cudaMalloc((void **)&po, qn * sizeof(float)));
        CUDA_GO(cudaMalloc((void **)&pdout, qn * sizeof(float)));
        CUDA_GO(cudaMalloc((void **)&pdq, qn * sizeof(float)));
        CUDA_GO(cudaMalloc((void **)&pdk, kn * sizeof(float)));
        CUDA_GO(cudaMalloc((void **)&pdv, kn * sizeof(float)));
        CUDA_GO(cudaMalloc((void **)&prm, mn * sizeof(float)));
        CUDA_GO(cudaMalloc((void **)&prl, mn * sizeof(float)));
        CUDA_GO(cudaMalloc((void **)&prd, mn * sizeof(float)));
        for (int i = 0; i < n; ++i) {
            const Shape *s = &shapes[i];
            // The tile the launcher will land on, read from the same helper rather
            // than from the launch, because the launchers do not return it and
            // inventing a second source for it would be the drift this file keeps
            // having to fix elsewhere.
            size_t fneed = 0, dneed = 0;
            const int ftile =
                zt_attn_fit_tile(s->head_dim, max_tile, 1, (size_t)shared_limit, &fneed);
            const int dtile =
                zt_attn_fit_tile_dq(s->head_dim, max_tile, (size_t)shared_limit, &dneed);
            const int rf = zt_attn_forward(pq, pk, pv, po, Tp, s->n_heads, s->n_kv_heads,
                                          s->head_dim, 1, max_tile, 0, Tp, 0);
            const int rb = zt_attn_backward(pq, pk, pv, pdout, pdq, pdk, pdv, prm, prl, prd, Tp,
                                           s->n_heads, s->n_kv_heads, s->head_dim, max_tile, 0);
            CUDA_GO(cudaDeviceSynchronize());
            printf("  %-12s head_dim %-4d tile %2d/%-2d %6zu/%-6zu bytes  forward %d  backward %d%s\n",
                   s->tag, s->head_dim, ftile, dtile, fneed, dneed, rf, rb,
                   (rf != 0 || rb != 0) ? "   <-- REFUSED" : "");
            if (rf != 0 || rb != 0) ok = 0;
        }
        CUDA_GO(cudaFree(pq));
        CUDA_GO(cudaFree(pk));
        CUDA_GO(cudaFree(pv));
        CUDA_GO(cudaFree(po));
        CUDA_GO(cudaFree(pdout));
        CUDA_GO(cudaFree(pdq));
        CUDA_GO(cudaFree(pdk));
        CUDA_GO(cudaFree(pdv));
        CUDA_GO(cudaFree(prm));
        CUDA_GO(cudaFree(prl));
        CUDA_GO(cudaFree(prd));
    }

    // Printed after both sweeps, not before either. Printed once per distinct
    // configuration the forward sweep actually used, so a knob that never reached
    // the binary shows up as one row repeated rather than as a plausible-looking
    // result. That is not hypothetical: ATTN_GROUP_Q was exported on the host,
    // `docker run` did not pass it in, and six runs of the default were read as
    // an A/B before the plumbing was noticed.
    printf("\nconfigurations used: %d\n", g_n_configs);
    for (int i = 0; i < g_n_configs; ++i) {
        printf("  group_q %-3d block %-4d threads  tile %-4d  %u query blocks per head-row\n",
               g_configs[i].group_q, g_configs[i].block, g_configs[i].tile, g_configs[i].blocks_x);
    }
    printf("attn: %s\n", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}