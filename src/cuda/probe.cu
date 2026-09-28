// A toolchain probe, not a kernel this project ships.
//
// Its whole job is to make one claim falsifiable by a reader who has no reason
// to believe this repository: the CUDA toolchain compiles this checked-in file,
// launches a kernel on this machine's GPU, and the value the device produced
// matches a number derived on paper here. Every choice below exists to keep
// that claim narrow and cheap to check. Nothing in this file is a transformer
// operation; the first kernel with a real job is src/cuda/norm.cu, pinned
// against the CPU twin in src/norm.zig.

#include <cuda_runtime.h>
#include <stdio.h>

// One thread, one element, 1024 of them. The kernel stores the square of its
// index, so the answer is sum(i^2) over i in [0, 1024), which has the closed
// form n(n+1)(2n+1)/6 with n = 1023, that is 357389824.
//
// The integer is deliberate and load-bearing. An f32 probe would need an
// expected value derived from libm and the driver's own reduction order, so a
// difference in that policy would be reported as a broken toolchain. This has
// an exact answer, and a wrong one is a wrong one.
#define PROBE_N 1024

// 256 threads per block, chosen for exactly one reason: it divides 1024. The
// kernel carries no bounds check (see below), so a launch that does not cover
// PROBE_N exactly would leave part of the buffer untouched and the sum would
// still be a plausible-looking number rather than a failure.
#define PROBE_BLOCK 256

#define PROBE_EXPECTED_SUM 357389824LL

// A bounds check in the kernel would convert a mis-sized launch into a
// partially filled buffer, which is the failure mode this file is least able to
// detect from the outside. Refusing to launch is loud; a short sum is not. So
// the guard lives on the host, as a compile-time assertion: nvcc compiles .cu as
// C++17, which is where static_assert comes from.
static_assert(PROBE_N % PROBE_BLOCK == 0,
              "the kernel has no bounds check, so the launch must cover PROBE_N exactly");
constexpr unsigned int kBlocks = PROBE_N / PROBE_BLOCK;

// 64-bit even though this probe's sum fits comfortably inside int32, because
// 357389824 is only a factor of six below INT32_MAX and the kernel that follows
// this one will run at a size where it does not fit at all. A wrapped sum
// reads as a number, which is the one thing a probe must never do.
__global__ void fill_squares(unsigned long long *out) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    out[i] = (unsigned long long)i * (unsigned long long)i;
}

// Reports the call that failed, the line it failed on, and what the code means,
// then returns 1 for main to propagate. All three parts are here because each
// one is the part that is missing when a bare cudaError turns into an hour of
// guessing: "cudaErrorNoKernelImageForDevice" on a first run means the object
// was built for the wrong architecture, and nothing in the number says so.
static int fail(const char *call, cudaError_t err, int line) {
    fprintf(stderr, "probe: FAIL %s failed at line %d: %s (%d)\n", call, line,
            cudaGetErrorString(err), (int)err);
    return 1;
}

int main(void) {
    // The evidence lines and the FAIL line reach the same log, and a pipe
    // buffers stdout in 4 KB blocks while stderr is unbuffered. Without this the
    // FAIL line would arrive first on a run that failed after printing, and the
    // log would read as though nothing had been computed.
    setvbuf(stdout, NULL, _IOLBF, 0);

    int devices = 0;
    cudaError_t err = cudaGetDeviceCount(&devices);
    if (err != cudaSuccess) return fail("cudaGetDeviceCount", err, __LINE__);
    if (devices < 1) {
        // "no CUDA device visible" is what a missing nvidia-container-runtime
        // looks like from inside, and it means the toolchain is fine and the
        // wiring is not. Said plainly, because the alternative reading is that
        // the GPU is broken.
        fprintf(stderr, "probe: FAIL no CUDA device visible to the runtime\n");
        return 1;
    }

    // One call rather than three cudaDeviceGetAttribute calls for name, major
    // and minor. It copies about a kilobyte that goes unused, which is cheaper
    // than the three-call form's question of how a string attribute reports its
    // own length.
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) return fail("cudaGetDeviceProperties", err, __LINE__);

    int driver = 0;
    err = cudaDriverGetVersion(&driver);
    if (err != cudaSuccess) return fail("cudaDriverGetVersion", err, __LINE__);
    int runtime = 0;
    err = cudaRuntimeGetVersion(&runtime);
    if (err != cudaSuccess) return fail("cudaRuntimeGetVersion", err, __LINE__);

    printf("probe: device %s\n", prop.name);
    printf("probe: compute capability %d.%d\n", prop.major, prop.minor);
    printf("probe: driver %d, runtime %d\n", driver, runtime);

    unsigned long long *d_out = NULL;
    err = cudaMalloc((void **)&d_out, PROBE_N * sizeof(unsigned long long));
    // cudaMalloc taking void** is the API's own signature, not a cast of
    // convenience. It cannot fail to compile, and changing the allocation to
    // cudaMallocManaged would move a failure from here to the launch above it.
    if (err != cudaSuccess) return fail("cudaMalloc", err, __LINE__);

    // The device buffer is deliberately not memset to zero. Zeroing it would
    // make a kernel that writes nothing at all look like a kernel that wrote
    // the value 0, and the sum would be 0 rather than garbage. Garbage is
    // diagnosable; a clean-looking zero is not.
    fill_squares<<<kBlocks, PROBE_BLOCK>>>(d_out);
    err = cudaGetLastError();
    // The launch call returning cudaSuccess says the grid was well formed. It
    // does not say a kernel started, and it cannot say a kernel finished. This
    // is the call that turns a bad block size or an over-large grid into a
    // reported error instead of a buffer that was never written.
    if (err != cudaSuccess) return fail("cudaLaunchKernel", err, __LINE__);

    // An illegal access inside the kernel surfaces here and nowhere else. Skip
    // this and a faulting kernel leaves d_out holding whatever was in it, the
    // sum below comes out wrong or stale, and the process exits 0.
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) return fail("cudaDeviceSynchronize", err, __LINE__);

    unsigned long long host[PROBE_N];
    err = cudaMemcpy(host, d_out, sizeof host, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) return fail("cudaMemcpy", err, __LINE__);

    // The failure paths above return without freeing d_out, and that is not an
    // oversight: the process exits immediately and the driver tears the primary
    // context down with it, which releases every allocation in that context. A
    // goto-cleanup here would be several lines to avoid reclaiming something
    // that is reclaimed anyway.
    err = cudaFree(d_out);
    if (err != cudaSuccess) return fail("cudaFree", err, __LINE__);

    // Summed on the host, on purpose. The sum is the one value in this file a
    // reader can check against a closed form by eye, and a device-side
    // reduction would add an atomics or a two-pass structure to a file whose
    // only subject is whether the toolchain works. Every such line is a line
    // that can fail for a reason that has nothing to do with the toolchain.
    long long sum = 0;
    for (int i = 0; i < PROBE_N; ++i) sum += (long long)host[i];

    printf("probe: n %d, sum %lld\n", PROBE_N, sum);
    if (sum != PROBE_EXPECTED_SUM) {
        // The comparison is here rather than in run-probe.sh so that running
        // the binary by hand is as meaningful as running the script, and both
        // report the same way.
        fprintf(stderr, "probe: FAIL sum %lld, expected %lld\n", sum, PROBE_EXPECTED_SUM);
        return 1;
    }
    printf("probe: OK\n");
    return 0;
}
