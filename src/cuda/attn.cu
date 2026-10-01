// Fused causal grouped-query attention, forward, on the GPU.
//
// Why this file exists: `zig build attn-bench` measures one call of
// `attention.forward` at each context length and prints it beside the PCIe floor
// a GPU version has to clear. On the 32-core Linux host that floor read two
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
// What this file does NOT do, and must not be read as doing: it is the forward
// pass only. There is no backward kernel, so nothing here is wired into
// `zig build train` and this is not yet a step anyone can train through. The
// parity table is against `attention.forward`, the CPU implementation this would
// replace -- NOT against a external reference, which still runs entirely on
// the CPU and is unaffected by anything in this file.
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
// attnDkDvKernel keeps h-outer, t-inner, which IS the reference's order
// (autograd.zig:677 outer over h, :767 inner over s), deliberately, because k[s]
// and v[s] are staged once and stay live for the whole nest. The same comment
// also quoted sqrt(8192) * 5.96e-8 * 0.37 as "1.6e-6"; it is 2.0e-6.
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
// Seven shapes, worst 8.345e-07, so 1e-5 leaves 12x over the worst row and 4x over
// the spread between shapes. 1e-4 would have been 120x, which is a gate that cannot
// detect anything this project has evidence to look for. The norm kernel's own gate
// is 1e-5 at 0.6% used, and the precedent for setting a gate from the measurement
// and recording why is that table.
//
// The ORDER of the three is the diagnostic, and it is not arbitrary:
//
//   dq  the reference accumulates it in f64 and narrows ONCE at the store
//       (autograd.zig:754), so the whole budget is a single f32 rounding of a
//       well-conditioned sum. It reads 7.451e-09.
//   dk  same shape as dq, reads within 2.5x of it.
//   dv  the reference accumulates it as an f32 READ-MODIFY-WRITE, one narrowing
//       per term, in a chain up to group * (T - s) long (autograd.zig:782). At
//       llama3-T512 that is 4 * 512 = 2048 roundings against dq's one, and dv
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

#define CUDA_GO(x)                                                                                \
    do {                                                                                           \
        cudaError_t e_ = (x);                                                                      \
        if (e_ != cudaSuccess) {                                                                   \
            fprintf(stderr, "attn: FAIL %s at line %d: %s\n", #x, __LINE__,                       \
                    cudaGetErrorString(e_));                                                      \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

// The host side of the shared-memory layout: sq[dim], ss[tile], sk[tile*(dim+1)],
// sv[tile*dim]. It does NOT bind the kernel, which derives the same layout by
// pointer arithmetic, and an earlier version of this comment claimed it did. If
// the two ever disagree the result is a silent overrun rather than a compile
// error, which is the reason to say so here rather than imply a safety that
// nothing checks.
#define TILE_STRIDE(dim) ((dim) + 1)
#define SHARED_FLOATS(dim, tile) ((dim) + (tile) + 2 * (tile) * TILE_STRIDE(dim))

// One thread per output column, so `dim` is the block width at `group_q` 1 and
// `dim * group_q` above it, one query's worth of threads per group member. That is
// what lets each thread own one element of the output accumulator for the whole
// walk and never stage it through shared memory. It also means the QK pass needs no
// cross-thread reduction, because a thread computes a whole dot product for a
// whole key rather than a partial sum that would have to be combined -- which is
// a property of the strided loop, not of the tile's width. The tile is NOT always
// `dim`: the two Llama-3 rows run at dim 128 with a tile of 64.
//
// The launch does require `dim` to be a power of two in [32, 256], and refuses
// rather than assuming. Nothing in the loops needs a power of two -- `i += dim`
// covers any `dim` -- so that range is this file's own choice, made when the tile
// was pinned to `dim` and not revisited since. It is a guard, not a hardware
// constraint, and it is recorded here so nobody later treats it as one.
#define MIN_DIM 32
#define MAX_DIM 256

// BROKEN variants exist so this file's parity table can be attacked on the same
// terms as the RMSNorm kernel's. Each is a defect a real kernel has shipped with.
//   0  correct
//   1  the causal mask is dropped, so a query reads keys after it
//   2  every query head reads kv head 0, so GQA degenerates to MHA
//   3  the running-max correction is skipped, so the online softmax never
//      rescales the accumulator it already has
//   4  the 1/sqrt(dim) scale is dropped, so the softmax runs at the wrong
//      temperature
// `group_q` is how many consecutive queries one block owns. It is a launch
// parameter and it defaults to 1, which reduces this kernel EXACTLY to the one
// the published table was measured from: one query per block, `dim` threads, the
// same shared-memory layout, the same grid, the same arithmetic in the same
// order. That is deliberate, because the multi-query layout below changes every
// one of those and a table has to come from the source that ships.
//
// Why it exists. The shipped kernel runs at 11 warps per SM against a hardware
// limit of 32, and it gets there because a block is a single 32-thread warp and
// 8704 bytes of shared memory allows eleven of them. Widening the tile makes it
// worse, not better: 66816 bytes per block leaves exactly one resident block.
// Owning several queries per block is the other direction -- the block grows to
// `dim * group_q` threads, so the warps per SM go up with the block count rather
// than being capped by it, and one K and V load serves every query in the group
// instead of being re-read once per query.
//
// MEASURED, on an idle GPU, both arms of every round reading 0%. Three rounds,
// group_q 1 against group_q 4, min-of-3 timings:
//
//              group_q 1   group_q 4   the winner
//   ctx256       105.2 us    126.3 us   1, by 1.20x
//   ctx4096     19676.5 us  17074.8 us   4, by 1.15x
//   llama3-T512 12098.0 us   9012.3 us   4, by 1.34x
//
// The optimum flips, and the reason is how much work there is to fill the
// machine with. At head_dim 128 a block asks 66816 bytes, so ONE block is
// resident per SM: group_q 4 quarters the block count and quadruples the queries
// each resident block serves, which is why Llama-3's geometry gains most. At
// head_dim 32 and T = 256 there are already more blocks than resident slots --
// 256 query blocks per head-row against about 506 slots -- so group_q 4 does not
// add occupancy, it empties the machine, and it loses.
//
// It ships at 1 anyway. The honest reason is not that 1 is better everywhere: a
// table has to come from the source that ships, and picking per shape needs a
// rule for when to switch. That rule is a physical one -- switch when the grid
// still fills the SMs after the grouping -- and it is not written, so the claim
// is not made either. What is measured is the trade, and it is recorded here
// rather than in a table, because a table would imply the choice was made.
template <int BROKEN>
__global__ void fusedAttnForward(const float *__restrict__ q, // [T, n_heads*dim]
                                 const float *__restrict__ k, // [T, n_kv_heads*dim]
                                 const float *__restrict__ v, // [T, n_kv_heads*dim]
                                 float *__restrict__ out,     // [T, n_heads*dim]
                                 int T, int n_heads, int n_kv_heads, int dim, int tile,
                                 int group_q) {
    const int q0 = blockIdx.x * group_q; // first query this block owns
    const int h = blockIdx.y;           // query head
    const int g = threadIdx.x / dim;    // which query inside the group
    const int c = threadIdx.x % dim;    // output column inside that head

    const int t = q0 + g;      // this thread's query position
    const bool live = t < T;   // a tail block owns queries that do not exist

    const int group = n_heads / n_kv_heads;
    const int kvh = (BROKEN == 2) ? 0 : (h / group);

    extern __shared__ float smem[];
    float *sq = smem;                            // [group_q * dim]
    float *ss = sq + group_q * dim;              // [group_q * tile]
    // Both K and V are staged with a row stride of dim + 1, and the reason is a
    // bank conflict on both the READ and the WRITE, not on one of them.
    //
    // For a fixed inner index d, a warp of 32 threads writing row i = threadIdx.x
    // addresses c * dim + d. With dim a multiple of 32 -- 32 at the shipped
    // geometry, 128 at Llama-3's -- every one of those lands in bank d. That is a
    // 32-way conflict: 1 distinct bank out of 32. It was fixed on K first, and
    // then an audit pointed out that the V STORE has the identical shape and was
    // left conflicted; V's READS do not, being sv[i * dim + c] with c =
    // threadIdx.x, which is stride-1 across the warp.
    //
    // So the stride is dim + 1 for both. The bank becomes
    // (c * (dim + 1) + d) % 32 = (c + d) % 32, which is a bijection on the warp --
    // 32 distinct banks for the stores and for the QK reads. It stays conflict-
    // free for V's reads too: (i * (dim + 1) + c) % 32 = (i + c) % 32 across
    // c = 0..31.
    float *sk = ss + group_q * tile;                    // [tile * (dim + 1)]
    float *sv = sk + (size_t)tile * TILE_STRIDE(dim);  // [tile * (dim + 1)]

    const float scale = (BROKEN == 4) ? 1.0f : (1.0f / sqrtf((float)dim));
    const float *qrow = q + (size_t)t * (size_t)n_heads * dim + (size_t)h * dim;
    const float *kbase = k + (size_t)kvh * dim;
    const float *vbase = v + (size_t)kvh * dim;

    // Predicated, not branched: a tail block owns queries past the end when T is
    // not a multiple of group_q, and reading qrow there is past the end of a
    // cudaMalloc sized exactly T * n_heads * dim. The store below is guarded by
    // `live` and this load was not, which is an out-of-bounds global read on a
    // shipped path -- `compute-sanitizer` would flag it, and at head_dim 128 that
    // is 128 floats past the end. A ternary keeps the load inside the allocation
    // and leaves a value nobody reads, because a non-live thread's slot in `sq`
    // is only ever read by itself.
    sq[g * dim + c] = live ? qrow[c] : 0.0f;
    __syncthreads();

    float mrun = -INFINITY; // running max over the prefix seen so far
    float lrun = 0.0f;      // running softmax denominator
    float acc = 0.0f;       // this thread's one output column

    // Block-uniform, because it depends only on blockIdx.x and T: every thread
    // must cross the same number of barriers. A thread whose own query is past the
    // end keeps computing and simply does not store at the end -- returning early
    // here would be a divergent exit from a region containing __syncthreads().
    const int last = (BROKEN == 1) ? T : min(T, q0 + group_q);

    for (int j0 = 0; j0 < last; j0 += tile) {
        const int n = min(tile, last - j0);

        // K and V are shared by the whole group: loaded once, read by every
        // query in it. Each thread owns whole rows, so no two write the same one.
        for (int i = c; i < n; i += dim) {
            const float *kp = kbase + (size_t)(j0 + i) * (size_t)n_kv_heads * dim;
            const float *vp = vbase + (size_t)(j0 + i) * (size_t)n_kv_heads * dim;
            for (int d = 0; d < dim; ++d) {
                sk[i * TILE_STRIDE(dim) + d] = kp[d];
                sv[i * TILE_STRIDE(dim) + d] = vp[d];
            }
        }
        __syncthreads();

        // QK. Thread (c, g) owns key i of the tile for ITS query and no other, so
        // each score is written by exactly one thread and needs no reduction. The
        // stride matters if the tile is ever wider than the block: the `if (c < n)`
        // that this replaced left every key from `dim` onwards without a score.
        for (int i = c; i < n; i += dim) {
            const float *kp = sk + i * TILE_STRIDE(dim);
            const float *sg = sq + g * dim;
            float dot = 0.0f;
            for (int d = 0; d < dim; ++d) dot += sg[d] * kp[d];
            float score = dot * scale;
            // The causal mask, per query rather than per block: a key is visible to
            // query t only when j0 + i <= t, and the queries in a group have
            // different t, so this cannot be hoisted out of the per-query work.
            if (BROKEN != 1 && j0 + i > t) score = -INFINITY;
            ss[g * tile + i] = score;
        }
        __syncthreads();

        // Online softmax. Every thread reduces its OWN row redundantly -- it reads
        // all n scores of query g -- and so every thread handling that query lands
        // on the same numbers. Redundant arithmetic, and in exchange there is no
        // reduction across threads and no shared maximum or sum array.
        float tmax = -INFINITY;
        for (int i = 0; i < n; ++i) {
            const float sc = ss[g * tile + i];
            if (sc > tmax) tmax = sc;
        }
        const float mnew = fmaxf(mrun, tmax);
        const float corr = __expf(mrun - mnew); // exp(-inf) is 0, so tile one needs no case

        float lsum = 0.0f;
        float a = 0.0f;
        for (int i = 0; i < n; ++i) {
            const float p = __expf(ss[g * tile + i] - mnew);
            lsum += p;
            a += p * sv[i * TILE_STRIDE(dim) + c];
        }

        lrun = lrun * corr + lsum;
        acc = (BROKEN == 3) ? (acc + a) : (acc * corr + a);
        mrun = mnew;

        // Load-bearing for `ss`, `sk` and `sv`, and missing until an audit found
        // it. Every thread reads all n of its scores and column c of every staged
        // row, while the next iteration rewrites them; the barrier after the QK
        // phase sits BEFORE those reads and so cannot separate them. One warp
        // hides this because a warp retires in lockstep -- with `dim * group_q`
        // threads it is not one warp, and the failure is undefined rather than
        // wrong-looking.
        __syncthreads();
    }

    // `last` is at least 1 for every query a live thread owns, and the tile
    // carrying its own key contributes exp(0) = 1, so the denominator is >= 1 and
    // this cannot divide by zero.
    if (live) {
        out[(size_t)t * (size_t)n_heads * dim + (size_t)h * dim + c] = acc / lrun;
    }
}

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

// Shared memory for attnDqKernel: q's row, dout's row, two score rows, and the
// staged K and V tiles. `+ 2*dim` over the forward is dout's row, which the
// backward needs and the forward does not; `+ tile` is `sp`, the d-probability
// row, which the forward has no use for.
#define BWD_DQ_SHARED(dim, tile) (2 * (dim) + 2 * (tile) + 2 * (tile) * TILE_STRIDE(dim))
#define BWD_DKDV_SHARED(dim) (2 * (dim))

// Kernel A: dq, query-outer. One block per (query, head), `dim` threads, one
// thread per output column -- the same geometry as the forward, deliberately.
//
// `dq[t]` sums over the PREFIX s <= t, and `p_ds[s]` is a different value for
// every (t, s) pair, so there is nothing to reuse by making s the outer loop.
// This block owns the entire chain in registers and writes once.
//
// It also produces the three per-row scalars the other kernel needs: the row
// max `m`, the softmax denominator `denom`, and `delta = sum_s probs * d_probs`,
// which is the softmax Jacobian's other half. The reference accumulates that sum
// inline as `dot_pp` (autograd.zig:733); it is the same real number as the
// rowsum-of-dout-times-out that a flash backend precomputes, and precomputing is
// what makes the split into two kernels possible at all, because it turns an
// unbounded prefix reduction into a length-dim row reduction that can be written
// to a T*H buffer and read back.
//
// Why the online softmax and not the reference's two passes: `probs` is
// invariant to the shift (the final divide cancels it), so an online rescale is
// PERMITTED; and storing `ss` for the whole prefix would be O(T) shared memory,
// which caps T. What it costs is that `denom` is a rescaled running sum rather
// than a single finished f64 sum -- a different rounding path through `probs`,
// at roughly sqrt(n_tiles) * 6e-8 = 5e-7 relative, which is four orders below
// the gate. Do not "fix" this to a two-pass form without re-measuring.
template <int BROKEN>
__global__ void attnDqKernel(const float *__restrict__ q,    // [T, H*dim]
                             const float *__restrict__ k,    // [T, Hkv*dim]
                             const float *__restrict__ v,    // [T, Hkv*dim]
                             const float *__restrict__ dout, // [T, H*dim]
                             float *__restrict__ dq,          // [T, H*dim]
                             float *__restrict__ row_max,     // [T, H]
                             float *__restrict__ row_den,     // [T, H]
                             float *__restrict__ row_del,     // [T, H]
                             int T, int n_heads, int n_kv_heads, int dim, int tile) {
    const int t = blockIdx.x; // the query this block owns
    const int h = blockIdx.y; // its head
    const int c = threadIdx.x;

    const int group = n_heads / n_kv_heads;
    const int kvh = (BROKEN == 2) ? 0 : (h / group);

    extern __shared__ float smem[];
    float *sq = smem;                             // [dim]
    float *sdh = sq + dim;                        // [dim]
    float *ss = sdh + dim;                        // [tile]
    float *sp = ss + tile;                        // [tile]
    float *sk = sp + tile;                        // [tile * (dim + 1)]
    float *sv = sk + (size_t)tile * TILE_STRIDE(dim); // [tile * (dim + 1)]

    const float scale = (BROKEN == 4) ? 1.0f : (1.0f / sqrtf((float)dim));
    const float *qrow = q + (size_t)t * (size_t)n_heads * dim + (size_t)h * dim;
    const float *dhrow = dout + (size_t)t * (size_t)n_heads * dim + (size_t)h * dim;
    const float *kbase = k + (size_t)kvh * dim;
    const float *vbase = v + (size_t)kvh * dim;

    sq[c] = qrow[c];
    sdh[c] = dhrow[c];
    __syncthreads();

    float mrun = -INFINITY; // running max over the prefix
    float lrun = 0.0f;      // running sum of exp(score - running max) == the denominator
    float dlrun = 0.0f;     // running sum of that times d_probs
    // The dq sum is carried as TWO accumulators rather than one, because the
    // Jacobian term is not known until the whole prefix has been walked:
    //
    //   dq[c] = sum_s p_ds[s] * k[s][c] * scale,   p_ds[s] = p[s] * (dp[s] - delta)
    //         = (1/lrun) * ( sum_s pu[s]*dp[s]*k[s][c]*scale - delta * sum_s pu[s]*k[s][c]*scale )
    //         = (a1 - delta * a0) / lrun
    //
    // with pu[s] = exp(score - running max), i.e. p before the divide. Both sums
    // rescale by the same `corr` as the max does, so the online form carries
    // unchanged. `scale` rides on each term rather than being factored out of the
    // sum, because autograd.zig:754 writes it that way on purpose.
    float a0 = 0.0f;
    float a1 = 0.0f;

    const int last = t + 1; // causal: a query sees its own key and everything before

    for (int j0 = 0; j0 < last; j0 += tile) {
        const int n = min(tile, last - j0);

        for (int i = c; i < n; i += dim) {
            const float *kp = kbase + (size_t)(j0 + i) * (size_t)n_kv_heads * dim;
            const float *vp = vbase + (size_t)(j0 + i) * (size_t)n_kv_heads * dim;
            for (int d = 0; d < dim; ++d) {
                sk[i * TILE_STRIDE(dim) + d] = kp[d];
                sv[i * TILE_STRIDE(dim) + d] = vp[d];
            }
        }
        __syncthreads();

        // Two dots per key, both block-uniform and both owned by one thread each,
        // exactly as the forward's QK is. `sp` is d_probs[s] = dot(dh, v[s]).
        for (int i = c; i < n; i += dim) {
            const float *kp = sk + i * TILE_STRIDE(dim);
            const float *vp = sv + i * TILE_STRIDE(dim);
            float dot = 0.0f;
            float dotp = 0.0f;
            for (int d = 0; d < dim; ++d) {
                dot += sq[d] * kp[d];
                dotp += sdh[d] * vp[d];
            }
            ss[i] = dot * scale;
            sp[i] = dotp;
        }
        __syncthreads();

        float tmax = -INFINITY;
        for (int i = 0; i < n; ++i) {
            const float sc = ss[i];
            if (sc > tmax) tmax = sc;
        }
        const float mnew = fmaxf(mrun, tmax);
        const float corr = __expf(mrun - mnew); // exp(-inf) = 0, so tile one needs no case

        float lsum = 0.0f;
        float dlsum = 0.0f;
        float b0 = 0.0f;
        float b1 = 0.0f;
        for (int i = 0; i < n; ++i) {
            const float pu = __expf(ss[i] - mnew);
            const float dp = sp[i];
            lsum += pu;
            dlsum += pu * dp;
            b0 += pu * sk[i * TILE_STRIDE(dim) + c] * scale;
            b1 += pu * dp * sk[i * TILE_STRIDE(dim) + c] * scale;
        }

        lrun = lrun * corr + lsum;
        dlrun = dlrun * corr + dlsum;
        a0 = a0 * corr + b0;
        a1 = a1 * corr + b1;
        mrun = mnew;

        // Load-bearing for ss, sp, sk and sv. Every thread reads all n scores and
        // column c of every staged row while the next iteration rewrites them.
        __syncthreads();
    }

    // The denominator is >= 1: the tile carrying the query's own key contributes
    // exp(0) = 1, so this cannot divide by zero.
    const float delta = dlrun / lrun;
    const float acc = (BROKEN == 3) ? (a1 / lrun) : ((a1 - delta * a0) / lrun);

    dq[(size_t)t * (size_t)n_heads * dim + (size_t)h * dim + c] = acc;
    // Every thread computed the same three numbers -- they are block-uniform, by
    // the same redundant-reduction argument the forward's softmax makes -- so
    // one thread's store is enough.
    if (c == 0) {
        const size_t o = (size_t)t * (size_t)n_heads + (size_t)h;
        row_max[o] = mrun;
        row_den[o] = lrun;
        row_del[o] = (BROKEN == 3) ? 0.0f : delta;
    }
}

// Kernel B: dk and dv, KV-outer. One block per (key position, kv head), `dim`
// threads.
//
// `dk[s]` and `dv[s]` sum over the SUFFIX t >= s, across every query head in the
// group. Making s the outer loop is what puts the whole chain in one block's
// registers: the alternative, a query-outer kernel accumulating into dk, needs
// atomics, and atomics make the accumulation order -- and therefore max_abs --
// differ between runs, which is the one property this directory's published
// three-run reproducibility rests on.
//
// The accumulators are f32, and that is NOT a shortcut. The reference narrows
// each term to f32 before adding (autograd.zig:775), so an f64 register here
// would not be more accurate against the reference, only different from it, by
// the 1.4e-6 that a 8192-long f32 chain carries.
//
// The loop nest is h-outer, t-inner, which is the reference's own order
// (autograd.zig:677 outer over h, :767 inner over s accumulating into dk). That
// is deliberate: keeping it means the chain is not merely a permutation but the
// same chain, and it costs nothing here because k[s] and v[s] are staged once
// and stay live for the whole nest.
template <int BROKEN>
__global__ void attnDkDvKernel(const float *__restrict__ q,    // [T, H*dim]
                               const float *__restrict__ k,    // [T, Hkv*dim]
                               const float *__restrict__ v,    // [T, Hkv*dim]
                               const float *__restrict__ dout, // [T, H*dim]
                               const float *__restrict__ row_max,
                               const float *__restrict__ row_den,
                               const float *__restrict__ row_del,
                               float *__restrict__ dk,          // [T, Hkv*dim]
                               float *__restrict__ dv,          // [T, Hkv*dim]
                               int T, int n_heads, int n_kv_heads, int dim) {
    const int s = blockIdx.x;    // the key/value position this block owns
    const int kvh = blockIdx.y;  // which kv head
    const int c = threadIdx.x;   // output column

    const int group = n_heads / n_kv_heads;
    // BROKEN 2 collapses GQA the same way the forward's variant 2 does: every kv
    // head reads kv head 0's rows. It is caught at every shape in the table
    // because all of them have n_heads > n_kv_heads.
    const int kvr = (BROKEN == 2) ? 0 : kvh;

    extern __shared__ float smem[];
    float *sk = smem;     // [dim]
    float *sv = sk + dim; // [dim]
    const float *krow = k + (size_t)s * (size_t)n_kv_heads * dim + (size_t)kvr * dim;
    const float *vrow = v + (size_t)s * (size_t)n_kv_heads * dim + (size_t)kvr * dim;
    sk[c] = krow[c];
    sv[c] = vrow[c];
    __syncthreads();

    const float scale = (BROKEN == 4) ? 1.0f : (1.0f / sqrtf((float)dim));

    float adk = 0.0f;
    float adv = 0.0f;

    // BROKEN 1 walks the PREFIX instead of the suffix. This is the single most
    // likely defect in this kernel, because the reference's dk loop is written
    // with t outside and s inside (autograd.zig:677, :767) and reads as a prefix
    // at a glance, and because at s = 0 the two directions coincide, so position
    // zero -- the row a naive reader checks first -- is exactly right.
    const int t_first = (BROKEN == 1) ? 0 : s;
    const int t_end = (BROKEN == 1) ? min(s + 1, T) : T;

    for (int h = kvh * group; h < kvh * group + group; ++h) {
        for (int t = t_first; t < t_end; ++t) {
            const float *qt = q + (size_t)t * (size_t)n_heads * dim + (size_t)h * dim;
            const float *dt = dout + (size_t)t * (size_t)n_heads * dim + (size_t)h * dim;

            // Both dots are block-uniform and computed redundantly by all `dim`
            // threads, the same trade the forward makes and for the same reason:
            // no cross-thread reduction, no barrier inside the loop, and the
            // shared-memory reads are broadcasts.
            float dot = 0.0f;
            float dotp = 0.0f;
            for (int d = 0; d < dim; ++d) {
                dot += qt[d] * sk[d];
                dotp += dt[d] * sv[d];
            }
            const size_t o = (size_t)t * (size_t)n_heads + (size_t)h;
            const float m = row_max[o];
            const float den = row_den[o];
            const float del = row_del[o];

            const float p = __expf(dot * scale - m) / den;
            // The Jacobian. BROKEN 3 drops it, which is the backward's signature
            // defect -- the one thing it does that the forward does not.
            const float p_ds = (BROKEN == 3) ? p : (p * (dotp - del));

            adk += p_ds * qt[c] * scale;
            adv += p * dt[c]; // NOTE: dv carries no scale. autograd.zig:782.
        }
    }

    const size_t o = (size_t)s * (size_t)n_kv_heads * dim + (size_t)kvh * dim + (size_t)c;
    dk[o] = adk;
    dv[o] = adv;
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
    if (dim < MIN_DIM || dim > MAX_DIM || (dim & (dim - 1)) != 0) {
        fprintf(stderr,
                "attn: FAIL %s has head_dim %d, and this kernel needs a power of two in [%d, %d] "
                "because it runs one thread per output column. Refusing rather than guessing.\n",
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
    // worth something -- twelve warps per SM against a limit of thirty-two -- but
    // it was not what was stopping this kernel.
    //
    // A tile of 256 asks for
    // 66816 bytes of shared memory per block and an sm_86 block has about 100 KB
    // to give, so ONE block is resident per SM where 8704 bytes allowed about
    // twelve. With 32 threads per block there is nothing else to hide latency
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
    // memory allows about twelve blocks per SM and a block here is one warp, so
    // twelve warps against a hardware limit of thirty-two. An interleaved A/B of
    // tile 32 against tile 8 (2208 bytes, thirty-two warps) could not settle it:
    // the GPU was between 68% and 100% busy from another process throughout, and
    // the spread WITHIN the tile-32 arm at T=4096 was 9.8% against a 7.7%
    // difference between the arms, with the sign flipping at shorter contexts.
    // That is not a verdict and none is claimed. It needs a GPU that is actually
    // idle, which is the same requirement the rejected tile=256 numbers met.
    const int tile = dim < max_tile ? dim : max_tile;

    // How many consecutive queries one block owns. 1 is the configuration every
    // published row was measured at, so that is the default; ATTN_GROUP_Q exists
    // so the multi-query layout can be A/B'd without editing this file, and the
    // comparison has NOT been taken -- see the kernel's own header.
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

    const size_t shmem =
        (SHARED_FLOATS(dim, tile) + (size_t)(group_q - 1) * (size_t)(dim + tile)) * sizeof(float);
    if (shmem > shared_limit) {
        fprintf(stderr,
                "attn: FAIL %s needs %zu bytes of shared memory per block for head_dim %d, "
                "%d queries per block, tile %d, and this device will not give more than %zu. "
                "Refusing rather than launching something that fails later with a less useful "
                "message.\n",
                s->tag, shmem, dim, group_q, tile, shared_limit);
        return false;
    }
    // Past 48 KB a block has to ask the device for the memory, and the ask is per
    // kernel. Skipping it does not degrade anything, it fails: the launch returns
    // cudaErrorInvalidValue from inside the runtime with no mention of shared
    // memory, which is the least useful message this program could produce. It is
    // inside the template because `fusedAttnForward<BROKEN>` is a distinct kernel
    // for every variant and each needs its own opt-in.
    if (shmem > 48u * 1024u) {
        CUDA_GO(cudaFuncSetAttribute((const void *)fusedAttnForward<BROKEN>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem));
    }
    const unsigned blocks_x = (unsigned)((s->T + group_q - 1) / group_q);
    noteConfig(group_q, dim * group_q, tile, blocks_x);
    dim3 grid(blocks_x, (unsigned)s->n_heads);
    dim3 block((unsigned)(dim * group_q));

    for (int i = 0; i < 3; ++i) { // warm-up, untimed, so the timing is not the first touch
        fusedAttnForward<BROKEN><<<grid, block, shmem>>>(dq, dk, dv, dout, s->T, s->n_heads,
                                                          s->n_kv_heads, dim, tile, group_q);
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
                                                          s->n_kv_heads, dim, tile, group_q);
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
    if (dim < MIN_DIM || dim > MAX_DIM || (dim & (dim - 1)) != 0) {
        fprintf(stderr, "attn: bwd FAIL %s has head_dim %d, which is not a power of two in "
                        "[%d, %d].\n",
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

    const int tile = dim < max_tile ? dim : max_tile;
    const size_t shmemA = (size_t)BWD_DQ_SHARED(dim, tile) * sizeof(float);
    const size_t shmemB = (size_t)BWD_DKDV_SHARED(dim) * sizeof(float);
    if (shmemA > shared_limit || shmemB > shared_limit) {
        fprintf(stderr,
                "attn: bwd FAIL %s needs %zu bytes of shared memory per block and this device "
                "will not give more than %zu.\n",
                s->tag, shmemA > shmemB ? shmemA : shmemB, shared_limit);
        return false;
    }
    // Past 48 KB a block has to ask for the memory, and the ask is per kernel --
    // hence twice, because these are two distinct kernels.
    if (shmemA > 48u * 1024u) {
        CUDA_GO(cudaFuncSetAttribute((const void *)attnDqKernel<BROKEN>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmemA));
    }
    if (shmemB > 48u * 1024u) {
        CUDA_GO(cudaFuncSetAttribute((const void *)attnDkDvKernel<BROKEN>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmemB));
    }

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
    const int max_tile = 64;

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