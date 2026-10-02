// The attention kernels, and the C entry points that launch them.
//
// WHY THIS IS A SEPARATE FILE. `attn.cu` is a benchmark: it has a `main`, it reads
// blobs off disk, it prints tables. A training step needs the kernels and a way
// to call them, and none of the rest. Until this file existed the repository had
// no FFI surface at all -- zero declarations, four prose lines in a README
// describing a `zig build` edge that had never been written.
//
// WHY IT IS `#include`d RATHER THAN LINKED. `attn.cu` does `#include
// "attn_kernels.cu"`, so the benchmark and the library are ONE translation unit
// and cannot drift: there is no second copy of a kernel to fall out of date. The
// alternative -- compiling this separately and linking the object -- would buy
// nothing here and would introduce exactly the failure this repository keeps
// having to fix elsewhere, where two sources of the same arithmetic disagree and
// neither notices. The cost is that a caller who wants only the kernels compiles
// the benchmark too; `attn.cu` main is the price and it is a small one.
//
// THE ENTRY POINTS. Six functions with C linkage, and they are not six of one
// kind -- a reader sizing this surface needs the split:
//
//   zt_attn_forward, zt_attn_backward   launchers. Device pointers in. Return 0
//                                       on success and 1 on a refused
//                                       configuration, with the reason on
//                                       stderr. See the ASYNC note below.
//   zt_attn_dim_ok                      a predicate. Returns 0 or 1, prints
//                                       nothing.
//   zt_attn_*_shmem (three)             byte counts. Plain ints in, a byte count
//                                       out, no printing, no launch.
//
// ASYNC, AND IT IS NOT THE SAME AS "SUCCESS". Neither launcher synchronises, and
// that is deliberate: a training step calls attention hundreds of times per epoch
// and a synchronise per call would serialise the whole step against the device.
// So a 0 return means THE LAUNCH WAS ACCEPTED, not that the work finished. A
// device-side fault -- an illegal address, a misaligned access -- is not in the
// last-error slot and appears only at the next synchronising call. A caller that
// needs to know must synchronise `stream` itself and then read the error; the
// benchmark does exactly that, at attn.cu's own cudaDeviceSynchronize.
//
// None of them allocates: the caller owns the device memory.
// And one of them used to END THE PROCESS at a width it admitted: a CUDA error
// inside CUDA_GO calls exit(1), which is the benchmark's contract and is inherited
// here because this file is one translation unit with it. At `head_dim` 256 the
// shared-memory opt-in failed exactly that way, on a width zt_attn_dim_ok
// accepted, which is the worst shape a failure can have: a predicate that admits a
// configuration and a launcher that ends the process on it. The two launchers now
// size the tile against the device's own ceiling first and REFUSE -- 1, with the
// reason on stderr -- where they used to exit. Every other CUDA error here is
// still exit(1), and that is stated rather than discovered.
//
// That is deliberate and it is why this file needs no allocator -- a training
// step calls attention 4 layers x 123 windows x 2 (forward and backward) per
// epoch, so a cudaMalloc per call would be ~1000 synchronising allocations per
// epoch. `attn.cu` own harness does exactly that only because it visits each
// shape once.
//
// The shared-memory arithmetic is exported rather than recomputed here, and
// `attn.cu` calls THESE rather than its own copy. One source of truth for a
// number that, if the two ever disagreed, would be a silent shared-memory overrun
// rather than a compile error.

// Its own includes, so the file compiles on its own terms. It did not need them
// while `attn.cu` included these headers before this file, and it still does not --
// but a `.cu` file with a kernel in it that only compiles because of a sibling's
// include order is a trap for whoever adds it to a compile list.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

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
// The launch requires `dim` to be EVEN and inside [32, 256], and refuses rather
// than assuming. The two halves of that are different in kind, so they are stated
// separately instead of as one rule.
//
// EVEN is arithmetic, and it is the same arithmetic TILE_STRIDE already encodes.
// Both K and V are staged with a row stride of `dim + 1` so that a warp's store
// reaches 32 distinct banks rather than one, and the bank thread `c` lands in is
// `(c * (dim + 1) + d) % 32`. That is a bijection across c = 0..31 exactly while
// `dim + 1` is COPRIME WITH 32, which is to say while `dim` is even. At an odd
// `dim` the stride is even, the map collapses onto half the banks, and the
// conflict the padding exists to remove comes back on the staging store and on
// the QK read of it. It is also the width the layer above this one would reject
// anyway: `rope.forward` returns `error.OddHeadDim` for an odd head_dim, because
// RoPE rotates a head in adjacent pairs, so a kernel that accepted one would be
// accepting a width nothing upstream of it can produce.
//
// A POWER OF TWO is not required, and requiring one is what this file used to do.
// The old guard tested `(dim & (dim - 1)) != 0` and so refused 96, 192 and 80 --
// Phi-3's head, DeepSeek-V2's and V3's MLA head, and a family of 80-wide ones --
// while the loops that would have run at those widths need nothing of the kind:
// `i += dim` and `c = threadIdx.x % dim` are exact for any `dim`, and the guard
// was left over from when the tile was pinned to `dim` and the block had to
// divide a warp. This file's own comment said so at the time and the code did
// the opposite; that is recorded here so the two cannot drift apart again.
//
// The RANGE is still this file's own choice, still a guard and not a hardware
// limit, and it stays where it is for a reason that is about performance rather
// than correctness. `dim` is the block width, so below 32 the block is narrower
// than a warp and lanes sit idle while the QK pass hands one whole key to each
// live thread; above 256 there is no head this repository has been asked to
// grade. Relaxing either end wants a measurement, and an unmeasured widening of a
// guard is how a correctness range becomes a performance claim nobody made.
#define MIN_DIM 32
#define MAX_DIM 256

// BROKEN variants exist so the benchmark's parity table can be attacked on the same
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
                                 int group_q,
                                 // q_offset and n_keys exist for DECODE, and are what
                                 // makes this kernel usable with a KV cache. They are zero
                                 // and T for a training-shaped call, so every published row
                                 // is measured on the identical arithmetic.
                                 //
                                 // `q_offset` is the ABSOLUTE position of query row 0. With
                                 // it at 0 -- training, and the shape this file's table was
                                 // measured at -- `q_offset + t` is just `t` and the mask
                                 // below is the one that was always here.
                                 //
                                 // `n_keys` is how many rows of k and v actually exist. For
                                 // training that is T. For decode it is however much of the
                                 // cache has been filled, and it is what stops the prefix walk
                                 // from running off the end of a partially filled cache.
                                 int q_offset, int n_keys) {
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
    // `min(n_keys, ...)`, not `min(T, ...)`: this walk is over KEYS, and for a
    // decode call the keys are the cache -- up to n_keys rows -- while T counts
    // QUERY rows and is 1. Capping at T there would silently attend to a
    // one-element prefix of the cache and return the wrong answer with no error.
    //
    // At q_offset 0 and n_keys T this is exactly the old expression, which is what
    // keeps every published row measuring the same arithmetic.
    const int last = (BROKEN == 1) ? T : min(n_keys, q_offset + q0 + group_q);

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
            // query t only when j0 + i <= q_offset + t, and the queries in a group have
            // different t, so this cannot be hoisted out of the per-query work.
            //
            // `+ q_offset` is the whole of decode support and it is one term. At
            // q_offset 0 it is the identity, so training is unaffected; at q_offset
            // = pos with a single-token query, key index j is visible iff j <= pos,
            // which is the cache semantics. WITHOUT it the mask compares a key index
            // against a query index that are indexing different arrays, and for a
            // decode call -- q one row, k the whole cache -- every key past index 0 is
            // masked out and the kernel silently returns v[0] for every position.
            if (BROKEN != 1 && j0 + i > q_offset + t) score = -INFINITY;
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

// ---------------------------------------------------------------------------
// C entry points. Everything above this line is the kernels; everything below is
// the smallest host-side surface a training step needs.
// ---------------------------------------------------------------------------

extern "C" {

// Is `dim` a launchable width? Two requirements, and they are not the same kind:
// `dim` must be EVEN, which is arithmetic and is argued where TILE_STRIDE and
// MIN_DIM are defined, and it must be inside this file's own [MIN_DIM, MAX_DIM],
// which is a choice about occupancy and is argued there too. One predicate, so
// the library and the benchmark refuse exactly the same widths and cannot drift.
int zt_attn_dim_ok(int dim) {
    // SILENT. This reads as a predicate, and a caller asking a question should not
    // get output for it -- the two launchers call it and print their own message
    // with their own context, and printing here produced two lines of diagnostics
    // for one refusal, in two different vocabularies.
    //
    // The test is `dim & 1`, and it is the WHOLE of the width contract. It was
    // `dim & (dim - 1)`, which also demanded a power of two, and that rejected
    // head_dims of 96, 192 and 80 -- every one of them a shipping model -- for a
    // property no loop in this file uses.
    return !(dim < MIN_DIM || dim > MAX_DIM || (dim & 1) != 0);
}

// The one place that explains a refused width, called by both launchers and by
// nothing else. Two copies of a seven-line diagnostic is two diagnostics that will
// one day say different things about the same refusal, and a caller reading the
// shorter one will conclude the contract is shorter than it is.
static int zt_attn_refuse_dim(int dim) {
    fprintf(stderr, "zt_attn: head_dim %d is refused. It must be even and in [%d, %d].\n", dim,
            MIN_DIM, MAX_DIM);
    fprintf(stderr,
            "       Even, because the staged K and V rows are padded to a stride of dim + 1 and\n");
    fprintf(stderr,
            "       that stride spreads a warp over 32 banks only while it is coprime with 32.\n");
    fprintf(stderr,
            "       A power of two is not required, and the check that used to demand one\n");
    fprintf(stderr,
            "       refused 96, 192 and 80 -- Phi-3, DeepSeek's MLA, and a family of 80-wide\n");
    fprintf(stderr, "       heads -- for a property none of this file's loops read.\n");
    return 1;
}

// The tile ceiling, before shared memory has had a say. Both launchers reach this
// through `zt_attn_fit_tile` below, which is why it is still one helper and not
// two.
//
// A tile of ZERO does not produce a wrong answer. It produces a HANG: every tile
// loop is `for (j0 = 0; j0 < last; j0 += tile)` with `last >= 1` and a
// __syncthreads() inside, so with `tile == 0` the loop never advances and the
// device wedges until it is reset. `max_tile` became a public parameter when these
// entry points were written, and neither the benchmark (which hardcodes 64) nor
// the original caller could reach that value -- so the guard was missing from a
// signature that now accepts it. One static helper, so it cannot drift between
// the two launchers.
static int zt_attn_tile(int dim, int max_tile) {
    if (max_tile < 1) return 0;
    return dim < max_tile ? dim : max_tile;
}

size_t zt_attn_forward_shmem(int dim, int tile, int group_q) {
    return (size_t)(SHARED_FLOATS(dim, tile) + (size_t)(group_q - 1) * (size_t)(dim + tile)) *
           sizeof(float);
}

size_t zt_attn_bwd_dq_shmem(int dim, int tile) {
    return (size_t)BWD_DQ_SHARED(dim, tile) * sizeof(float);
}

size_t zt_attn_bwd_dkdv_shmem(int dim) { return (size_t)BWD_DKDV_SHARED(dim) * sizeof(float); }

// The device's per-block opt-in shared-memory ceiling, asked ONCE and remembered.
//
// It is a device constant. A training step asks it four times a layer, so querying
// it per launch is a runtime call per attention call to re-read a number that
// cannot move. Cached per DEVICE rather than once per process, because a process
// with two devices would otherwise size the second one's blocks against the first
// one's answer and refuse a launch the second card would have granted. No lock: two
// threads racing here write the same value into the same pair, so the worst case is
// a redundant query rather than a wrong answer.
//
// Zero means "not known", and it is also what a failed query returns. Callers read
// a limit of zero as a refusal, which is the right reading of both cases: a device
// whose ceiling cannot be read cannot be sized for.
static size_t zt_attn_shmem_limit(void) {
    static size_t cached = 0;
    static int cached_for = -1;
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return 0;
    if (cached > 0 && cached_for == dev) return cached;
    int bytes = 0;
    if (cudaDeviceGetAttribute(&bytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev) != cudaSuccess)
        return 0;
    cached_for = dev;
    cached = (size_t)bytes;
    return cached;
}

// The largest tile this device will actually give a block, and the bytes it wants.
// Returns 0 when nothing from `max_tile` down to 1 fits, which is the refusal case;
// `*need` then carries the ask at tile 1, so a caller can print both the smallest
// request the kernel can make and the ceiling that refused it.
//
// The tile is the one free variable left, and it is the right one to spend because
// it changes how many keys a single iteration stages and nothing about the
// arithmetic. `head_dim` 256 at the benchmark's tile cap of 64 asks
// 256 + 64 + 2*64*257 = 33216 floats = 132864 bytes, an sm_86 block will opt into
// 101376, and `cudaFuncSetAttribute` answers that with cudaErrorInvalidValue --
// through CUDA_GO, which is exit(1). So the shape its own width predicate admits
// ended the process rather than running. A tile of 48 asks 99904 bytes there and
// runs, and the parity table is what decides whether a run at a narrower tile is
// allowed to count, not this function.
//
// A TILE OF 1 is a correct configuration and not a degenerate one, which is why it
// is the floor rather than an arbitrary cut. The QK pass hands each thread whole
// keys -- `for (i = c; i < n; i += dim)` with `n = min(tile, last - j0)` -- so a
// tile narrower than the block simply leaves the rest of the warp out of that loop,
// and every thread still reads all n of its own query's scores out of `ss` after
// it. The online softmax's rescale is exact at any tile size: at n = 1 the
// correction is exp(mrun - max(mrun, score)), which is 1 whenever the new score is
// not the running max and exp(negative) otherwise. That is the flash recurrence
// run one key at a time, and it is slow, which is a different problem from being
// wrong. Below 1 there is nothing left to try, so that is a refusal and not a
// clamp. `attn.cu` grades it rather than taking it on trust:
// `ATTN_MAX_TILE=1` runs the whole parity table with every key in its own tile.
//
// The scan steps down one tile at a time, which is at most 64 trivial products on a
// host that is about to wait on a device. Solving the linear inequality instead
// would be fewer characters to write and not fewer to trust.
static int zt_attn_fit_tile(int dim, int max_tile, int group_q, size_t limit, size_t *need) {
    const int top = zt_attn_tile(dim, max_tile);
    if (top < 1) {
        *need = 0;
        return 0;
    }
    for (int tile = top; tile >= 1; --tile) {
        const size_t b = zt_attn_forward_shmem(dim, tile, group_q);
        if (b <= limit) {
            *need = b;
            return tile;
        }
    }
    *need = zt_attn_forward_shmem(dim, 1, group_q);
    return 0;
}

// The same question for the backward's kernel A, which stages a different layout
// and so gets its own scan rather than a parameter saying which layout it wanted.
// One scan each, calling the exported `zt_attn_*_shmem` above, is the whole reason
// this is not one function taking a bytes-at-tile callback: the shared-memory
// arithmetic in this file is computed in exactly one place on purpose, because a
// second copy of it that disagreed would be a silent overrun rather than a compile
// error. At `head_dim` 256 the two land on the same tile, 48, at 101120 bytes
// against the forward's 99904 -- the extra `2*dim` and `2*tile` this layout
// carries.
static int zt_attn_fit_tile_dq(int dim, int max_tile, size_t limit, size_t *need) {
    const int top = zt_attn_tile(dim, max_tile);
    if (top < 1) {
        *need = 0;
        return 0;
    }
    for (int tile = top; tile >= 1; --tile) {
        const size_t b = zt_attn_bwd_dq_shmem(dim, tile);
        if (b <= limit) {
            *need = b;
            return tile;
        }
    }
    *need = zt_attn_bwd_dq_shmem(dim, 1);
    return 0;
}

// Past 48 KiB a block must ask the device for the memory, and the ask is per
// KERNEL: one call for the forward, another for each of the two backward kernels,
// because `fusedAttnForward<BROKEN>` and its siblings are distinct symbols. Skipping
// the ask does not degrade anything, it fails, and it fails with
// cudaErrorInvalidValue from inside the runtime naming neither shared memory nor
// this line.
//
// Returns 1 rather than taking CUDA_GO's exit(1), and that is the one place in
// this file where a CUDA error is a refusal rather than a dead process. It has to
// be: this is the surface a training step calls, and a step that meets a device
// unwilling to hand out the memory needs a return code to branch on, not a
// vanished process. The caller has already bounded `bytes` by the device's own
// opt-in ceiling, so a failure here means the ceiling moved under the call.
static int zt_attn_optin(const void *fn, size_t bytes) {
    if (bytes <= 48u * 1024u) return 0;
    const cudaError_t e =
        cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)bytes);
    if (e == cudaSuccess) return 0;
    fprintf(stderr, "attn: FAIL the device would not grant %zu bytes of shared memory per block: %s\n",
            bytes, cudaGetErrorString(e));
    return 1;
}

// Forward. `out` is [T, n_heads*dim]. `group_q` is 1 for every published row.
int zt_attn_forward(const float *q, const float *k, const float *v, float *out, int T, int n_heads,
                    int n_kv_heads, int dim, int group_q, int max_tile, int q_offset, int n_keys,
                    cudaStream_t stream) {
    // The width check comes FIRST and it prints. It used to be a silent `return 1`
    // here as well as the printing one further down, and the silent one won: a
    // refused width produced a return code and no line at all, so the most common
    // bad argument reached a caller as a bare 1.
    if (!zt_attn_dim_ok(dim)) return zt_attn_refuse_dim(dim);
    if (n_heads <= 0 || n_kv_heads <= 0 || n_heads % n_kv_heads != 0) {
        fprintf(stderr, "zt_attn: %d heads over %d kv heads is not a group.\n", n_heads,
                n_kv_heads);
        return 1;
    }
    if (group_q < 1 || group_q > 64 || (size_t)dim * (size_t)group_q > 1024) {
        fprintf(stderr, "zt_attn: group_q %d at head_dim %d is not launchable.\n", group_q, dim);
        return 1;
    }
    // A decode call passes q_offset = pos and n_keys = however much cache exists. A
    // refusal here rather than a clamp, because the two ways this can be wrong are
    // both silent: n_keys below q_offset + T would mask every key away, and
    // q_offset below 0 would let a query see keys before it was generated.
    if (q_offset < 0 || n_keys < 1) {
        fprintf(stderr, "zt_attn: q_offset %d, n_keys %d is not launchable.\n", q_offset, n_keys);
        return 1;
    }
    if (!zt_attn_dim_ok(dim)) {
        fprintf(stderr, "zt_attn: head_dim %d is refused. It must be even and in [%d, %d].\n", dim,
                MIN_DIM, MAX_DIM);
        fprintf(stderr,
                "       Even, because the staged K and V rows are padded to a stride of dim + 1\n");
        fprintf(stderr,
                "       and that stride spreads a warp over 32 banks only while it is coprime\n");
        fprintf(stderr, "       with 32. A power of two is not required, and the old check that\n");
        fprintf(stderr,
                "       demanded one refused 96, 192 and 80 -- Phi-3, DeepSeek's MLA, and a\n");
        fprintf(stderr, "       family of 80-wide heads -- for nothing this file's loops read.\n");
        return 1;
    }
    if (max_tile < 1) {
        fprintf(stderr, "zt_attn: max_tile %d is not launchable. A tile of zero never advances\n",
                max_tile);
        fprintf(stderr, "       the key loop, and that loop holds a __syncthreads(), so the\n");
        fprintf(stderr, "       device would hang rather than return a wrong answer.\n");
        return 1;
    }
    // The tile is fitted to the device's own shared-memory ceiling before the
    // launch, and refused if nothing fits, rather than handed to
    // cudaFuncSetAttribute and discovered there. See zt_attn_fit_tile.
    const size_t limit = zt_attn_shmem_limit();
    if (limit == 0) {
        fprintf(stderr, "zt_attn: this device's opt-in shared-memory ceiling could not be read, so\n");
        fprintf(stderr, "       no tile can be sized against it. Refusing rather than guessing 48 KiB.\n");
        return 1;
    }
    size_t shmem = 0;
    const int tile = zt_attn_fit_tile(dim, max_tile, group_q, limit, &shmem);
    if (tile == 0) {
        fprintf(stderr,
                "zt_attn: head_dim %d with %d queries per block needs at least %zu bytes of shared\n",
                dim, group_q, shmem);
        fprintf(stderr,
                "       memory per block even at a tile of 1, and this device will not grant more\n");
        fprintf(stderr, "       than %zu. Narrowing the tile is the whole of the fix and it is spent.\n", limit);
        fprintf(stderr,
                "       Refusing rather than launching something that fails later with a less\n");
        fprintf(stderr, "       useful message.\n");
        return 1;
    }
    if (zt_attn_optin((const void *)fusedAttnForward<0>, shmem) != 0) return 1;
    const unsigned blocks_x = (unsigned)((T + group_q - 1) / group_q);
    dim3 grid(blocks_x, (unsigned)n_heads);
    dim3 block((unsigned)(dim * group_q));
    fusedAttnForward<0><<<grid, block, shmem, stream>>>(q, k, v, out, T, n_heads, n_kv_heads, dim,
                                                         tile, group_q, q_offset, n_keys);
    CUDA_GO(cudaGetLastError());
    return 0;
}

// Backward. `dq` is [T, n_heads*dim]; `dk` and `dv` are [T, n_kv_heads*dim].
//
// `dq`, `dk` and `dv` are ASSIGNED by the kernels, never accumulated into, so
// they need not be zeroed first -- kernel B grid is (T, n_kv_heads) blocks of
// `dim` threads, which is a one-to-one correspondence with dk elements. A caller
// that wants accumulation must therefore copy out and add, which is noted here
// because "the accumulators need zeroing" is the assumption a reader brings from
// every other kernel, and it is wrong for this one.
//
// The two kernels run back to back on `stream`, and kernel B reads the per-row
// scalars kernel A writes. Same-stream ordering is what makes that safe; there is
// no explicit synchronisation and none is needed on one stream.
int zt_attn_backward(const float *q, const float *k, const float *v, const float *dout, float *dq,
                     float *dk, float *dv, float *row_max, float *row_den, float *row_del, int T,
                     int n_heads, int n_kv_heads, int dim, int max_tile, cudaStream_t stream) {
    // The width check first, and printing, for the reason the forward's copy of it
    // gives: the silent one used to sit in front of this one and swallow every
    // message.
    if (!zt_attn_dim_ok(dim)) return zt_attn_refuse_dim(dim);
    if (n_heads <= 0 || n_kv_heads <= 0 || n_heads % n_kv_heads != 0) {
        fprintf(stderr, "zt_attn: %d heads over %d kv heads is not a group.\n", n_heads,
                n_kv_heads);
        return 1;
    }
    if (max_tile < 1) {
        fprintf(stderr, "zt_attn: max_tile %d is not launchable. A tile of zero never advances\n",
                max_tile);
        fprintf(stderr, "       the key loop, and that loop holds a __syncthreads(), so the\n");
        fprintf(stderr, "       device would hang rather than return a wrong answer.\n");
        return 1;
    }
    // Both kernels are fitted to the device's ceiling before the launch, and
    // refused if nothing fits. See zt_attn_fit_tile and zt_attn_fit_tile_dq.
    const size_t limit = zt_attn_shmem_limit();
    if (limit == 0) {
        fprintf(stderr, "zt_attn: this device's opt-in shared-memory ceiling could not be read, so\n");
        fprintf(stderr, "       no tile can be sized against it. Refusing rather than guessing 48 KiB.\n");
        return 1;
    }
    size_t shmemA = 0;
    const int tile = zt_attn_fit_tile_dq(dim, max_tile, limit, &shmemA);
    if (tile == 0) {
        fprintf(stderr,
                "zt_attn: head_dim %d needs at least %zu bytes of shared memory per block even at a\n",
                dim, shmemA);
        fprintf(stderr,
                "       tile of 1, and this device will not grant more than %zu. Narrowing the tile is\n",
                limit);
        fprintf(stderr, "       the whole of the fix and it is spent. Refusing rather than launching\n");
        fprintf(stderr, "       something that fails later with a less useful message.\n");
        return 1;
    }
    // Kernel B stages one k and one v row and no tile, so there is no tile to
    // narrow and no case where it silently under-asks: 2*dim floats is 2048 bytes
    // at the widest head_dim this file admits, and the opt-in below is the only
    // thing that can refuse it.
    const size_t shmemB = zt_attn_bwd_dkdv_shmem(dim);
    if (zt_attn_optin((const void *)attnDqKernel<0>, shmemA) != 0) return 1;
    if (zt_attn_optin((const void *)attnDkDvKernel<0>, shmemB) != 0) return 1;
    dim3 gridA((unsigned)T, (unsigned)n_heads);
    dim3 gridB((unsigned)T, (unsigned)n_kv_heads);
    dim3 blockA((unsigned)dim);
    dim3 blockB((unsigned)dim);
    attnDqKernel<0><<<gridA, blockA, shmemA, stream>>>(q, k, v, dout, dq, row_max, row_den, row_del,
                                                       T, n_heads, n_kv_heads, dim, tile);
    attnDkDvKernel<0><<<gridB, blockB, shmemB, stream>>>(q, k, v, dout, row_max, row_den, row_del, dk,
                                                         dv, T, n_heads, n_kv_heads, dim);
    CUDA_GO(cudaGetLastError());
    return 0;
}

} // extern "C"
