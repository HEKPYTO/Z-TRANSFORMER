//! Generation: one token at a time, through a KV cache, with the logits read back
//! out and the next token chosen from them.
//!
//! WHY THIS FILE EXISTS. A training step hands `model.forwardWith` a whole window,
//! so the forward recomputes every position's keys and values from scratch and the
//! entire sequence is present at once. Generation cannot do that: the token at
//! position `t` does not exist until the logits at `t - 1` have been sampled, so
//! tokens arrive one at a time and each step needs the keys and values of every
//! position before it. That is what `kv_cache.Cache` keeps, and this is the caller
//! its header said was missing -- before this, nothing in the tree had read or
//! written a cached row at all.
//!
//! THE THREE THINGS THAT ARE SILENT WHEN WRONG. None of them crash, all three return
//! a plausible tensor, and that is why `decode_test.zig` checks the finished
//! `generate` against a full forward pass rather than any of them directly.
//!
//!   1. RoPE at position 0 instead of `pos`. `model.zig` rotates at absolute
//!      position 0 because row r of a training batch IS position r. Here the only
//!      row is position `pos`. Left at 0, every generated token is told it is the
//!      first one, and the model emits a stream that ignores its own history while
//!      looking entirely healthy.
//!   2. The wrong number of cached keys. A step at `pos` sees keys `0..=pos` and
//!      nothing after. Fewer returns `v[0]` with no error -- `attention_test.zig`
//!      pins exactly what a one-key prefix looks like -- so a decode loop carrying
//!      the bug reads as a working one that is fast, which is the worst shape a
//!      wrong answer takes.
//!   3. The key cached before rotation. `kv_cache.zig`'s header owns that trap and
//!      `Layer.append` cannot detect it, because a rotated key and an unrotated one
//!      are the same `[]const f32`. `step` below passes `k_pos` and never `k`.
//!
//! WHY THE ATTENTION LOOP IS A COPY. `attention.forward` reads a query's position
//! out of its own row index and refuses `k.rows != q.rows`, so a one-token query over
//! an n-token prefix has no spelling there. Padding q out to n rows with zeros until
//! the shape check passes computes n queries and keeps one, which makes every step
//! O(n^2) -- exactly the cost the cache exists to remove. So `attnStep` below is that
//! function's inner loop with the loop over query positions deleted: same f64
//! scores, same row maximum, same ascending accumulation. The change that would
//! retire it is a `q_offset` and an `n_keys` on `attention.zig`; `addInto` and
//! `tiedHead` are duplicated here for the same reason, both being private in
//! `model.zig`. Three private helpers, and this file does not own any of them.
//!
//! WHERE THE CUDA KERNEL FITS. `attn_kernels.cu`'s forward carries the two
//! parameters a decode step needs: `q_offset` is the absolute position of query
//! row 0 and `n_keys` is how many rows of k and v exist. `device.zig`'s
//! `Attn.forwardAt` takes both, plus a `t` that is the CALLER'S query row count
//! rather than the holder's buffer capacity. That third parameter is what made the
//! pair reachable: while `t` was the one number `init` sized all eleven buffers
//! from, query rows and key capacity were the same number, and a holder built at
//! `n_ctx` to hold the cache ran `n_ctx` query rows of which one was real. So for
//! the step at `pos` below the three are `q_offset = pos`, `n_keys = pos + 1`,
//! `t = 1`, on a holder built at `n_ctx`: `k` and `v` hold the whole cache and the
//! kernel walks the one query row that exists. `step` appends before it attends, so
//! the query's own key is already in the prefix at index `pos`, which is why
//! `n_keys` counts its own key too and is `pos + 1` rather than `pos`.
//!
//! ONE HOLDER PER `generate`, NOT ONE PER STEP, and that is the entire point of
//! the cache: eleven `cudaMalloc`s for the run and eleven `cudaFree`s at the end,
//! against eleven per step and the step attending to a single row. Nothing else on
//! this path is on the device, so every tensor still crosses the bus four times
//! per layer -- q, k, v up and the context down -- which is the PCIe floor
//! `src/cuda/README.md` measures and which is what says the rest of the step is
//! still the CPU. This is the first version of the path, not the fast one.
//!
//! WHAT GATES IT, AND WHAT DOES NOT. `model.cuda_attn` is the one constant, and it
//! chooses the arm of the seam in `step` below and the arm in `model.forwardWith`
//! together -- a tree with it false links no CUDA object at all, and `generate` is
//! the byte-for-byte loop it was. `zig build cuda-attn-check` does NOT reach this
//! file: that step is rooted at `src/train.zig` rather than `src/tests.zig`, on
//! purpose, so `decode_test.zig`'s CUDA test is not in it and skips in every build
//! in the graph. The command that runs it is written on that test.
//!
//! SAMPLING IS GREEDY AND NOTHING ELSE. `argmax`, ties to the lowest id, no
//! temperature, no top-k, no RNG. Greedy is the whole of it because it is the only
//! rule that makes a test deterministic, and a temperature needs a distribution to
//! divide by, which this file has no other use for. `Tokenizer.decode` turns the ids
//! below into text; nothing here knows a token as anything but a `u32`.

const std = @import("std");
const model = @import("model.zig");
const norm = @import("norm.zig");
const rope = @import("rope.zig");
const mlp = @import("mlp.zig");
const kv_cache = @import("kv_cache.zig");
const tensor = @import("tensor.zig");

// The CUDA half of the attention seam. Container scope because `@import` is, and
// for the reason `model.zig` puts its own copy there rather than one further down:
// an `extern fn` becomes a link-time requirement the moment the function CALLING it
// is analysed, and `model.zig`'s `cuda_attn` comment is the argument for why that
// has to be a comptime constant here as well.
const device = @import("cuda/device.zig");

const Tensor = tensor.Tensor;

/// Greedily generate `n_new` tokens after `prompt` and return them, owned by
/// `allocator`.
///
/// `prompt` must hold at least one id: the first generated token is the argmax of
/// the logits at the LAST prompt position, and with an empty prompt there is no
/// last position and nothing to put token 0 in front of. Refused rather than
/// invented, because there is no BOS id in this repository to invent it from.
///
/// Returns `error.SequenceTooLong` if `prompt.len + n_new` exceeds `n_ctx`, checked
/// before the cache is built: the cache would refuse the same run one step later
/// with the same answer, having already spent the prefill. `error.TokenOutOfRange`
/// for an id the embedding table cannot be indexed with, and `error.DimensionMismatch`
/// for parameters that were built from a different `model.Config` -- the same three
/// checks `model.forwardWith` makes, and for the same reason, which is that a
/// mismatch here reads out of bounds rather than reporting itself.
pub fn generate(
    allocator: std.mem.Allocator,
    p: model.Params,
    cfg: model.Config,
    prompt: []const u32,
    n_new: usize,
) ![]u32 {
    try model.validate(cfg);
    if (p.tok_embed.rows != cfg.vocab_size or p.tok_embed.cols != model.dModel(cfg))
        return error.DimensionMismatch;
    if (p.layers.len != cfg.n_layers) return error.DimensionMismatch;
    if (prompt.len == 0) return error.EmptyPrompt;
    if (prompt.len + n_new > cfg.n_ctx) return error.SequenceTooLong;

    var cache = try kv_cache.Cache.init(
        allocator,
        cfg.n_layers,
        cfg.n_ctx,
        cfg.n_kv_heads * cfg.head_dim,
    );
    defer cache.deinit(allocator);

    // One device holder for the WHOLE generation rather than one per step, which is
    // the same argument `model.forwardWith` makes one layer at a time and it is the
    // same argument one level up: the shape is fixed for the run, `cudaMalloc`
    // synchronises, and a step attends to a single row. Eleven allocations for the
    // run, against eleven per step over a cache that exists precisely so each step
    // does as little as possible.
    //
    // Sized at `cfg.n_ctx`, NOT at the prefix. `k` and `v` have to hold every row
    // the run will ever write, and `t` -- the QUERY row count -- is a separate
    // argument to `forwardAt` now, so nothing is wasted on the `q` and `out`
    // buffers past row 0. Sizing it at the first step's prefix instead would make
    // every later upload walk off the end of `k` and `v`.
    //
    // The `comptime` is on both lines, initialiser and `defer`, for the reason and
    // in the words `model.forwardWith` gives: as a runtime `if (dev)` the payload
    // is `cudaFree`, and that is what takes `zig build test` down at the link step
    // on every host with no CUDA toolchain.
    var dev: ?device.Attn = if (comptime model.cuda_attn)
        try device.Attn.init(cfg.n_ctx, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim)
    else
        null;
    defer if (comptime model.cuda_attn) dev.?.deinit();
    // Comptime-null on the CPU path, and Zig folds `if` over a comptime-known
    // optional, so a CPU build never reaches the payload. Same reliance
    // `model.forwardWith` states about its own `dev`.
    const dev_ptr: ?*device.Attn = if (dev) |*d| d else null;

    var out = try allocator.alloc(u32, n_new);
    errdefer allocator.free(out);

    // The prompt goes in one token at a time, through the same step the generated
    // tokens use. Prefilling through one batched `forwardWith` and copying keys out
    // of a `Sink` would be faster and would also make the test in `decode_test.zig`
    // compare the prompt's positions against themselves.
    //
    // `logits` is carried rather than dropped because out[0] is the argmax of the
    // LAST prompt position's logits; discarding them and re-stepping that token
    // would be a second forward pass per call.
    var logits = try step(allocator, p, cfg, &cache, dev_ptr, prompt[0], 0);
    defer logits.deinit();
    for (prompt[1..], 1..) |t, i| {
        const next = try step(allocator, p, cfg, &cache, dev_ptr, t, i);
        logits.deinit();
        logits = next;
    }

    var n: usize = 0;
    while (n < n_new) : (n += 1) {
        out[n] = @intCast(argmax(logits.rowConst(0)));
        // out[n] lands at `prompt.len + n`, and it is out[n]'s own step that produces
        // out[n + 1], so this loop runs n_new - 1 steps and not n_new: the token
        // sampled last is never fed anywhere. The highest position written is
        // therefore `prompt.len + n_new - 2` and the bound above leaves two to spare,
        // which costs two rows of a cache and buys a bound that reads as what it
        // says -- the prompt and everything after it, inside n_ctx.
        if (n + 1 < n_new) {
            const next = try step(allocator, p, cfg, &cache, dev_ptr, out[n], prompt.len + n);
            logits.deinit();
            logits = next;
        }
    }
    return out;
}

/// Token `tok` at absolute position `pos`: one whole block, once, with the cache.
///
/// Returns the [1, vocab] logits for that token. Every intermediate is a single row,
/// which is what makes the arithmetic here `model.forwardWith`'s arithmetic on one
/// row of it -- the residual stream is built and read per row by every op in it, so
/// the other positions in a batched forward do not touch this one.
///
/// `dev` is `null` in every build where `model.cuda_attn` is false, and the CUDA arm
/// below is not analysed in those builds. `generate` holds one holder across the
/// whole run and hands the same pointer to every step.
fn step(
    allocator: std.mem.Allocator,
    p: model.Params,
    cfg: model.Config,
    cache: *kv_cache.Cache,
    dev: ?*device.Attn,
    tok: u32,
    pos: usize,
) !Tensor {
    const d = model.dModel(cfg);
    // Widened first: `tok` is u32 from a prompt and `tok_embed.rowConst` would take
    // it as a usize, so a corrupt id out of range would arrive in bounds.
    const v: usize = tok;
    if (v >= cfg.vocab_size) return error.TokenOutOfRange;

    var x = try Tensor.init(allocator, 1, d);
    defer x.deinit();
    @memcpy(x.row(0), p.tok_embed.rowConst(v));
    // Ping-ponged with the stream exactly as `model.forwardWith` does, and both
    // names are freed at the point of the return rather than one buffer twice: a
    // `defer` reads its variable when the scope exits and the swap moves the buffers
    // between the two names, so {x, acc} holds the same two allocations whichever
    // side of an odd or even number of swaps this ends on.
    var acc = try Tensor.init(allocator, 1, d);
    defer acc.deinit();

    for (p.layers, 0..) |l, i| {
        var attn_in = try norm.forward(allocator, x, l.attn_norm);
        defer attn_in.deinit();
        var q = try tensor.matmul(attn_in, l.wq);
        defer q.deinit();
        var k = try tensor.matmul(attn_in, l.wk);
        defer k.deinit();
        var v_proj = try tensor.matmul(attn_in, l.wv);
        defer v_proj.deinit();

        // `pos`, not 0. This is the "sampling with a cache brings its own position"
        // that `model.forwardWith`'s own comment defers to.
        var q_pos = try rope.forward(allocator, q, pos, model.rope_theta, cfg.head_dim);
        defer q_pos.deinit();
        var k_pos = try rope.forward(allocator, k, pos, model.rope_theta, cfg.head_dim);
        defer k_pos.deinit();

        // `k_pos` and never `k`: the cache stores rotated keys, and rotating a
        // second time composes to R(2 * pos) rather than R(pos), which is wrong
        // without crashing. `v` is stored as projected -- nothing rotates it.
        //
        // `Layer.append` rather than `Cache.appendAll`: layer i's attention reads
        // layer i's rows and nothing else, so a position is per layer and there is
        // no cross-layer advance to make atomic. The one thing `appendAll`'s
        // all-or-nothing pass buys is a refusal mid-position, and `Error.Full` is
        // unreachable because `generate` bounded the run by `n_ctx` before the cache
        // existed.
        try cache.layers[i].append(k_pos.rowConst(0), v_proj.rowConst(0));

        // THE SEAM. Both arms compute the same function over the same filled prefix
        // and return the same shape; only `comptime cuda_attn` chooses, for the
        // reason `model.zig`'s own comment on that constant gives. `q_pos` is
        // already rotated at `pos` and the cache already holds this token's row at
        // index `pos`, so neither arm is handed a position -- both derive it from
        // the filled prefix, and `cudaAttnStep`'s comment says how.
        var ctx = if (comptime model.cuda_attn)
            try cudaAttnStep(allocator, dev.?, q_pos, &cache.layers[i], cfg)
        else
            try attnStep(allocator, q_pos, &cache.layers[i], cfg);
        defer ctx.deinit();
        var proj = try tensor.matmul(ctx, l.wo);
        defer proj.deinit();
        try addInto(&acc, x, proj);
        const after_attn = x;
        x = acc;
        acc = after_attn;

        var mlp_in = try norm.forward(allocator, x, l.mlp_norm);
        defer mlp_in.deinit();
        var ff = try mlp.forward(allocator, mlp_in, l.w_gate, l.w_up, l.w_down);
        defer ff.deinit();
        try addInto(&acc, x, ff);
        const after_mlp = x;
        x = acc;
        acc = after_mlp;
    }

    // Every layer advanced one position, so the counter the layers share follows
    // them. Nothing above reads it -- `attnStep` asks the layer it is handed, which
    // is the point -- but a caller that inspected the cache afterwards must not find
    // it still reading the length the cache was constructed with.
    cache.len = pos + 1;

    var final_h = try norm.forward(allocator, x, p.final_norm);
    defer final_h.deinit();
    return tiedHead(allocator, p, final_h.rowConst(0));
}

/// One query row over a layer cache's filled prefix, as [1, n_heads * head_dim].
///
/// `l.visible()` is the key count and nothing is passed alongside it. `step` appended
/// this token's row before calling, so the count is `pos + 1`: every key a causal
/// query at `pos` may see, and every key that exists. The loop bound IS the causal
/// mask here -- the query sits at `n_keys - 1` and the bound is `s < n_keys` -- so
/// there is no `-inf` term and none is needed.
///
/// `attention.forward` cannot be called for this and the header says why; the
/// arithmetic below is its inner loop with the query loop removed.
///
/// PUBLIC because it is the CPU twin of `cudaAttnStep` and the only thing that can
/// grade it. `src/train.zig`'s gate reads `attention.forward` -- the CPU
/// implementation the CUDA arm is a claim about -- rather than a second copy of it,
/// because a comparison against a re-typed loop only proves the copy is stable. The
/// same argument puts this one in reach of `decode_test.zig`.
///
/// That argument only holds if this function refuses what it cannot compute, so it
/// carries the same three refusals as `cudaAttnStep` below plus one of its own. Every
/// loop below is bounded by `n_keys` or by `dim`, and an empty prefix skips all of
/// them: `row_max` keeps its `-inf`, `denom` stays `0`, no division runs, and `out`
/// reads back the zeros `Tensor.init` gave it. That is a plausible zero context and
/// no error, which is the same shape as the `head_dim == 0` bug `attentionBackward`
/// carried for a while -- where nothing was written at all and the tensor read back
/// as its initialisation. Not reachable from `step`, which appends before it attends;
/// reachable from any other caller, which is why `attnStep` is `pub`.
pub fn attnStep(
    allocator: std.mem.Allocator,
    q: Tensor,
    l: *const kv_cache.Layer,
    cfg: model.Config,
) !Tensor {
    const dim = cfg.head_dim;
    const q_cols = cfg.n_heads * dim;
    const kv_cols = cfg.n_kv_heads * dim;
    // Mirrors `cudaAttnStep`'s two, in the same order, so a shape one refuses the
    // other refuses too rather than the CPU arm silently answering what the GPU arm
    // refused to.
    if (cfg.n_kv_heads == 0 or dim == 0 or cfg.n_heads % cfg.n_kv_heads != 0) {
        return error.InvalidHeadConfig;
    }
    if (q.cols != q_cols or l.k.cols != kv_cols or l.v.cols != kv_cols) {
        return error.DimensionMismatch;
    }
    const group = cfg.n_heads / cfg.n_kv_heads;
    const n_keys = l.visible();
    // The one this arm needs and the device one does not: `cudaAttnStep` gets `n_keys`
    // from the caller and hands it to a kernel that checks it, while here `n_keys` IS
    // the loop bound, so an empty prefix is a zero-length reduction rather than a
    // refusal. Refusing it is what stops an unwritten `out` from reading as an answer.
    if (n_keys < 1) return error.DimensionMismatch;
    // Scores stay f64 and narrow to f32 only in the store, for `attention.forward`'s
    // reason: an fp32 sum over the prefix drops bits the softmax denominator and the
    // weighted sum of v are both built from.
    const scale = 1.0 / @sqrt(@as(f64, @floatFromInt(dim)));

    var out = try Tensor.init(allocator, 1, cfg.n_heads * dim);
    errdefer out.deinit();
    // Not zeroed and not reused across steps, unlike the training path's row: this is
    // `n_keys` floats per head and every one of them is written below.
    var scores = try allocator.alloc(f64, n_keys);
    defer allocator.free(scores);

    for (0..cfg.n_heads) |h| {
        const kv = h / group;
        const q_head = q.rowConst(0)[h * dim ..][0..dim];
        // The maximum cancels against the denominator, so subtracting it leaves the
        // softmax unchanged while keeping every exp in range.
        var row_max: f64 = -std.math.inf(f64);
        for (0..n_keys) |s| {
            const k_head = l.k.rowConst(s)[kv * dim ..][0..dim];
            var dot: f64 = 0;
            for (q_head, k_head) |a, b| dot += @as(f64, a) * @as(f64, b);
            const score = dot * scale;
            scores[s] = score;
            row_max = @max(row_max, score);
        }
        var denom: f64 = 0;
        for (0..n_keys) |s| {
            scores[s] = @exp(scores[s] - row_max);
            denom += scores[s];
        }
        for (0..n_keys) |s| scores[s] /= denom;

        const dst = out.row(0);
        for (0..dim) |j| {
            var a: f64 = 0;
            for (0..n_keys) |s| a += scores[s] * @as(f64, l.v.rowConst(s)[kv * dim + j]);
            dst[h * dim + j] = @floatCast(a);
        }
    }
    return out;
}

/// One query row over a layer cache's filled prefix, on `src/cuda/attn_kernels.cu`.
///
/// `attnStep` on the device, same argument list, so the seam in `step` is a swap of
/// one name rather than of a call's shape. Public for the reason `attnStep` is:
/// `decode_test.zig` grades this against it, and `model.zig`'s `cudaForward` is
/// reachable only through `model.forwardWith` because there is a model behind it --
/// there is not one here, so this function is the seam and has to be nameable.
///
/// It needs no guard of its own, for `model.cudaForward`'s reason: the call inside
/// `step` is `if (comptime cuda_attn)`, so nothing here is analysed on a CPU build
/// and `zt_attn_forward` never reaches an object that links no CUDA runtime.
pub fn cudaAttnStep(
    allocator: std.mem.Allocator,
    a: *device.Attn,
    q: Tensor,
    l: *const kv_cache.Layer,
    cfg: model.Config,
) !Tensor {
    const dim = cfg.head_dim;
    const q_cols = cfg.n_heads * dim;
    const kv_cols = cfg.n_kv_heads * dim;
    // The same count `attnStep` derives, and for the same reason: `step` appended
    // this token's row before calling either arm.
    const n_keys = l.visible();

    // `model.cudaForward`'s shape contract, restated rather than assumed, for its
    // reason: `upload` bounds the SOURCE against the device buffer and knows
    // nothing of what a column MEANS, so a `q` or a cache narrower than the head
    // layout names passes every check below and silently attends to a truncated
    // head.
    if (cfg.n_kv_heads == 0 or dim == 0 or cfg.n_heads % cfg.n_kv_heads != 0) {
        return error.InvalidHeadConfig;
    }
    if (q.cols != q_cols or l.k.cols != kv_cols or l.v.cols != kv_cols) {
        return error.DimensionMismatch;
    }

    // Row 0 of `q` only, and `n_keys` rows of the cache. A cache longer than the
    // holder is caught by `upload` rather than by a check here: `upload` compares
    // against the buffer's real byte count, which is the number that matters, and
    // `forwardBounds` refuses `n_keys` above `c_t` a line later anyway.
    try a.upload(a.q, q.data[0..q_cols]);
    try a.upload(a.k, l.k.data[0 .. n_keys * kv_cols]);
    try a.upload(a.v, l.v.data[0 .. n_keys * kv_cols]);

    // THE THREE NUMBERS, and each one is load-bearing.
    //
    // `q_offset` is the query's absolute position, and it is `n_keys - 1` rather
    // than a `pos` handed in. `step` appends this token's own row before it
    // attends, so the filled prefix ends at `pos` and the two ARE the same
    // number -- which is what makes the pairing `forwardBounds` enforces, `q_offset
    // + t <= n_keys`, true by construction instead of by a caller remembering.
    // The kernel adds `q_offset` to every key index, so a `q_offset` left at 0
    // makes every position after the first attend to key 0 alone, which is `v[0]`
    // with no error -- the exact failure `attnStep`'s own header calls the worst
    // shape a wrong answer takes.
    //
    // `n_keys` is the whole filled prefix, never `a.c_t`: rows past it are whatever
    // the previous step's upload happened to leave, and a step whose prefix is
    // shorter than the cache would read them as keys.
    //
    // `t` is 1, the caller's QUERY row count and the reason this function can be
    // written at all. The holder's eleven buffers are sized from the `n_ctx` it was
    // built with, and `t` is not that number: `q` and `out` hold `n_ctx` rows and
    // row 0 is the one that exists. `group_q` 1 and a cap of 64 are the
    // configuration every published row in `src/cuda/README.md` was measured at,
    // and `zt_attn_tile` is `min(dim, max_tile)`, so 64 reproduces that table at
    // head_dim 32 and at Llama-3's 128.
    try a.forwardAt(1, 64, try device.toCInt(n_keys - 1), try device.toCInt(n_keys), 1);

    var out = try Tensor.init(allocator, 1, q_cols);
    errdefer out.deinit();
    // This copy is what waits for the kernel, and that is load-bearing rather than
    // incidental, for `model.cudaForward`'s reason restated: the launcher returns as
    // soon as the launch is accepted, and a pageable device-to-host `cudaMemcpy` is
    // the synchronising point. Moving the launch off the default stream would make
    // this a race and the result a wrong answer rather than an error.
    try a.download(out.data, a.out);
    return out;
}

/// `x = base + branch` in place, one row at a time. `model.zig`'s copy, which is
/// private. The width check is kept because a `wo` or `w_down` narrower than the
/// stream would otherwise be read past its own row.
fn addInto(out: *Tensor, base: Tensor, branch: Tensor) !void {
    if (out.cols != base.cols or out.cols != branch.cols) return error.DimensionMismatch;
    for (0..out.rows) |r| {
        const b = base.rowConst(r);
        const a = branch.rowConst(r);
        const dst = out.row(r);
        for (0..out.cols) |i| dst[i] = b[i] + a[i];
    }
}

/// Logits for one hidden row, as [1, vocab], off the tied embedding.
///
/// `model.zig`'s `tiedHead` is private and its two public entry points take a whole
/// sequence, rotated at position 0 -- the one thing a decode step cannot use. The f64
/// accumulator walking the embedding in ascending order is that function's, so the
/// sum here is the same sum in the same order and the two agree on every element
/// rather than merely agreeing to a tolerance.
fn tiedHead(allocator: std.mem.Allocator, p: model.Params, hidden: []const f32) !Tensor {
    var out = try Tensor.init(allocator, 1, p.tok_embed.rows);
    const dst = out.row(0);
    for (0..p.tok_embed.rows) |v| {
        const e = p.tok_embed.rowConst(v);
        var acc: f64 = 0;
        for (hidden, e) |h, ei| acc += @as(f64, h) * @as(f64, ei);
        dst[v] = @floatCast(acc);
    }
    return out;
}

/// The largest logit, ties to the LOWEST id because the comparison is strict.
///
/// Strict is the whole of the determinism argument: `>=` would make the winner depend
/// on the order the loop happened to visit, which is ascending today and is a fact
/// about the loop rather than a promise about the result.
fn argmax(logits: []const f32) usize {
    var best: usize = 0;
    for (logits, 0..) |l, i| {
        if (l > logits[best]) best = i;
    }
    return best;
}
