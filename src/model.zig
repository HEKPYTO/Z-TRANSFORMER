//! GPT-mini assembly: pre-norm blocks of RMSNorm, RoPE, GQA and SwiGLU, with tied embeddings.
const std = @import("std");
const tensor = @import("tensor.zig");
const norm = @import("norm.zig");
const rope = @import("rope.zig");
const mlp = @import("mlp.zig");
const attention = @import("attention.zig");
const Tensor = tensor.Tensor;

/// Public because the parity exporter writes it into the config the
/// external side builds its reference model from. A harness that hardcoded
/// 500000 next to a model that changed it would fail on RoPE and read as a
/// numerics bug rather than as a stale number.
pub const rope_theta: f64 = 500000;
const init_stddev: f64 = 0.02;

/// One intermediate the forward pass hands to a `Sink`, named for what it is
/// rather than for the local it is bound to.
///
/// `attn_probs` and the SwiGLU hidden state are the two the design names and
/// this enum does not carry, because `attention.forward` and `mlp.forward`
/// compute them internally and return only the reduced result. Reaching either
/// means changing those two files, so the parity harness gates the fourteen
/// this pass can hand over and says so in its own output.
pub const Name = enum {
    attn_norm_out,
    q,
    k,
    v,
    q_rope,
    k_rope,
    attn_ctx,
    attn_proj,
    residual1,
    mlp_norm_out,
    mlp_out,
    residual2,
    final_norm,
    logits,
};

/// A callback the forward pass hands each intermediate to as it is produced.
///
/// A function pointer rather than a bag of tensor pointers, because the
/// intermediates live in buffers the pass frees before it returns: a pointer
/// recorded during the pass would dangle by the time a reader got to it. A
/// callback fires while the tensor is still valid, so the reader copies out
/// whatever it needs at that moment.
///
/// Nothing in the pass reads the sink except to call it, so a null sink runs
/// the same arithmetic in the same order and returns the same tensor.
pub const Sink = struct {
    /// `layer` is the 0-based layer index, and 0 for `final_norm` and
    /// `logits`, which are not part of any one layer.
    put: *const fn (sink: *Sink, name: Name, layer: usize, t: Tensor) void,
};

pub const Config = struct {
    n_layers: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    n_ctx: usize,
    vocab_size: usize,
    ffn_mult: usize,
};

pub fn dModel(cfg: Config) usize {
    return cfg.n_heads * cfg.head_dim;
}

pub fn ffnDim(cfg: Config) usize {
    return cfg.ffn_mult * dModel(cfg);
}

pub fn defaultConfig() Config {
    return .{
        .n_layers = 4,
        .n_heads = 4,
        .n_kv_heads = 2,
        .head_dim = 32,
        .n_ctx = 256,
        .vocab_size = 1024,
        .ffn_mult = 4,
    };
}

pub const Layer = struct {
    attn_norm: Tensor, // [d]
    wq: Tensor, // [d, n_heads * head_dim]
    wk: Tensor, // [d, n_kv_heads * head_dim]
    wv: Tensor, // [d, n_kv_heads * head_dim]
    wo: Tensor, // [n_heads * head_dim, d]
    mlp_norm: Tensor, // [d]
    w_gate: Tensor, // [d, h]
    w_up: Tensor, // [d, h]
    w_down: Tensor, // [h, d]
};

pub const Params = struct {
    tok_embed: Tensor, // [vocab, d]
    layers: []Layer,
    final_norm: Tensor, // [d]

    pub fn deinit(self: *Params) void {
        for (self.layers) |*l| freeLayer(l);
        // Every tensor was built from one allocator, and reading it back off
        // tok_embed is how deinit frees the slice itself without Params having
        // to carry the allocator as a second source of truth.
        const allocator = self.tok_embed.allocator;
        allocator.free(self.layers);
        self.layers = &.{};
        self.tok_embed.deinit();
        self.final_norm.deinit();
    }
};

/// Every parameter a training run writes to, in a fixed order, so one seed
/// reproduces one set of weights exactly.
///
/// The nine norm weights are initialised to one, which is RMSNorm's identity
/// scale and not a special case: y = x / rms(x) * 1. Zero looks like the
/// quieter choice, and it is a trap. A zero norm weight makes its branch output
/// exactly zero, so every branch contributes nothing at step 0, so the loss is
/// flat in all 28 projection tensors, and d(loss)/d(tok_embed) is zero too
/// because the zero final_norm severs the path back to the embedding. Only the
/// nine norm weights would receive gradient, and AdamW's 0 / (0 + 1e-8) then
/// pins the other 29 at their step 0 values for another step, so a run that
/// starts at zero spends its first steps moving the norm gains and nothing
/// else. Ones leave every parameter with a real gradient from the first step.
pub fn initParams(allocator: std.mem.Allocator, cfg: Config, seed: u64) !Params {
    try validate(cfg);
    const d = dModel(cfg);

    // Xoshiro256++ named rather than reached through `std.Random.DefaultPrng`.
    // That alias resolves to whichever engine the standard library happens to
    // pick, and the standard library documents it as an implementation choice
    // rather than a promise, so a Zig upgrade could move these weights without
    // touching this file. Every published number in this project descends from
    // this stream, so the algorithm is part of the artifact and gets written
    // down. It is also what the alias resolves to today, so naming it changes
    // no bytes.
    var prng = std.Random.Xoshiro256.init(seed);
    const rnd = prng.random();

    // Assigned one field at a time, each with its own errdefer registered right
    // after it exists. A struct literal that fails part way through is never
    // assigned, so the tensors built before the failure would leak with nothing
    // holding a handle to them.
    var p: Params = undefined;
    p.tok_embed = try Tensor.init(allocator, cfg.vocab_size, d);
    errdefer p.tok_embed.deinit();
    p.layers = try allocator.alloc(Layer, cfg.n_layers);
    errdefer allocator.free(p.layers);
    p.final_norm = try Tensor.init(allocator, 1, d);
    errdefer p.final_norm.deinit();

    // `alloc` hands back uninitialised layers, so the unwind walks only the
    // prefix that was built. The count has to live outside the loop: an
    // errdefer inside the loop body belongs to that body, not to this function,
    // and would fire for the failing layer and drop every layer before it.
    var built: usize = 0;
    errdefer for (p.layers[0..built]) |*l| freeLayer(l);
    while (built < cfg.n_layers) : (built += 1) {
        p.layers[built] = try initLayer(allocator, cfg, rnd);
    }

    // GPT-2's scaled embedding init. The 1/sqrt(2 * n_layers) factor holds the
    // variance of the residual stream flat as depth grows, so the sums added by
    // later layers do not widen the stream out from under the final norm.
    const embed_scale = 1.0 / @sqrt(2.0 * @as(f64, @floatFromInt(cfg.n_layers)));
    for (p.tok_embed.data) |*v| v.* = @floatCast(normal(rnd, init_stddev) * embed_scale);
    p.final_norm.fill(1);
    return p;
}

/// Logits for `tokens` as [T, vocab]. Loss is `loss.forward`'s job, so this
/// returns the raw scores and nothing else.
///
/// Pre-norm, so the norm sits inside the residual branch and the stream only
/// ever grows:
///     h = x + attention(norm1(x))
///     y = h + mlp(norm2(h))
/// A post-norm block would fold each norm into the residual sum instead, which
/// is the older arrangement and the wrong one.
pub fn forward(allocator: std.mem.Allocator, p: Params, cfg: Config, tokens: []const u32) !Tensor {
    return forwardWith(allocator, p, cfg, tokens, null);
}

/// `forward`, with every intermediate handed to `sink` as it is produced.
///
/// The body moved here rather than being copied, so there is one forward pass
/// in the tree and the two entry points cannot drift apart. `forward` above is
/// this with no sink, which is the whole of the difference: the callback is
/// the only thing the pass does that the arithmetic does not already do.
pub fn forwardWith(
    allocator: std.mem.Allocator,
    p: Params,
    cfg: Config,
    tokens: []const u32,
    sink: ?*Sink,
) !Tensor {
    try validate(cfg);
    if (tokens.len > cfg.n_ctx) return error.SequenceTooLong;
    const d = dModel(cfg);
    const t_count = tokens.len;
    // The token bound and the tied head both index tok_embed by id, so a config
    // that disagrees with the parameters it was built from would panic on a
    // valid token rather than report the mismatch.
    if (p.tok_embed.rows != cfg.vocab_size or p.tok_embed.cols != d) return error.DimensionMismatch;
    // The layer count is the one shape the embedding cannot imply: a config
    // built for a shallower or deeper stack would pass the check above and then
    // run the depth it was handed, ignoring the parameters it was handed.
    if (p.layers.len != cfg.n_layers) return error.DimensionMismatch;

    var x = try Tensor.init(allocator, t_count, d);
    errdefer x.deinit();
    for (tokens, 0..) |tok, t| {
        // Token ids come from corpus data. Widening u32 to usize is lossless on
        // every Zig target, which is what keeps a corrupt id from wrapping down
        // into the valid range before the check.
        const v: usize = tok;
        if (v >= cfg.vocab_size) return error.TokenOutOfRange;
        @memcpy(x.row(t), p.tok_embed.rowConst(v));
    }

    // The residual target, ping-ponged with the stream. Two buffers instead of
    // one per layer, and the swap keeps the branch output a separate tensor so
    // the later backward pass has something to differentiate.
    //
    // x and acc are freed by hand on the success path, not by a defer each. A
    // defer reads the variable when the scope exits, and the swap below moves
    // the buffers between the two names, so what a defer captured at
    // registration is not what a later free would have to release. Reading both
    // names at the point of the free is correct whatever the loop did in
    // between, because a swap only ever exchanges the two: the pair
    // {x, acc} holds the same two tensors after an even number of swaps and
    // after an odd one, so there is no parity to keep track of and nothing to
    // get wrong on the next edit. The errdefers above cover the unwind, where
    // the same two names are read the same way.
    var acc = try Tensor.init(allocator, t_count, d);
    errdefer acc.deinit();

    const attn_cfg = attention.Config{
        .n_heads = cfg.n_heads,
        .n_kv_heads = cfg.n_kv_heads,
        .head_dim = cfg.head_dim,
    };

    for (p.layers, 0..) |l, layer| {
        var attn_in = try norm.forward(allocator, x, l.attn_norm);
        defer attn_in.deinit();
        if (sink) |s| s.put(s, .attn_norm_out, layer, attn_in);

        var q = try tensor.matmul(attn_in, l.wq);
        defer q.deinit();
        if (sink) |s| s.put(s, .q, layer, q);
        var k = try tensor.matmul(attn_in, l.wk);
        defer k.deinit();
        if (sink) |s| s.put(s, .k, layer, k);
        var v = try tensor.matmul(attn_in, l.wv);
        defer v.deinit();
        if (sink) |s| s.put(s, .v, layer, v);

        // RoPE sits between the projection and the attention, at absolute
        // position 0, because a training batch always starts there. Sampling
        // with a cache is a later phase and brings its own position.
        var q_pos = try rope.forward(allocator, q, 0, rope_theta, cfg.head_dim);
        defer q_pos.deinit();
        if (sink) |s| s.put(s, .q_rope, layer, q_pos);
        var k_pos = try rope.forward(allocator, k, 0, rope_theta, cfg.head_dim);
        defer k_pos.deinit();
        if (sink) |s| s.put(s, .k_rope, layer, k_pos);
        // v is not rotated, so it goes to attention as projected.

        var ctx = try attention.forward(allocator, q_pos, k_pos, v, attn_cfg);
        defer ctx.deinit();
        if (sink) |s| s.put(s, .attn_ctx, layer, ctx);
        var proj = try tensor.matmul(ctx, l.wo);
        defer proj.deinit();
        if (sink) |s| s.put(s, .attn_proj, layer, proj);

        try addInto(&acc, x, proj);
        const after_attn = x;
        x = acc;
        acc = after_attn;
        // After the swap `x` is the first residual. The next layer's swap
        // reuses this same buffer, so a sink that kept the pointer instead of
        // copying here would see layer n+1 overwrite layer n.
        if (sink) |s| s.put(s, .residual1, layer, x);

        var mlp_in = try norm.forward(allocator, x, l.mlp_norm);
        defer mlp_in.deinit();
        if (sink) |s| s.put(s, .mlp_norm_out, layer, mlp_in);
        var ff = try mlp.forward(allocator, mlp_in, l.w_gate, l.w_up, l.w_down);
        defer ff.deinit();
        if (sink) |s| s.put(s, .mlp_out, layer, ff);

        try addInto(&acc, x, ff);
        const after_mlp = x;
        x = acc;
        acc = after_mlp;
        if (sink) |s| s.put(s, .residual2, layer, x);
    }

    var final_h = try norm.forward(allocator, x, p.final_norm);
    defer final_h.deinit();
    if (sink) |s| s.put(s, .final_norm, 0, final_h);
    const logits = try tiedHead(allocator, final_h, p.tok_embed);
    if (sink) |s| s.put(s, .logits, 0, logits);

    // The success path frees here, where the two names can be read against the
    // swap above. The errdefers registered at allocation do not fire on this
    // return, so the two buffers are released exactly once.
    x.deinit();
    acc.deinit();
    return logits;
}

fn addInto(out: *Tensor, base: Tensor, branch: Tensor) !void {
    // A `wo` or `w_down` narrower than the stream would otherwise be read past
    // its own row, so the mismatch is reported instead.
    if (out.cols != base.cols or out.cols != branch.cols) return error.DimensionMismatch;
    for (0..out.rows) |r| {
        const b = base.rowConst(r);
        const a = branch.rowConst(r);
        const dst = out.row(r);
        for (0..out.cols) |i| dst[i] = b[i] + a[i];
    }
}

/// Weight-tied logits: logit[t][v] is the dot product of embedding row v with
/// the final hidden state at t. There is no lm_head, so walking v directly is
/// what saves the [d, vocab] transpose of tok_embed that `matmul` would need.
fn tiedHead(allocator: std.mem.Allocator, hidden: Tensor, tok_embed: Tensor) !Tensor {
    const d = hidden.cols;
    var out = try Tensor.init(allocator, hidden.rows, tok_embed.rows);

    for (0..hidden.rows) |t| {
        const h = hidden.rowConst(t);
        const dst = out.row(t);
        for (0..tok_embed.rows) |v| {
            const e = tok_embed.rowConst(v);
            // f64 accumulator, narrowed once per logit, for the reason
            // attention gives: a long embedding sum in f32 drops bits the
            // softmax in loss.forward then exponentiates.
            var acc: f64 = 0;
            for (0..d) |i| acc += @as(f64, h[i]) * @as(f64, e[i]);
            dst[v] = @floatCast(acc);
        }
    }
    return out;
}

fn validate(cfg: Config) !void {
    // A zero kv head count divides by zero in the group split, a zero head dim
    // makes the attention scale infinite, an odd head dim has no rotary pair,
    // and a model with no layer is a bare embedding table.
    if (cfg.n_layers == 0) return error.InvalidConfig;
    if (cfg.n_kv_heads == 0 or cfg.n_heads == 0 or cfg.head_dim == 0) return error.InvalidConfig;
    if (cfg.n_heads % cfg.n_kv_heads != 0) return error.InvalidConfig;
    if (cfg.head_dim % 2 != 0) return error.InvalidConfig;
    if (cfg.n_ctx == 0 or cfg.vocab_size == 0 or cfg.ffn_mult == 0) return error.InvalidConfig;
}

fn initLayer(allocator: std.mem.Allocator, cfg: Config, rnd: std.Random) !Layer {
    const d = dModel(cfg);
    const h = ffnDim(cfg);
    const kv_dim = cfg.n_kv_heads * cfg.head_dim;
    const shapes = [9][2]usize{
        .{ 1, d }, // attn_norm
        .{ d, d }, // wq
        .{ d, kv_dim }, // wk
        .{ d, kv_dim }, // wv
        .{ d, d }, // wo
        .{ 1, d }, // mlp_norm
        .{ d, h }, // w_gate
        .{ d, h }, // w_up
        .{ h, d }, // w_down
    };

    // The tensors land in an array first, not straight into a `Layer` literal.
    // A literal that fails part way through is never assigned at all, so an
    // errdefer on it would never be registered and every tensor drawn before the
    // failure would leak. Here the unwind frees the prefix that exists.
    var t: [9]Tensor = undefined;
    var built: usize = 0;
    errdefer for (t[0..built]) |*one| one.deinit();
    while (built < t.len) : (built += 1) {
        t[built] = try Tensor.init(allocator, shapes[built][0], shapes[built][1]);
    }

    // Indices 0 and 5 are the two norms, set to one so every parameter has a
    // gradient on the first step. The rest are drawn in field order, so one seed
    // walks one path.
    t[0].fill(1);
    t[5].fill(1);
    for ([_]usize{ 1, 2, 3, 4, 6, 7, 8 }) |i| fillNormal(&t[i], rnd);

    return .{
        .attn_norm = t[0],
        .wq = t[1],
        .wk = t[2],
        .wv = t[3],
        .wo = t[4],
        .mlp_norm = t[5],
        .w_gate = t[6],
        .w_up = t[7],
        .w_down = t[8],
    };
}

fn freeLayer(l: *Layer) void {
    l.attn_norm.deinit();
    l.wq.deinit();
    l.wk.deinit();
    l.wv.deinit();
    l.wo.deinit();
    l.mlp_norm.deinit();
    l.w_gate.deinit();
    l.w_up.deinit();
    l.w_down.deinit();
}

fn fillNormal(t: *Tensor, rnd: std.Random) void {
    for (t.data) |*v| {
        // The narrowing to f32 is the parameter's storage type, not a silent
        // cast: the model is f32 end to end and the scale factor is well inside
        // range, so the value the optimizer updates is the value that was drawn.
        v.* = @floatCast(normal(rnd, init_stddev));
    }
}

/// Box-Muller normal deviate. `1 - u` keeps the radius off zero, because a zero
/// radius is log(0) and would put -inf into the tensor. The second variate is
/// discarded rather than cached, so the draw count per parameter is one and the
/// stream depends on nothing but the seed and the loop order.
fn normal(rnd: std.Random, stddev: f64) f64 {
    const radius = 1.0 - rnd.float(f64);
    const phase = rnd.float(f64);
    return stddev * @sqrt(-2.0 * @log(radius)) * @cos(2.0 * std.math.pi * phase);
}
