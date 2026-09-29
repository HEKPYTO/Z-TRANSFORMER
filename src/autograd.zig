//! Hand-written reverse-mode gradients for the GPT-mini forward pass.
//!
//! One backward function per forward function, no tape and no graph nodes: the
//! pass walks the same ops in reverse and rebuilds each block's intermediates
//! once, from the layer input it already has, then consumes them. Every gradient
//! is therefore readable from the loss down to one tensor without jumping, and
//! checkable against a finite difference of the real forward.
const std = @import("std");
const model = @import("model.zig");
const tensor = @import("tensor.zig");
const norm = @import("norm.zig");
const rope = @import("rope.zig");
const mlp = @import("mlp.zig");
const attention = @import("attention.zig");
const Tensor = tensor.Tensor;

/// The same constant `model.zig` passes to `rope.forward`. RoPE sits between
/// the projection and the attention, and its inverse has to use the same angle.
const rope_theta: f64 = 500000;

pub const LayerGrads = struct {
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

pub const Grads = struct {
    tok_embed: Tensor, // [vocab, d]
    layers: []LayerGrads,
    final_norm: Tensor, // [d]

    pub fn deinit(self: *Grads) void {
        for (self.layers) |*l| freeLayerGrads(l);
        // Reading the allocator back off tok_embed is how the layer slice is
        // released without Grads carrying a second source of truth, the same
        // trade model.Params makes.
        const allocator = self.tok_embed.allocator;
        allocator.free(self.layers);
        self.layers = &.{};
        self.tok_embed.deinit();
        self.final_norm.deinit();
    }
};

/// A gradient set shaped exactly like `like`, filled with exact zeros. The
/// caller zeroes once and `backward` accumulates into it, so one buffer serves
/// a whole accumulation schedule.
pub fn zeroGrads(allocator: std.mem.Allocator, like: model.Params) !Grads {
    const d = like.tok_embed.cols;
    var g: Grads = undefined;
    g.tok_embed = try Tensor.init(allocator, like.tok_embed.rows, d);
    errdefer g.tok_embed.deinit();
    g.layers = try allocator.alloc(LayerGrads, like.layers.len);
    errdefer allocator.free(g.layers);
    g.final_norm = try Tensor.init(allocator, 1, d);
    errdefer g.final_norm.deinit();

    var built: usize = 0;
    errdefer for (g.layers[0..built]) |*l| freeLayerGrads(l);
    while (built < like.layers.len) : (built += 1) {
        g.layers[built] = try zeroLayerGrads(allocator, &like.layers[built]);
    }
    return g;
}

/// dLoss/dlogits for `loss.forward`, with the mean over rows already folded in:
///
///     dlogits[t][v] = (softmax(logits[t])[v] - (v == targets[t])) / T
///
/// The row max is pulled out from under the exponent for the reason
/// `loss.forward` gives, and the softmax is formed in f64 and narrowed once, so
/// a wide vocabulary does not lose the denominator to f32 rounding.
pub fn dLossDLogits(allocator: std.mem.Allocator, logits: Tensor, targets: []const u32) !Tensor {
    const t_count = logits.rows;
    if (t_count == 0) return error.EmptyBatch;
    if (targets.len != t_count) return error.TargetCountMismatch;
    const v_count = logits.cols;

    var out = try Tensor.init(allocator, t_count, v_count);
    errdefer out.deinit();
    const inv_t: f64 = 1.0 / @as(f64, @floatFromInt(t_count));

    for (targets, 0..) |target, i| {
        // targets come from corpus data, so this is a trust boundary. The
        // widening to usize is lossless on every Zig target, which is what
        // keeps a corrupt id from wrapping down into the valid range.
        const t: usize = target;
        if (t >= v_count) return error.TargetOutOfRange;

        const row = logits.rowConst(i);
        var max: f64 = @as(f64, row[0]);
        for (row[1..]) |z| max = @max(max, @as(f64, z));
        var denom: f64 = 0;
        for (row) |z| denom += @exp(@as(f64, z) - max);

        const dst = out.row(i);
        for (row, 0..) |z, v| {
            const p = @exp(@as(f64, z) - max) / denom;
            const onehot: f64 = if (v == t) 1.0 else 0.0;
            dst[v] = @floatCast((p - onehot) * inv_t);
        }
    }
    return out;
}

/// Accumulates dLoss/d(parameters) into `g`. It does not zero `g` first: the
/// caller owns that, and one buffer then serves an accumulation schedule.
///
/// The walk is the forward pass in reverse:
///     tied head, final norm, then each block from the last layer to the first,
///     and finally the scatter into the embedding table.
///
/// This runs the forward pass itself and throws the logits away. A caller that
/// already has them, which is every training step, wants `backwardFrom` and a
/// `Cache` instead: the two then share the whole of `backwardFrom`, and the
/// only thing this adds is the one forward a standalone gradient check has to
/// run anyway.
pub fn backward(
    allocator: std.mem.Allocator,
    p: model.Params,
    g: *Grads,
    cfg: model.Config,
    tokens: []const u32,
    dlogits: Tensor,
) !void {
    try check(p, g, cfg, tokens, dlogits);
    var cache = try Cache.init(allocator, p, cfg, tokens);
    defer cache.deinit();
    var logits = try model.forwardWith(allocator, p, cfg, tokens, &cache.sink);
    defer logits.deinit();
    try backwardFrom(allocator, p, g, cfg, tokens, dlogits, &cache);
}

/// `backward`, over a forward pass that has already run and been collected into
/// `c` by its caller.
///
/// The signature is the honest cost of not running the forward twice: a
/// training step computes the logits for the loss and the intermediates for the
/// gradients in one pass, and there is no way to hand the second set to
/// `backward` without either recomputing them or passing them in. Everything
/// after the validation is the same code the six-argument form runs.
pub fn backwardFrom(
    allocator: std.mem.Allocator,
    p: model.Params,
    g: *Grads,
    cfg: model.Config,
    tokens: []const u32,
    dlogits: Tensor,
    c: *const Cache,
) !void {
    try check(p, g, cfg, tokens, dlogits);
    const t_count = tokens.len;
    const d = model.dModel(cfg);
    if (c.blocks.len != p.layers.len) return error.DimensionMismatch;
    const stream = c.blocks[c.blocks.len - 1].out;

    var d_final_h = try Tensor.init(allocator, t_count, d);
    defer d_final_h.deinit();

    // The tied head, and the one place both of tok_embed's paths meet:
    //
    //     logits[t][v] = dot(tok_embed[v], final_h[t])
    //
    //     d final_h[t][i] += dlogits[t][v] * tok_embed[v][i]
    //     d tok_embed[v][i] += dlogits[t][v] * final_h[t][i]
    //
    // The second line is the output projection, and the matching input scatter
    // is at the very bottom of this function. Dropping either one leaves
    // tok_embed half trained, which is invisible in the loss curve and is the
    // easiest mistake in this file.
    var final_h = try norm.forward(allocator, stream, p.final_norm);
    defer final_h.deinit();
    for (0..t_count) |t| {
        const dl = dlogits.rowConst(t);
        const fh = final_h.rowConst(t);
        const dfh = d_final_h.row(t);
        for (0..cfg.vocab_size) |v| {
            const gv = dl[v];
            const e = p.tok_embed.rowConst(v);
            const de = g.tok_embed.row(v);
            for (0..d) |i| {
                dfh[i] += gv * e[i];
                de[i] += gv * fh[i];
            }
        }
    }

    // Through the final norm, onto the residual stream the last block wrote.
    var d_x = try Tensor.init(allocator, t_count, d);
    defer d_x.deinit();
    normBackward(stream, p.final_norm, d_final_h, &g.final_norm, &d_x);

    // Each block, last to first, on the intermediates the forward already built.
    var layer = p.layers.len;
    while (layer > 0) {
        layer -= 1;
        // Two buffers, swapped. One would mean a memcpy of the whole stream per
        // layer, and one allocation per layer, for a tensor whose only reader is
        // the scatter at the bottom of this function.
        var d_next = try Tensor.init(allocator, t_count, d);
        defer d_next.deinit();
        try blockBackward(&d_next, p.layers[layer], &g.layers[layer], c.blocks[layer], c.xIn(layer), d_x, cfg);
        const spare = d_x;
        d_x = d_next;
        d_next = spare;
    }

    // The input path of the tied embedding: x[t] is tok_embed[tokens[t]], so
    // the gradient lands on exactly the row that row was read from.
    for (tokens, 0..) |tok, t| {
        const dx = d_x.rowConst(t);
        const de = g.tok_embed.row(@as(usize, tok));
        for (0..d) |i| de[i] += dx[i];
    }
}

/// Everything `backwardFrom` needs that only the forward pass knows.
///
/// The eleven tensors one block's backward pass reads, plus the block's output.
/// A tape would keep each op's output separately, which is a wider set than
/// these twelve, so this holds one `Block` per layer instead. They are the
/// values the loss was built from, which is the point: the backward
/// differentiates the forward that actually ran rather than a second one built
/// to look like it.
const Block = struct {
    attn_in: Tensor, // norm(x_in, attn_norm)          [T, d]
    q_pos: Tensor, // rope(x_in @ wq)                   [T, d]
    k_pos: Tensor, // rope(x_in @ wk)                   [T, kv]
    v: Tensor, // x_in @ wv, unrotated                 [T, kv]
    ctx: Tensor, // attention(q_pos, k_pos, v)         [T, d]
    x_mid: Tensor, // x_in + ctx @ wo                   [T, d]
    mlp_in: Tensor, // norm(x_mid, mlp_norm)            [T, d]
    gate: Tensor, // mlp_in @ w_gate                    [T, h]
    up: Tensor, // mlp_in @ w_up                        [T, h]
    a: Tensor, // silu(gate) * up                       [T, h]
    out: Tensor, // x_mid + a @ w_down                   [T, d]

    pub fn deinit(self: *Block) void {
        self.attn_in.deinit();
        self.q_pos.deinit();
        self.k_pos.deinit();
        self.v.deinit();
        self.ctx.deinit();
        self.x_mid.deinit();
        self.mlp_in.deinit();
        self.gate.deinit();
        self.up.deinit();
        self.a.deinit();
        self.out.deinit();
    }
};

/// The twelve tensors one block's forward produced, copied out as it produces
/// them, and the first block's input.
///
/// Every buffer is allocated up front, in `init`, and `put` only copies into
/// them. That is what lets `model.Sink.put` return `void`: an allocation inside
/// the callback would have to be parked in an error field and checked by the
/// caller, which is the shape `src/removed.zig` needs because it is writing to
/// an append-only list and this is not because the shapes are all known before
/// the pass starts.
///
/// The pass frees its own buffers as it goes, so the copy is not optional:
/// `residual1` of one layer and the next layer's input are the same two
/// ping-ponged buffers, and `gate` dies before `mlp_out` is even written.
pub const Cache = struct {
    /// x[t] = tok_embed[tokens[t]], the first block's input. Copied rather than
    /// emitted, because the embedding is one row memcpy per token and the sink
    /// names are there for tensors only a block can produce.
    embed: Tensor,
    blocks: []Block,
    sink: model.Sink = .{ .put = put },

    pub fn init(
        allocator: std.mem.Allocator,
        p: model.Params,
        cfg: model.Config,
        tokens: []const u32,
    ) !Cache {
        if (tokens.len == 0) return error.EmptyBatch;
        // `embed` below reads tok_embed by token id with no bounds check of its
        // own, and a training step builds the cache before anything has looked
        // at the ids. The check has to live here, at the boundary the untrusted
        // slice first reaches, rather than in the backward pass that used to
        // make it before doing anything else.
        for (tokens) |tok| {
            if (@as(usize, tok) >= cfg.vocab_size) return error.TokenOutOfRange;
        }
        const blocks = try allocator.alloc(Block, p.layers.len);
        var built: usize = 0;
        errdefer {
            for (blocks[0..built]) |*one| one.deinit();
            allocator.free(blocks);
        }
        while (built < blocks.len) : (built += 1) {
            blocks[built] = try initBlock(allocator, cfg, tokens.len);
        }
        return .{
            .embed = try embed(allocator, p, tokens, model.dModel(cfg)),
            .blocks = blocks,
        };
    }

    pub fn deinit(self: *Cache) void {
        // Reading the allocator back off a tensor is how the slice is released
        // without Cache carrying a second source of truth, the trade
        // `model.Params` and `Grads` already make.
        const allocator = self.embed.allocator;
        for (self.blocks) |*one| one.deinit();
        allocator.free(self.blocks);
        self.blocks = &.{};
        self.embed.deinit();
    }

    /// The stream a block starts from, which is the block before it's output.
    /// Carried as a field of the previous block rather than in an array of its
    /// own, so the same tensor is not held twice.
    fn xIn(self: *const Cache, layer: usize) Tensor {
        return if (layer == 0) self.embed else self.blocks[layer - 1].out;
    }

    /// `model.Sink.put` takes the sink as its own first argument so a caller
    /// does not have to carry a context pointer beside it, which is what makes
    /// `@fieldParentPtr` how the cache gets back to its own fields.
    fn put(sink: *model.Sink, name: model.Name, layer: usize, t: Tensor) void {
        const self: *Cache = @fieldParentPtr("sink", sink);
        // `model.forwardWith` has already refused a parameter set whose depth
        // disagrees with the config by the time anything reaches here, so the
        // index cannot be out of range on any path that runs.
        const b = &self.blocks[layer];
        switch (name) {
            .attn_norm_out => copyInto(b.attn_in, t),
            .q_rope => copyInto(b.q_pos, t),
            .k_rope => copyInto(b.k_pos, t),
            .v => copyInto(b.v, t),
            .attn_ctx => copyInto(b.ctx, t),
            .residual1 => copyInto(b.x_mid, t),
            .mlp_norm_out => copyInto(b.mlp_in, t),
            .mlp_gate => copyInto(b.gate, t),
            .mlp_up => copyInto(b.up, t),
            .mlp_hidden => copyInto(b.a, t),
            .residual2 => copyInto(b.out, t),
            // Exhaustively listed rather than collapsed into `else => {}`. The
            // catch-all was a silent data-loss path across a four-file seam: add
            // a `model.Name` and this arm swallows the new tensor without
            // complaint, the gradient comes out wrong, and nothing fails.
            // Worse, the test that should have caught it derived its expected
            // count from the same enum the omission lives behind, so it passes
            // too. A name with no case here is now a compile error, which is
            // the only version of this failure that cannot be shipped.
            //
            // These six are emitted but deliberately not stored:
            //   `q`, `k`       the projections RoPE consumes; the backward reads
            //                  the rotated pair, which is `q_rope`/`k_rope`.
            //   `attn_proj`    the attention output after `wo`, which is `x_mid`
            //                  minus the block input. The backward gets d_wo from
            //                  `ctx` and rebuilds the sum, so storing the
            //                  projection would be 512 KB per layer of nothing.
            //   `mlp_out`      the feed-forward before the second residual add;
            //                  the backward reads the sum, which is `residual2`.
            //   `final_norm`   the model's, not any layer's.
            //   `logits`       likewise; `dLossDLogits` is the backward's entry
            //                  point rather than a stored intermediate.
            .q,
            .k,
            .attn_proj,
            .mlp_out,
            .final_norm,
            .logits,
            => {},
        }
    }
};

/// `dst` is the same shape as `src` because `initBlock` sized it from the same
/// config the pass that fills it was built from, so a difference here is an
/// internal invariant that broke rather than a caller that lied.
fn copyInto(dst: Tensor, src: Tensor) void {
    std.debug.assert(dst.data.len == src.data.len);
    @memcpy(dst.data, src.data);
}

fn initBlock(allocator: std.mem.Allocator, cfg: model.Config, t_count: usize) !Block {
    const d = model.dModel(cfg);
    const kv = cfg.n_kv_heads * cfg.head_dim;
    const h = model.ffnDim(cfg);
    // Field order, the same eleven the struct declares, so a shape and the
    // field it belongs to sit side by side.
    const shapes = [11][2]usize{
        .{ t_count, d }, // attn_in
        .{ t_count, d }, // q_pos
        .{ t_count, kv }, // k_pos
        .{ t_count, kv }, // v
        .{ t_count, d }, // ctx
        .{ t_count, d }, // x_mid
        .{ t_count, d }, // mlp_in
        .{ t_count, h }, // gate
        .{ t_count, h }, // up
        .{ t_count, h }, // a
        .{ t_count, d }, // out
    };
    // The tensors land in an array first, not straight into a `Block` literal.
    // A literal that fails part way through is never assigned at all, so an
    // errdefer on it would never be registered and every tensor drawn before
    // the failure would leak. Here the unwind frees the prefix that exists.
    var t: [11]Tensor = undefined;
    var n: usize = 0;
    errdefer for (t[0..n]) |*one| one.deinit();
    while (n < t.len) : (n += 1) {
        t[n] = try Tensor.init(allocator, shapes[n][0], shapes[n][1]);
    }
    return .{
        .attn_in = t[0],
        .q_pos = t[1],
        .k_pos = t[2],
        .v = t[3],
        .ctx = t[4],
        .x_mid = t[5],
        .mlp_in = t[6],
        .gate = t[7],
        .up = t[8],
        .a = t[9],
        .out = t[10],
    };
}

/// The shape and token checks both entry points make, in the order that decides
/// which error a caller sees. Shared so the six-argument form and the seven-
/// argument one cannot disagree about what is a valid request.
fn check(
    p: model.Params,
    g: *const Grads,
    cfg: model.Config,
    tokens: []const u32,
    dlogits: Tensor,
) !void {
    const t_count = tokens.len;
    if (t_count == 0) return error.EmptyBatch;
    if (t_count > cfg.n_ctx) return error.SequenceTooLong;
    if (dlogits.rows != t_count or dlogits.cols != cfg.vocab_size) return error.DimensionMismatch;
    if (g.layers.len != p.layers.len) return error.DimensionMismatch;
    // The token index is a trust boundary: the embedding scatter at the bottom
    // of `backwardFrom` reads it with no bounds check of its own.
    for (tokens) |tok| {
        if (@as(usize, tok) >= cfg.vocab_size) return error.TokenOutOfRange;
    }
    if (p.tok_embed.rows != cfg.vocab_size or p.tok_embed.cols != model.dModel(cfg)) return error.DimensionMismatch;
}

/// Accumulates dLoss/d(block input) into `d_x_in`, and the block's nine weight
/// gradients into `lg`.
///
/// Reads the forward as
///     x_mid = x_in + (attn(norm1(x_in) @ wo))
///     x_out = x_mid + swiglu(norm2(x_mid)) @ w_down
/// and walks it back out. A residual add contributes no derivative of its own,
/// so the stream gradient passes through both adds unchanged.
fn blockBackward(
    d_x_in: *Tensor,
    l: model.Layer,
    lg: *LayerGrads,
    b: Block,
    x_in: Tensor,
    d_x_out: Tensor,
    cfg: model.Config,
) !void {
    const t_count = d_x_out.rows;
    const d = model.dModel(cfg);
    const ffn = model.ffnDim(cfg);

    // ff = a @ w_down, with d_ff = d_x_out from the residual add.
    //     d a = d_ff @ w_down^T,  d w_down = a^T @ d_ff
    var d_a = try Tensor.init(d_x_in.allocator, t_count, ffn);
    defer d_a.deinit();
    inputGrad(&d_a, d_x_out, l.w_down);
    weightGrad(&lg.w_down, b.a, d_x_out);

    // a = silu(gate) * up, so
    //     d up   = d a * silu(gate)
    //     d gate = d a * up * silu'(gate)
    // with silu'(z) = s * (1 + z * (1 - s)), s = sigmoid(z). The derivative is
    // the same on both sides of mlp.silu's branch, because silu is z*sigmoid(z)
    // on both of them.
    var d_gate = try Tensor.init(d_x_in.allocator, t_count, ffn);
    defer d_gate.deinit();
    var d_up = try Tensor.init(d_x_in.allocator, t_count, ffn);
    defer d_up.deinit();
    for (0..t_count) |t| {
        const gate = b.gate.rowConst(t);
        const up = b.up.rowConst(t);
        const da = d_a.rowConst(t);
        const dg = d_gate.row(t);
        const du = d_up.row(t);
        for (0..ffn) |i| {
            const z = gate[i];
            const s = sigmoid(z);
            du[i] = da[i] * mlp.silu(z);
            dg[i] = da[i] * up[i] * (s * (1 + z * (1 - s)));
        }
    }
    weightGrad(&lg.w_gate, b.mlp_in, d_gate);
    weightGrad(&lg.w_up, b.mlp_in, d_up);

    // d mlp_in = d gate @ w_gate^T + d up @ w_up^T, then through the norm.
    var d_mlp_in = try Tensor.init(d_x_in.allocator, t_count, d);
    defer d_mlp_in.deinit();
    inputGrad(&d_mlp_in, d_gate, l.w_gate);
    inputGrad(&d_mlp_in, d_up, l.w_up);

    var d_of_norm = try Tensor.init(d_x_in.allocator, t_count, d);
    defer d_of_norm.deinit();
    normBackward(b.x_mid, l.mlp_norm, d_mlp_in, &lg.mlp_norm, &d_of_norm);

    // The residual add carries d_x_out straight onto x_mid, and d_mlp_in reaches
    // x_mid only through the norm above, which has already folded it in. Adding
    // d_mlp_in again here would count the whole branch twice.
    var d_x_mid = try Tensor.init(d_x_in.allocator, t_count, d);
    defer d_x_mid.deinit();
    for (0..t_count) |t| {
        const out = d_x_out.rowConst(t);
        const of_norm = d_of_norm.rowConst(t);
        const dst = d_x_mid.row(t);
        for (0..d) |i| dst[i] = out[i] + of_norm[i];
    }

    // proj = ctx @ wo, so d ctx = d_x_mid @ wo^T and d wo = ctx^T @ d_x_mid.
    var d_ctx = try Tensor.init(d_x_in.allocator, t_count, d);
    defer d_ctx.deinit();
    inputGrad(&d_ctx, d_x_mid, l.wo);
    weightGrad(&lg.wo, b.ctx, d_x_mid);

    var ag = try attentionBackward(d_x_in.allocator, b.q_pos, b.k_pos, b.v, d_ctx, cfg);
    defer ag.deinit();

    // RoPE is a rotation, so its inverse is the transpose: the same pair of
    // multiply-adds with the sign of the sine flipped.
    var d_q = try Tensor.init(d_x_in.allocator, t_count, ag.dq.cols);
    defer d_q.deinit();
    ropeBackward(ag.dq, &d_q, 0, rope_theta, cfg.head_dim);
    var d_k = try Tensor.init(d_x_in.allocator, t_count, ag.dk.cols);
    defer d_k.deinit();
    ropeBackward(ag.dk, &d_k, 0, rope_theta, cfg.head_dim);

    weightGrad(&lg.wq, b.attn_in, d_q);
    weightGrad(&lg.wk, b.attn_in, d_k);
    weightGrad(&lg.wv, b.attn_in, ag.dv);

    // d attn_in = d q @ wq^T + d k @ wk^T + d v @ wv^T, then through the norm.
    var d_attn_in = try Tensor.init(d_x_in.allocator, t_count, d);
    defer d_attn_in.deinit();
    inputGrad(&d_attn_in, d_q, l.wq);
    inputGrad(&d_attn_in, d_k, l.wk);
    inputGrad(&d_attn_in, ag.dv, l.wv);

    var d_of_norm2 = try Tensor.init(d_x_in.allocator, t_count, d);
    defer d_of_norm2.deinit();
    normBackward(x_in, l.attn_norm, d_attn_in, &lg.attn_norm, &d_of_norm2);

    // x_mid = x_in + proj, so the stream gradient reaching x_in is the one
    // that came out of the attention branch plus the one that skipped it.
    for (0..t_count) |t| {
        const mid = d_x_mid.rowConst(t);
        const of_norm = d_of_norm2.rowConst(t);
        const dst = d_x_in.row(t);
        for (0..d) |i| dst[i] += mid[i] + of_norm[i];
    }
}

const AttnGrads = struct {
    dq: Tensor, // [T, n_heads * head_dim]
    dk: Tensor, // [T, n_kv_heads * head_dim]
    dv: Tensor, // [T, n_kv_heads * head_dim]

    pub fn deinit(self: *AttnGrads) void {
        self.dq.deinit();
        self.dk.deinit();
        self.dv.deinit();
    }
};

/// Backward of `attention.forward`.
///
/// With p the softmax over the causal prefix 0..t,
///     d v[s] += p[t][s] * d h[t]
///     d p[t][s] = dot(d h[t], v[s])
///     d score[t][s] = p[t][s] * (d p[t][s] - sum_j p[t][j] d p[t][j])
/// and the scores are the rotated dot product times the head scale, so
///     d q[t] = sum_{s <= t} d score[t][s] * k[s] * scale
///     d k[s] += sum_{t >= s} d score[t][s] * q[t] * scale
///
/// The prefix is what makes d k and d v accumulate over t >= s rather than over
/// every row: a key or value at position s is read only by rows t >= s, and
/// dropping the restriction lets a later row's gradient leak backwards.
///
/// Four loops are unrolled eight ways and the softmax, exp and p_ds loops are
/// not, because those three are already flat in `s` or a single chain over a
/// handful of terms. The four that are:
///     score    dot(q[t], k[s])      one chain of `dim` adds per position
///     d p      dot(d h[t], v[s])    the same, against the same gathered rows
///     d q      sum_s d score k[s]   `dim` independent chains over the prefix
///     d k d v  d score q[t], p d h  `dim` independent read-modify-writes
/// Each split runs the same terms into the same per-accumulator chain in the
/// same ascending order as the scalar loop did, and no lane is ever added to
/// another, so every value is bit for bit what it was. The split is on an axis
/// that is independent per accumulator for that reason; a `@Vector` reduction
/// over `j` or over `s` would reassociate and move the bytes, and none of these
/// do. A count that is not a multiple of eight finishes on the scalar loop, so
/// an odd `head_dim` or an odd prefix length costs nothing and changes nothing.
fn attentionBackward(
    allocator: std.mem.Allocator,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    dout: Tensor,
    cfg: model.Config,
) !AttnGrads {
    const dim = cfg.head_dim;
    const group = cfg.n_heads / cfg.n_kv_heads;
    // Four of the six loops below are unrolled this many ways, and it is the
    // same latency fix `weightGrad` and `inputGrad` made, on the same two
    // shapes: a dot product per position and a column per head, both a serial
    // chain of f64 adds over a loop LLVM will not split because the trip count
    // is `t + 1` and the rows are gathered. The lanes are never summed
    // together, in any of the four, which is the property that keeps the bytes.
    const lanes = 8;
    // The scale is the same one attention.forward divides by, kept in f64 so
    // the softmax this rebuilds matches the one the forward built.
    const scale = 1.0 / @sqrt(@as(f64, @floatFromInt(dim)));

    var g: AttnGrads = undefined;
    g.dq = try Tensor.init(allocator, q.rows, q.cols);
    errdefer g.dq.deinit();
    g.dk = try Tensor.init(allocator, k.rows, k.cols);
    errdefer g.dk.deinit();
    g.dv = try Tensor.init(allocator, v.rows, v.cols);
    errdefer g.dv.deinit();

    // One row of probabilities and one row of d p at a time, both f64 to match
    // the forward's accumulation, and both reused across rows the way
    // attention.forward reuses its score buffer.
    var probs = try allocator.alloc(f64, q.rows);
    defer allocator.free(probs);
    var d_probs = try allocator.alloc(f64, q.rows);
    defer allocator.free(d_probs);
    // probs * (d p - d p . p), which is the same number for every j and every
    // accumulator below. It was computed inside those loops, `dim` times per
    // row over a term that does not mention j, and hoisting it is the same
    // operations in the same order: `p_ds[s]` is the f64 product of the same
    // two f64 values the inner loop multiplied, and the multiply by the operand
    // and by `scale` still happens left to right behind it.
    var p_ds = try allocator.alloc(f64, q.rows);
    defer allocator.free(p_ds);

    for (0..cfg.n_heads) |h| {
        const kv = h / group;
        for (0..q.rows) |t| {
            const q_head = q.rowConst(t)[h * dim ..][0..dim];
            const dh = dout.rowConst(t)[h * dim ..][0..dim];

            // Rebuild the forward's softmax over the causal prefix. The `s`
            // loop is unrolled eight ways: eight prefixes are independent dot
            // products, and one of them is a serial chain of `dim` f64 adds that
            // cannot retire faster than the adder's latency. The lanes are
            // never summed together, so each `probs[s]` is bit for bit what
            // the scalar loop wrote.
            var row_max: f64 = -std.math.inf(f64);
            {
                var s: usize = 0;
                while (s + lanes <= t + 1) : (s += lanes) {
                    var rows: [lanes][]const f32 = undefined;
                    for (0..lanes) |u| rows[u] = k.rowConst(s + u)[kv * dim ..][0..dim];
                    var dot: [lanes]f64 = @splat(0);
                    for (q_head, 0..) |a, jj| {
                        const af: f64 = @floatCast(a);
                        for (0..lanes) |u| dot[u] += af * @as(f64, rows[u][jj]);
                    }
                    for (0..lanes) |u| {
                        probs[s + u] = dot[u] * scale;
                        row_max = @max(row_max, probs[s + u]);
                    }
                }
                while (s < t + 1) : (s += 1) {
                    const k_head = k.rowConst(s)[kv * dim ..][0..dim];
                    var dot: f64 = 0;
                    for (q_head, k_head) |a, b| dot += @as(f64, a) * @as(f64, b);
                    probs[s] = dot * scale;
                    row_max = @max(row_max, probs[s]);
                }
            }
            var denom: f64 = 0;
            for (0..t + 1) |s| {
                probs[s] = @exp(probs[s] - row_max);
                denom += probs[s];
            }
            for (0..t + 1) |s| probs[s] /= denom;

            // d p[s] = dot(d h, v[s]), then d score[s] = p[s] (d p[s] - sum p d p)
            // with the sum over the prefix, which is the softmax Jacobian.
            var dot_pp: f64 = 0;
            {
                var s: usize = 0;
                while (s + lanes <= t + 1) : (s += lanes) {
                    var rows: [lanes][]const f32 = undefined;
                    for (0..lanes) |u| rows[u] = v.rowConst(s + u)[kv * dim ..][0..dim];
                    var acc: [lanes]f64 = @splat(0);
                    for (dh, 0..) |d, jj| {
                        const df: f64 = @floatCast(d);
                        for (0..lanes) |u| acc[u] += df * @as(f64, rows[u][jj]);
                    }
                    for (0..lanes) |u| {
                        d_probs[s + u] = acc[u];
                        dot_pp += probs[s + u] * acc[u];
                    }
                }
                while (s < t + 1) : (s += 1) {
                    const v_head = v.rowConst(s)[kv * dim ..][0..dim];
                    var acc: f64 = 0;
                    for (0..dim) |j| acc += @as(f64, dh[j]) * @as(f64, v_head[j]);
                    d_probs[s] = acc;
                    dot_pp += probs[s] * acc;
                }
            }

            for (0..t + 1) |s| p_ds[s] = probs[s] * (d_probs[s] - dot_pp);

            // d q[t] is written by this row alone, so it is assigned. d k and
            // d v accumulate over every row that reads the position.
            for (0..dim / lanes) |g_lane| {
                const j = g_lane * lanes;
                var acc: [lanes]f64 = @splat(0);
                for (0..t + 1) |s| {
                    const k_head = k.rowConst(s)[kv * dim + j ..][0..lanes];
                    for (0..lanes) |u| acc[u] += p_ds[s] * @as(f64, k_head[u]) * scale;
                }
                for (0..lanes) |u| g.dq.set(t, h * dim + j + u, @floatCast(acc[u]));
            }
            for (0..dim % lanes) |u| {
                const j = (dim / lanes) * lanes + u;
                var acc: f64 = 0;
                for (0..t + 1) |s| {
                    const k_head = k.rowConst(s)[kv * dim ..][0..dim];
                    acc += p_ds[s] * @as(f64, k_head[j]) * scale;
                }
                g.dq.set(t, h * dim + j, @floatCast(acc));
            }
            for (0..t + 1) |s| {
                const dk_row = g.dk.row(s);
                const dv_row = g.dv.row(s);
                const p_ds_s = p_ds[s];
                const probs_s = probs[s];
                for (0..dim / lanes) |g_lane| {
                    const j = g_lane * lanes;
                    for (0..lanes) |u| {
                        dk_row[kv * dim + j + u] += @floatCast(p_ds_s * @as(f64, q_head[j + u]) * scale);
                        dv_row[kv * dim + j + u] += @floatCast(probs_s * @as(f64, dh[j + u]));
                    }
                }
                for (0..dim % lanes) |u| {
                    const j = (dim / lanes) * lanes + u;
                    dk_row[kv * dim + j] += @floatCast(p_ds_s * @as(f64, q_head[j]) * scale);
                    dv_row[kv * dim + j] += @floatCast(probs_s * @as(f64, dh[j]));
                }
            }
        }
    }
    return g;
}

/// d q = R^T d q_pos. The rotation
///     [out_lo, out_hi] = [[c, -s], [s, c]] [lo, hi]
/// is orthogonal, so the transpose undoes it: the same two multiply-adds with
/// the sine's sign flipped. The pairing has to be the half-split one
/// rope.forward uses, element `i` with element `i + head_dim/2` of the same
/// head block, and the angle has to be the same f64 angle, or a query silently
/// picks up a neighbour's phase.
fn ropeBackward(dout: Tensor, din: *Tensor, pos: usize, theta: f64, head_dim: usize) void {
    std.debug.assert(head_dim != 0 and dout.cols % head_dim == 0);
    const half = head_dim / 2;
    for (0..dout.rows) |r| {
        const src = dout.rowConst(r);
        const dst = din.row(r);
        const t: f64 = @floatFromInt(pos + r);
        for (0..dout.cols / head_dim) |h| {
            const base = h * head_dim;
            for (0..half) |i| {
                const exponent = @as(f64, @floatFromInt(2 * i)) / @as(f64, @floatFromInt(head_dim));
                const angle = t / std.math.pow(f64, theta, exponent);
                const c = @cos(angle);
                const s = @sin(angle);
                const lo = @as(f64, src[base + i]);
                const hi = @as(f64, src[base + i + half]);
                dst[base + i] = @floatCast(lo * c + hi * s);
                dst[base + i + half] = @floatCast(hi * c - lo * s);
            }
        }
    }
}

/// RMSNorm, backward. With rms = sqrt(mean(x^2) + eps), n = x / rms and
/// y = n * w, for one row of the batch:
///
///     d w[i] += g[i] * n[i]
///     d x[i]  = (d n[i] - n[i] * (d n . n) / d) / rms,  d n[i] = g[i] * w[i]
///
/// The mean is over d, not over the batch, and eps keeps the division finite at
/// a zero row, which is the shape a zero norm weight leaves behind.
///
/// `x` is the norm's input, `g` is the gradient of the norm's output, `dw`
/// accumulates and `dx` is written whole.
fn normBackward(x: Tensor, w: Tensor, g: Tensor, dw: *Tensor, dx: *Tensor) void {
    const d = x.cols;
    // The forward pass's own constant, not a second copy of it. A gradient that
    // differentiates a slightly different function than the one evaluated is
    // still a plausible-looking gradient, and a finite difference of the forward
    // pass agrees with it closely enough to pass.
    const eps = norm.eps;
    const w_row = w.data[0..d];

    // dw is the norm's own [1, d] weight gradient, so every row of the batch
    // folds into the same d elements rather than into a row of its own.
    const dw_row = dw.data[0..d];
    for (0..x.rows) |r| {
        const x_row = x.rowConst(r);
        const g_row = g.rowConst(r);
        const dx_row = dx.row(r);

        // f64 accumulator, narrowed once per row to f32, which is what
        // norm.forward stores. The narrowing is the point, not a rounding
        // detail: the forward divides by an f32 rms, so keeping the f64 value
        // here would differentiate a marginally different function than the one
        // loss.forward measured.
        var sum_sq: f64 = 0;
        for (x_row) |val| sum_sq += @as(f64, val) * @as(f64, val);
        const rms: f32 = @floatCast(@sqrt(sum_sq / @as(f64, @floatFromInt(d)) + eps));

        // d n . n, with d n[i] = g[i] * w[i] and n[i] = x[i] / rms.
        var dot: f64 = 0;
        for (0..d) |i| {
            const n = @as(f64, @as(f32, @floatCast(x_row[i])) / rms);
            dot += @as(f64, g_row[i]) * @as(f64, w_row[i]) * n;
        }
        const inv_d = 1.0 / @as(f64, @floatFromInt(d));
        for (0..d) |i| {
            const n = @as(f64, @as(f32, @floatCast(x_row[i])) / rms);
            const dn = @as(f64, g_row[i]) * @as(f64, w_row[i]);
            dw_row[i] += @floatCast(@as(f64, g_row[i]) * n);
            dx_row[i] = @floatCast((dn - n * dot * inv_d) / rms);
        }
    }
}

/// d W += input^T @ dout, the weight gradient of `C = input @ W`.
///
/// f32, and the same i-k-j order `tensor.matmul` uses, so a step that sums a
/// gradient over two batches reduces the terms in the order the forward sums
/// the same terms. f64 here would buy accuracy the f32 forward cannot
/// reproduce anyway.
///
/// The column loop is unrolled eight ways, and like the tied head's this is a
/// latency fix rather than a volume one. The op moves 12 bytes per element for
/// one FMA and every one of those bytes is an L1 hit, so there is no volume to
/// remove; what there is instead is a loop LLVM will not touch. `w_row[j] +=`
/// is a read-modify-write through two pointers that may alias, and the
/// vectoriser declines a store it cannot prove is safe, so it emitted
/// `ldr/fmul/fadd/str` plus three instructions of loop overhead per element and
/// ran the whole backward pass at 0.57 GFLOP/s. Eight independent columns give
/// the scheduler eight chains to overlap and cut the overhead to under one
/// instruction per element.
///
/// Byte-identity is a property of the split. Accumulator `j` accumulates
/// exactly the terms `j` accumulated before, still in ascending `t`, so
/// `w_row[j]` is bit for bit what the scalar loop left; nothing is combined
/// across lanes and there is no second value to narrow. A width that is not a
/// multiple of eight finishes on the scalar loop, so an odd shape costs
/// nothing and changes nothing.
fn weightGrad(w: *Tensor, input: Tensor, dout: Tensor) void {
    const lanes = 8;
    for (0..input.cols) |i| {
        const w_row = w.row(i);
        for (0..input.rows) |t| {
            const scale = input.rowConst(t)[i];
            const dout_row = dout.rowConst(t);
            var j: usize = 0;
            while (j + lanes <= dout.cols) : (j += lanes) {
                for (0..lanes) |k| w_row[j + k] += scale * dout_row[j + k];
            }
            while (j < dout.cols) : (j += 1) w_row[j] += scale * dout_row[j];
        }
    }
}

/// d input += dout @ W^T, the input gradient of `C = input @ W`.
///
///     d input[t][i] += sum_j dout[t][j] * W[i][j]
///
/// The inner sum runs over W's columns at a fixed row, so both operands are
/// read as contiguous rows. `out` arrives zeroed from Tensor.init, so
/// accumulating in place is the whole assignment.
///
/// The row loop is unrolled eight ways, and it is the same latency fix as
/// `weightGrad`'s column loop, on the other side of the pair. This is a dot
/// product per output element, so a scalar accumulator is a serial chain of
/// `W.cols` f32 adds that cannot retire faster than the adder's latency: at
/// [T=256, d=128, ffn=512] the MLP branch below is two of these over 33.6M
/// multiply-adds and it ran 12x under the rate the same machine retires a
/// plain vector FMA at. Eight rows are independent, so eight chains fill the
/// latency, and the operand rows are fetched once for the group instead of
/// once per element.
///
/// Byte-identity is a property of the split. Accumulator `k` sums the same
/// `j` in the same ascending order over the same terms as the scalar loop's
/// accumulator did for row `k`, so `out_row[i + k]` receives bit for bit the
/// value the scalar loop stored there. The lanes are never summed together,
/// which is the one change that would have moved the bytes: a `@Vector`
/// accumulator over `j` would reassociate the reduction, and this does not.
/// A row count that is not a multiple of eight finishes on the scalar loop.
fn inputGrad(out: *Tensor, dout: Tensor, w: Tensor) void {
    const lanes = 8;
    for (0..dout.rows) |t| {
        const dout_row = dout.rowConst(t);
        const out_row = out.row(t);
        var i: usize = 0;
        while (i + lanes <= w.rows) : (i += lanes) {
            var rows: [lanes][]const f32 = undefined;
            for (0..lanes) |k| rows[k] = w.rowConst(i + k);
            var acc: [lanes]f32 = @splat(0);
            for (0..w.cols) |j| {
                const dj = dout_row[j];
                for (0..lanes) |k| acc[k] += dj * rows[k][j];
            }
            for (0..lanes) |k| out_row[i + k] += acc[k];
        }
        while (i < w.rows) : (i += 1) {
            const w_row = w.rowConst(i);
            var acc: f32 = 0;
            for (0..w.cols) |j| acc += dout_row[j] * w_row[j];
            out_row[i] += acc;
        }
    }
}

/// sigmoid(z) = 1 / (1 + exp(-z)) on the same two branches `mlp.silu` uses, so
/// the derivative reported here is the derivative of the function that ran.
fn sigmoid(z: f32) f32 {
    if (z >= 0) return 1.0 / (1.0 + @exp(-z));
    const e = @exp(z);
    return e / (1 + e);
}

/// x[t] = tok_embed[tokens[t]]. The gradient of this scatter is at the bottom
/// of `backwardFrom`; the forward half is here so the cache and the real
/// forward read the same rows.
fn embed(allocator: std.mem.Allocator, p: model.Params, tokens: []const u32, d: usize) !Tensor {
    var x = try Tensor.init(allocator, tokens.len, d);
    errdefer x.deinit();
    for (tokens, 0..) |tok, t| {
        @memcpy(x.row(t), p.tok_embed.rowConst(@as(usize, tok)));
    }
    return x;
}

fn zeroLayerGrads(allocator: std.mem.Allocator, l: *const model.Layer) !LayerGrads {
    var g: LayerGrads = undefined;
    g.attn_norm = try Tensor.init(allocator, l.attn_norm.rows, l.attn_norm.cols);
    errdefer g.attn_norm.deinit();
    g.wq = try Tensor.init(allocator, l.wq.rows, l.wq.cols);
    errdefer g.wq.deinit();
    g.wk = try Tensor.init(allocator, l.wk.rows, l.wk.cols);
    errdefer g.wk.deinit();
    g.wv = try Tensor.init(allocator, l.wv.rows, l.wv.cols);
    errdefer g.wv.deinit();
    g.wo = try Tensor.init(allocator, l.wo.rows, l.wo.cols);
    errdefer g.wo.deinit();
    g.mlp_norm = try Tensor.init(allocator, l.mlp_norm.rows, l.mlp_norm.cols);
    errdefer g.mlp_norm.deinit();
    g.w_gate = try Tensor.init(allocator, l.w_gate.rows, l.w_gate.cols);
    errdefer g.w_gate.deinit();
    g.w_up = try Tensor.init(allocator, l.w_up.rows, l.w_up.cols);
    errdefer g.w_up.deinit();
    g.w_down = try Tensor.init(allocator, l.w_down.rows, l.w_down.cols);
    return g;
}

fn freeLayerGrads(l: *LayerGrads) void {
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
