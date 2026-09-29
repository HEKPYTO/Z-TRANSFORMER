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
pub fn backward(
    allocator: std.mem.Allocator,
    p: model.Params,
    g: *Grads,
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
    // of this function reads it with no bounds check of its own.
    for (tokens) |tok| {
        if (@as(usize, tok) >= cfg.vocab_size) return error.TokenOutOfRange;
    }

    const d = model.dModel(cfg);
    if (p.tok_embed.rows != cfg.vocab_size or p.tok_embed.cols != d) return error.DimensionMismatch;

    // Replay the forward, keeping every block's intermediates. A tape would keep
    // each op's output separately, which is a wider set than the eleven tensors
    // `blockBackward` actually reads, so the replay keeps a `Block` per layer
    // instead. They are built once and the second loop below consumes them, so
    // the values the loss was built from are the values the gradients are
    // differentiated from. At the shipped shape that is 9.5 MiB, which is the
    // one place the hand-written decision shows up in the memory profile.
    var xs = try allocator.alloc(Tensor, p.layers.len + 1);
    defer allocator.free(xs);
    var kept: usize = 0;
    defer {
        for (xs[0..kept]) |*one| one.deinit();
    }
    var blocks = try allocator.alloc(Block, p.layers.len);
    defer allocator.free(blocks);
    var built: usize = 0;
    defer {
        for (blocks[0..built]) |*one| one.deinit();
    }
    xs[0] = try embed(allocator, p, tokens, d);
    kept = 1;
    for (p.layers) |l| {
        // Stored before the residual add, so a failed add unwinds the block that
        // exists rather than leaking it or half-freeing it.
        blocks[built] = try blockForward(allocator, l, xs[kept - 1], cfg);
        built += 1;
        // x_out = x_mid + ff, and x_mid already carries the attention branch.
        // Adding ff to the block input instead would drop that branch from the
        // stream every layer after the first.
        xs[kept] = try residual(allocator, blocks[built - 1].x_mid, blocks[built - 1].ff);
        kept += 1;
    }
    const stream = xs[kept - 1];

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

    // Each block, last to first, on the intermediates the replay above already
    // built. Recomputing them here would be a second forward per block and a
    // backward that costs more than the forward it differentiates.
    var layer = p.layers.len;
    while (layer > 0) {
        layer -= 1;
        // Two buffers, swapped. One would mean a memcpy of the whole stream per
        // layer, and one allocation per layer, for a tensor whose only reader is
        // the scatter at the bottom of this function.
        var d_next = try Tensor.init(allocator, t_count, d);
        defer d_next.deinit();
        try blockBackward(&d_next, p.layers[layer], &g.layers[layer], blocks[layer], xs[layer], d_x, cfg);
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

/// The eleven tensors one block's backward pass reads. `model.forward` builds
/// the same values in the same order and frees them again; this struct exists
/// because a hand-written backward has no tape to hang them on, so it rebuilds
/// them from the block input and keeps them, one per layer, for the second loop
/// in `backward` to read.
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
    ff: Tensor, // a @ w_down                           [T, d]

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
        self.ff.deinit();
    }
};

/// The forward half of one pre-norm block, in the order `model.forward` runs
/// it, so the backward below reads the same values the loss was built from.
fn blockForward(allocator: std.mem.Allocator, l: model.Layer, x: Tensor, cfg: model.Config) !Block {
    const t: [11]Tensor = blk: {
        var built: [11]Tensor = undefined;
        // A struct literal that fails part way through is never assigned, so an
        // errdefer on the result would never fire. The unwind walks the prefix
        // that exists instead.
        var n: usize = 0;
        errdefer for (built[0..n]) |*one| one.deinit();
        while (n < built.len) : (n += 1) {
            built[n] = try buildStep(allocator, l, x, cfg, built[0..n], n);
        }
        break :blk built;
    };
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
        .ff = t[10],
    };
}

fn buildStep(
    allocator: std.mem.Allocator,
    l: model.Layer,
    x: Tensor,
    cfg: model.Config,
    t: []Tensor,
    n: usize,
) !Tensor {
    return switch (n) {
        0 => norm.forward(allocator, x, l.attn_norm),
        1 => blk: {
            var q = try tensor.matmul(t[0], l.wq);
            defer q.deinit();
            break :blk try rope.forward(allocator, q, 0, rope_theta, cfg.head_dim);
        },
        2 => blk: {
            var k = try tensor.matmul(t[0], l.wk);
            defer k.deinit();
            break :blk try rope.forward(allocator, k, 0, rope_theta, cfg.head_dim);
        },
        3 => tensor.matmul(t[0], l.wv),
        4 => attention.forward(allocator, t[1], t[2], t[3], .{
            .n_heads = cfg.n_heads,
            .n_kv_heads = cfg.n_kv_heads,
            .head_dim = cfg.head_dim,
        }),
        5 => blk: {
            var proj = try tensor.matmul(t[4], l.wo);
            defer proj.deinit();
            break :blk try residual(allocator, x, proj);
        },
        6 => norm.forward(allocator, t[5], l.mlp_norm),
        7 => tensor.matmul(t[6], l.w_gate),
        8 => tensor.matmul(t[6], l.w_up),
        9 => swiglu(allocator, t[7], t[8]),
        10 => tensor.matmul(t[9], l.w_down),
        // The caller's loop walks 0 through 10, so no other value arrives. The
        // switch cannot be written as exhaustive over usize without a prong for
        // everything above 10, which the loop already rules out.
        else => unreachable,
    };
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

    for (0..cfg.n_heads) |h| {
        const kv = h / group;
        for (0..q.rows) |t| {
            const q_head = q.rowConst(t)[h * dim ..][0..dim];
            const dh = dout.rowConst(t)[h * dim ..][0..dim];

            // Rebuild the forward's softmax over the causal prefix.
            var row_max: f64 = -std.math.inf(f64);
            for (0..t + 1) |s| {
                const k_head = k.rowConst(s)[kv * dim ..][0..dim];
                var dot: f64 = 0;
                for (q_head, k_head) |a, b| dot += @as(f64, a) * @as(f64, b);
                probs[s] = dot * scale;
                row_max = @max(row_max, probs[s]);
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
            for (0..t + 1) |s| {
                const v_head = v.rowConst(s)[kv * dim ..][0..dim];
                var acc: f64 = 0;
                for (0..dim) |j| acc += @as(f64, dh[j]) * @as(f64, v_head[j]);
                d_probs[s] = acc;
                dot_pp += probs[s] * acc;
            }

            // d q[t] is written by this row alone, so it is assigned. d k and
            // d v accumulate over every row that reads the position.
            for (0..dim) |j| {
                var acc: f64 = 0;
                for (0..t + 1) |s| {
                    const k_head = k.rowConst(s)[kv * dim ..][0..dim];
                    acc += probs[s] * (d_probs[s] - dot_pp) * @as(f64, k_head[j]) * scale;
                }
                g.dq.set(t, h * dim + j, @floatCast(acc));
            }
            for (0..t + 1) |s| {
                const dk_row = g.dk.row(s);
                const dv_row = g.dv.row(s);
                for (0..dim) |j| {
                    dk_row[kv * dim + j] += @floatCast(probs[s] * (d_probs[s] - dot_pp) * @as(f64, q_head[j]) * scale);
                    dv_row[kv * dim + j] += @floatCast(probs[s] * @as(f64, dh[j]));
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
fn weightGrad(w: *Tensor, input: Tensor, dout: Tensor) void {
    for (0..input.cols) |i| {
        const w_row = w.row(i);
        for (0..input.rows) |t| {
            const scale = input.rowConst(t)[i];
            const dout_row = dout.rowConst(t);
            for (0..dout.cols) |j| w_row[j] += scale * dout_row[j];
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
fn inputGrad(out: *Tensor, dout: Tensor, w: Tensor) void {
    for (0..dout.rows) |t| {
        const dout_row = dout.rowConst(t);
        const out_row = out.row(t);
        for (0..w.rows) |i| {
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

fn swiglu(allocator: std.mem.Allocator, gate: Tensor, up: Tensor) !Tensor {
    const a = try Tensor.init(allocator, gate.rows, gate.cols);
    for (a.data, gate.data, up.data) |*dst, gate_z, up_z| dst.* = mlp.silu(gate_z) * up_z;
    return a;
}

/// out = base + branch, the residual add. It has no derivative of its own, which
/// is why the backward pass writes the stream gradient onto both operands
/// unchanged.
fn residual(allocator: std.mem.Allocator, base: Tensor, branch: Tensor) !Tensor {
    if (base.rows != branch.rows or base.cols != branch.cols) return error.DimensionMismatch;
    var out = try Tensor.init(allocator, base.rows, base.cols);
    for (0..out.rows) |r| {
        const b = base.rowConst(r);
        const br = branch.rowConst(r);
        const dst = out.row(r);
        for (0..out.cols) |i| dst[i] = b[i] + br[i];
    }
    return out;
}

/// x[t] = tok_embed[tokens[t]]. The gradient of this scatter is at the bottom
/// of `backward`; the forward half is here so the replay and the real forward
/// read the same rows.
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
