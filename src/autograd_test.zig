//! Tests for the hand-written backward pass. The oracle is finite differences of
//! the real forward pass in `model.zig`; no test here compares a gradient
//! against a second backward pass written for the purpose.
const std = @import("std");
const model = @import("model.zig");
const autograd = @import("autograd.zig");
const gradcheck = @import("gradcheck.zig");
const loss = @import("loss.zig");
const tensor = @import("tensor.zig");
const Tensor = tensor.Tensor;

/// 1 layer, d_model 8, 2 heads, 2 kv heads, head_dim 4, vocab 16, ctx 32. The
/// same shape `model_test.zig` uses, for the same reason: the 4 layer default is
/// slow per test and proves no more.
pub const tiny = model.Config{
    .n_layers = 1,
    .n_heads = 2,
    .n_kv_heads = 2,
    .head_dim = 4,
    .n_ctx = 32,
    .vocab_size = 16,
    .ffn_mult = 1,
};

/// Two layers, for the paths only a real stack of blocks reaches. A backward
/// pass that handles the last layer and stops passes with one and fails here.
///
/// `n_kv_heads = 1` against two query heads, so `group = 2` and every query head
/// shares one kv head. With `n_kv_heads == n_heads` the grouping is the
/// identity: `kv = h / group` becomes `kv = h`, the dk/dv accumulation across
/// query heads sums one term instead of two, and none of the code that makes
/// grouped-query attention different from plain attention is ever differenced.
/// The parity harness is forward-only, so a regression here would have no
/// oracle at all and would ship green.
const two_layers = model.Config{
    .n_layers = 2,
    .n_heads = 2,
    .n_kv_heads = 1,
    .head_dim = 4,
    .n_ctx = 32,
    .vocab_size = 16,
    .ffn_mult = 1,
};

/// Four distinct tokens, so every position owns a tok_embed row of its own and
/// the gradient scatter is position resolvable.
const tok: []const u32 = &.{ 3, 1, 4, 0 };
const tgt: []const u32 = &.{ 1, 4, 0, 2 };

fn layerTensors(l: *model.Layer) [9]*Tensor {
    return .{
        &l.attn_norm, &l.wq,     &l.wk,   &l.wv,     &l.wo,
        &l.mlp_norm,  &l.w_gate, &l.w_up, &l.w_down,
    };
}

/// Every parameter drawn the way a trained model looks rather than a random draw
/// from the shipped initialisation, which is what this once described.
///
/// The reason it needed saying at all: an earlier version of `initParams` parked
/// both norm weights of every layer and the final norm at zero, and a zero norm
/// weight erases the branch it sits on, so the whole model collapsed to a
/// constant logit row and every gradient in it was exactly zero. Gradchecking
/// that proves nothing. `initParams` now sets the norm weights to one precisely
/// so this trap is not reachable, which `model.zig` explains at length; this
/// helper stays because a deterministic draw at stddev 0.5 is still not what
/// `initParams` produces, and it puts the logits at O(1) where the finite
/// difference has a usable signal to noise ratio.
pub fn liveParams(allocator: std.mem.Allocator, cfg: model.Config) !model.Params {
    var p = try model.initParams(allocator, cfg, 20260926);
    errdefer p.deinit();
    var prng = std.Random.DefaultPrng.init(0xbeef);
    const rnd = prng.random();
    for (p.tok_embed.data) |*v| v.* = @floatCast(0.5 * (rnd.float(f64) * 2.0 - 1.0));
    p.final_norm.fill(1);
    for (p.layers) |*l| {
        for (layerTensors(l)) |t| {
            for (t.data) |*v| v.* = @floatCast(0.5 * (rnd.float(f64) * 2.0 - 1.0));
        }
        l.attn_norm.fill(1);
        l.mlp_norm.fill(1);
    }
    return p;
}

/// Forward, dLoss/dlogits, backward: the whole path a training step takes.
fn lossAndGrads(
    allocator: std.mem.Allocator,
    cfg: model.Config,
    p: model.Params,
    tokens: []const u32,
    targets: []const u32,
    g: *autograd.Grads,
) !f64 {
    var logits = try model.forward(allocator, p, cfg, tokens);
    defer logits.deinit();
    var dl = try autograd.dLossDLogits(allocator, logits, targets);
    defer dl.deinit();
    try autograd.backward(allocator, p, g, cfg, tokens, dl);
    return loss.forward(logits, targets);
}

/// Backward from a dlogits the caller fixed, so the output gradient can be held
/// still while the parameters change.
fn gradsFrom(
    allocator: std.mem.Allocator,
    cfg: model.Config,
    p: model.Params,
    tokens: []const u32,
    dl: Tensor,
    g: *autograd.Grads,
) !void {
    try autograd.backward(allocator, p, g, cfg, tokens, dl);
}

fn dlogitsOf(allocator: std.mem.Allocator, cfg: model.Config, p: model.Params, tokens: []const u32, targets: []const u32) !Tensor {
    var logits = try model.forward(allocator, p, cfg, tokens);
    defer logits.deinit();
    return autograd.dLossDLogits(allocator, logits, targets);
}

fn expectAllZero(s: []const f32) !void {
    for (s) |v| try std.testing.expectEqual(@as(u32, 0), @as(u32, @bitCast(v)));
}

fn expectDiffers(a: []const f32, b: []const f32) !void {
    try std.testing.expectEqual(a.len, b.len);
    for (a, b) |x, y| {
        if (@as(u32, @bitCast(x)) != @as(u32, @bitCast(y))) return;
    }
    std.debug.print("\nall {d} elements were bit identical, expected a difference\n", .{a.len});
    return error.TestUnexpectedResult;
}

test "autograd: gradcheck every parameter element on the tiny config" {
    var p = try liveParams(std.testing.allocator, tiny);
    defer p.deinit();
    try gradcheck.checkAll(std.testing.allocator, tiny, p, tok, tgt);
}

test "autograd: gradcheck passes for two layers" {
    var p = try liveParams(std.testing.allocator, two_layers);
    defer p.deinit();
    try gradcheck.checkAll(std.testing.allocator, two_layers, p, tok, tgt);
}

test "autograd: gradcheck passes when every norm weight is one" {
    var p = try liveParams(std.testing.allocator, two_layers);
    defer p.deinit();
    for (p.layers) |*l| {
        l.attn_norm.fill(1);
        l.mlp_norm.fill(1);
    }
    p.final_norm.fill(1);
    try gradcheck.checkAll(std.testing.allocator, two_layers, p, tok, tgt);
}

test "autograd: gradcheck passes when every norm weight is zero" {
    // A zero norm weight erases the branch it sits on, so both residual branches
    // add nothing, every logit is 0 and the loss is log(vocab). The check is
    // here because it is the degenerate end of the same code path: rms still
    // carries eps, so a backward that dropped it would report a NaN rather than
    // a zero and fail here.
    //
    // The layer gradients are then exactly zero, because a gradient that reaches
    // the stream only through a zeroed norm is zero. final_norm's is not: the
    // final norm's input is the embedding lookup itself, which the zeroing does
    // not touch, so the tied head still feeds gradient through it.
    var p = try liveParams(std.testing.allocator, tiny);
    defer p.deinit();
    for (p.layers) |*l| {
        l.attn_norm.fill(0);
        l.mlp_norm.fill(0);
    }
    p.final_norm.fill(0);
    try gradcheck.checkAll(std.testing.allocator, tiny, p, tok, tgt);

    var g = try autograd.zeroGrads(std.testing.allocator, p);
    defer g.deinit();
    _ = try lossAndGrads(std.testing.allocator, tiny, p, tok, tgt, &g);
    for (g.layers) |lg| {
        try expectAllZero(lg.wq.data);
        try expectAllZero(lg.attn_norm.data);
        try expectAllZero(lg.w_down.data);
    }
    // finite, and not the zero a backward that dropped the head would report
    for (g.final_norm.data) |v| try std.testing.expect(std.math.isFinite(v));
    var live: f32 = 0;
    for (g.final_norm.data) |v| live += @abs(v);
    try std.testing.expect(live > 0);
}

test "autograd: a failed loss evaluation leaves the parameter it moved untouched" {
    // `compare` moves one parameter element by the step, evaluates the loss, and
    // puts it back. An evaluation that fails before the second one has to put it
    // back anyway, or a caller that catches the error and carries on trains from
    // a parameter that is one step off, in a direction it never asked for.
    var p = try liveParams(std.testing.allocator, tiny);
    defer p.deinit();
    var g = try autograd.zeroGrads(std.testing.allocator, p);
    defer g.deinit();
    _ = try lossAndGrads(std.testing.allocator, tiny, p, tok, tgt, &g);

    // The first target is out of range, so the very first loss evaluation fails,
    // which is the one taken while tok_embed[0] is a step off its true value.
    const before = p.tok_embed.data[0];
    try std.testing.expectError(error.TargetOutOfRange, gradcheck.compare(std.testing.allocator, tiny, p, tok, &.{ tiny.vocab_size, 0, 0, 0 }, &g));
    try std.testing.expectEqual(@as(u32, @bitCast(before)), @as(u32, @bitCast(p.tok_embed.data[0])));
}

test "autograd: a wrong gradient is named, with its index" {
    // The silence on a passing run is only worth anything if a failing run
    // still says what went wrong, so the failure path needs a gradient that is
    // actually wrong. The corruption has to be small enough to prove the budget
    // discriminates and large enough to be past it, which at one percent of the
    // element is the case on the largest one: the old budget was nine percent of
    // the element wide and swallowed a one percent error outright.
    var p = try liveParams(std.testing.allocator, tiny);
    defer p.deinit();
    var g = try autograd.zeroGrads(std.testing.allocator, p);
    defer g.deinit();
    _ = try lossAndGrads(std.testing.allocator, tiny, p, tok, tgt, &g);

    // The largest gradient in the tensor, so one percent of it is as far above
    // the floor as an element gets. An element below the floor cannot be
    // resolved in f32 at all, and corrupting one of those proves nothing: no
    // derived budget can see it, and a chosen one that could would be a chosen
    // one.
    var wrong: usize = 0;
    for (g.tok_embed.data, 0..) |v, i| {
        if (@abs(@as(f64, @floatCast(v))) > @abs(@as(f64, @floatCast(g.tok_embed.data[wrong])))) wrong = i;
    }
    const truth = g.tok_embed.data[wrong];
    const one_percent = 0.01 * @as(f64, @floatCast(truth));

    // The positive half, first: the untouched gradient is accepted. A check that
    // rejected everything would satisfy the rest of this test just as well.
    const clean = try gradcheck.compare(std.testing.allocator, tiny, p, tok, tgt, &g);
    defer clean.deinit();
    try std.testing.expectEqual(@as(?gradcheck.Mismatch, null), clean.mismatch);
    // And the floor it was accepted against has to be the derived one, small
    // enough that one percent of this element is well past it. This is the
    // assertion that fails if the budget is loosened again.
    try std.testing.expect(one_percent > 10 * clean.floor);

    g.tok_embed.data[wrong] = truth + @as(f32, @floatCast(one_percent));

    const r = try gradcheck.compare(std.testing.allocator, tiny, p, tok, tgt, &g);
    defer r.deinit();

    const m = r.mismatch orelse {
        std.debug.print("\ntok_embed[{d}] wrong by 1% was not reported, floor {e}\n", .{ wrong, r.floor });
        return error.TestUnexpectedResult;
    };
    try std.testing.expectEqual(@as(?usize, null), m.layer);
    try std.testing.expectEqualStrings("tok_embed", m.field);
    try std.testing.expectEqual(wrong, m.index);
    // The gap is the corruption, and the finite difference still reads the
    // truth, both to within the same budget the check itself used.
    try std.testing.expect(m.diff > m.budget);
    try std.testing.expect(@abs(m.diff - one_percent) < m.budget);
    try std.testing.expect(@abs(m.numeric - @as(f64, @floatCast(truth))) < m.budget);

    const text = try gradcheck.line(std.testing.allocator, m);
    defer std.testing.allocator.free(text);
    var want: [64]u8 = undefined;
    const prefix = try std.fmt.bufPrint(&want, "gradient mismatch at tok_embed[{d}]:", .{wrong});
    try std.testing.expect(std.mem.startsWith(u8, text, prefix));
}

test "autograd: the tied head's two paths into tok_embed are separable" {
    // The easiest thing to get wrong here is tok_embed's two paths, so this
    // separates them without a second backward pass.
    //
    // logits[t][v] = dot(tok_embed[v], final_h[t]) gives the output path
    //     d tok_embed[v] = sum_t dlogits[t][v] * final_h[t]
    // and x[t] = tok_embed[tokens[t]] gives the input path
    //     d tok_embed[token] += d x[t]
    //
    // A softmax hands every vocabulary row a non zero dlogits, so no row's
    // gradient is ever exactly zero and "an unused row stays at zero" is false
    // for this head. What is checkable is the split: with one token and one
    // target that no token equals, a row that is neither carries the output path
    // alone, and the token's own row carries the output path plus whatever the
    // input path adds. Subtracting the output path, which the forward's own
    // softmax gives, leaves the input path, and that has to be non zero.
    const cfg = tiny;
    var p = try liveParams(std.testing.allocator, cfg);
    defer p.deinit();
    const one: []const u32 = &.{7};
    const away: []const u32 = &.{2};

    var g = try autograd.zeroGrads(std.testing.allocator, p);
    defer g.deinit();
    var logits = try model.forward(std.testing.allocator, p, cfg, one);
    defer logits.deinit();
    var dl = try autograd.dLossDLogits(std.testing.allocator, logits, away);
    defer dl.deinit();
    try autograd.backward(std.testing.allocator, p, &g, cfg, one, dl);

    // final_h is the model's, so the output path for any row is read straight
    // off the gradient tensor: p_row * final_h, with p_row the softmax entry
    // dlogits reports for that row.
    var final_h = try finalHidden(std.testing.allocator, p, cfg, one);
    defer final_h.deinit();

    // Row 2 is the target and row 7 is the token, so rows 0,1 and 3..15 are
    // reached by the output path alone.
    for ([_]usize{ 0, 1, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15 }) |v| {
        const coef = dl.at(0, @intCast(v));
        for (0..model.dModel(cfg)) |i| {
            // No second term: this row is read by no other path.
            try std.testing.expectApproxEqRel(coef * final_h.at(0, i), g.tok_embed.at(v, i), 1e-5);
        }
    }

    // The token's own row is the one that has to carry the input path too.
    const coef = dl.at(0, 7);
    var input_path: f32 = 0;
    for (0..model.dModel(cfg)) |i| {
        input_path += g.tok_embed.at(7, i) - coef * final_h.at(0, i);
    }
    try std.testing.expect(@abs(input_path) > 0);

    // And the finite difference agrees with the whole thing, which is the part
    // that matters: the split above is a property, the gradcheck is the proof.
    try gradcheck.checkAll(std.testing.allocator, cfg, p, one, away);
}

/// The final hidden state the tied head sees, rebuilt from the stream the same
/// way `model.forward` builds it. The stream is the embedding lookup plus one
/// block, which `model.forward` does not hand back, so this recomputes it from
/// the public ops in the same order.
fn finalHidden(allocator: std.mem.Allocator, p: model.Params, cfg: model.Config, tokens: []const u32) !tensor.Tensor {
    const norm = @import("norm.zig");
    const rope = @import("rope.zig");
    const mlp = @import("mlp.zig");
    const attention = @import("attention.zig");
    const d = model.dModel(cfg);

    var x = try tensor.Tensor.init(allocator, tokens.len, d);
    defer x.deinit();
    for (tokens, 0..) |id, t| @memcpy(x.row(t), p.tok_embed.rowConst(@intCast(id)));

    for (p.layers) |*l| {
        var attn_in = try norm.forward(allocator, x, l.attn_norm);
        defer attn_in.deinit();
        var q = try tensor.matmul(attn_in, l.wq);
        defer q.deinit();
        var k = try tensor.matmul(attn_in, l.wk);
        defer k.deinit();
        var v = try tensor.matmul(attn_in, l.wv);
        defer v.deinit();
        var qp = try rope.forward(allocator, q, 0, 500000, cfg.head_dim);
        defer qp.deinit();
        var kp = try rope.forward(allocator, k, 0, 500000, cfg.head_dim);
        defer kp.deinit();
        var ctx = try attention.forward(allocator, qp, kp, v, .{
            .n_heads = cfg.n_heads,
            .n_kv_heads = cfg.n_kv_heads,
            .head_dim = cfg.head_dim,
        });
        defer ctx.deinit();
        var proj = try tensor.matmul(ctx, l.wo);
        defer proj.deinit();
        for (0..x.rows) |t| {
            for (0..d) |i| x.set(t, i, x.at(t, i) + proj.at(t, i));
        }
        var mlp_in = try norm.forward(allocator, x, l.mlp_norm);
        defer mlp_in.deinit();
        var ff = try mlp.forward(allocator, mlp_in, l.w_gate, l.w_up, l.w_down);
        defer ff.deinit();
        for (0..x.rows) |t| {
            for (0..d) |i| x.set(t, i, x.at(t, i) + ff.at(t, i));
        }
    }
    return norm.forward(allocator, x, p.final_norm);
}

test "autograd: the reverse pass is causal" {
    // Position 0's loss does not depend on the stream at any later position, so
    // the gradient at a later position must not move when one row of dlogits is
    // zeroed. A reverse pass that read past the causal mask picks position 0 up
    // again and fails here.
    //
    // The only position resolvable output is the tok_embed scatter, and the tied
    // head's output path sums over every position, so the whole gradient of a
    // row does move. What must not move is the part that is not the output path.
    // The output path is linear in the row's dlogits coefficient, so a row that
    // is not a token measures the change exactly, and a later position's token
    // row has to move by that same change scaled by its own coefficient. The
    // residual after that scaling is the leak this is looking for, and it is
    // zero to f32 rounding. The token at position 0 does carry a leak, because
    // its own stream gradient is what changed.
    const cfg = two_layers;
    var p = try liveParams(std.testing.allocator, cfg);
    defer p.deinit();
    var dl = try dlogitsOf(std.testing.allocator, cfg, p, tok, tgt);
    defer dl.deinit();

    var g1 = try autograd.zeroGrads(std.testing.allocator, p);
    defer g1.deinit();
    try gradsFrom(std.testing.allocator, cfg, p, tok, dl, &g1);

    var no_first = try Tensor.init(std.testing.allocator, dl.rows, dl.cols);
    defer no_first.deinit();
    @memcpy(no_first.data, dl.data);
    @memset(no_first.row(0), 0);

    var g2 = try autograd.zeroGrads(std.testing.allocator, p);
    defer g2.deinit();
    try gradsFrom(std.testing.allocator, cfg, p, tok, no_first, &g2);

    // Row 2 is a target and no token, so its gradient is the output path alone
    // and its dlogits coefficient is the largest in the row.
    const ref: usize = 2;
    const d = model.dModel(cfg);
    const coef_ref = dl.at(0, @intCast(ref));
    try std.testing.expect(coef_ref != 0);
    for (tok[1..]) |token| {
        // The output path scales with this row's own dlogits coefficient.
        const scale = @as(f32, @floatCast(dl.at(0, @intCast(token)))) / coef_ref;
        for (0..d) |i| {
            const measured = g1.tok_embed.at(ref, i) - g2.tok_embed.at(ref, i);
            const moved = g1.tok_embed.at(@intCast(token), i) - g2.tok_embed.at(@intCast(token), i);
            try std.testing.expectApproxEqRel(scale * measured, moved, 1e-4);
        }
    }
    // The token at position 0 moves by more than its output path, because the
    // stream at position 0 is what the removed row acted on.
    var own: f32 = 0;
    for (0..d) |i| {
        const measured = g1.tok_embed.at(ref, i) - g2.tok_embed.at(ref, i);
        const scale = @as(f32, @floatCast(dl.at(0, 0))) / coef_ref;
        const moved = g1.tok_embed.at(0, i) - g2.tok_embed.at(0, i);
        own += @abs(moved - scale * measured);
    }
    try std.testing.expect(own > 0);

    // Changing k and v, which the key and value weights do, has to change the
    // key and value gradients. dk and dv reach the parameters through those two
    // tensors and nowhere else, in every layer, since the weights are shared.
    p.layers[0].wk.set(2, 1, 0.7);
    p.layers[0].wv.set(3, 2, -0.4);
    var g3 = try autograd.zeroGrads(std.testing.allocator, p);
    defer g3.deinit();
    try gradsFrom(std.testing.allocator, cfg, p, tok, dl, &g3);
    try expectDiffers(g1.layers[0].wk.data, g3.layers[0].wk.data);
    try expectDiffers(g1.layers[0].wv.data, g3.layers[0].wv.data);
    try expectDiffers(g1.layers[1].wk.data, g3.layers[1].wk.data);
    try expectDiffers(g1.layers[1].wv.data, g3.layers[1].wv.data);
}

test "autograd: zeroGrads is exactly zero" {
    var p = try liveParams(std.testing.allocator, two_layers);
    defer p.deinit();
    var g = try autograd.zeroGrads(std.testing.allocator, p);
    defer g.deinit();
    try expectAllZero(g.tok_embed.data);
    try expectAllZero(g.final_norm.data);
    for (g.layers) |lg| {
        try expectAllZero(lg.attn_norm.data);
        try expectAllZero(lg.wq.data);
        try expectAllZero(lg.wk.data);
        try expectAllZero(lg.wv.data);
        try expectAllZero(lg.wo.data);
        try expectAllZero(lg.mlp_norm.data);
        try expectAllZero(lg.w_gate.data);
        try expectAllZero(lg.w_up.data);
        try expectAllZero(lg.w_down.data);
    }
    // A seeded value has to survive a backward, which is accumulation seen
    // from the other side.
    const seed: f32 = 0.25;
    for (g.layers[0].wq.data) |*v| v.* = seed;
    _ = try lossAndGrads(std.testing.allocator, two_layers, p, tok, tgt, &g);
    for (g.layers[0].wq.data) |v| try std.testing.expect(v != seed);
}

test "autograd: backward adds to what is already in the gradient tensor" {
    var p = try liveParams(std.testing.allocator, two_layers);
    defer p.deinit();
    const a_tok: []const u32 = &.{ 3, 1, 4, 0 };
    const a_tgt: []const u32 = &.{ 1, 4, 0, 2 };
    const b_tok: []const u32 = &.{ 5, 2, 9, 6 };
    const b_tgt: []const u32 = &.{ 0, 1, 2, 3 };

    var g = try autograd.zeroGrads(std.testing.allocator, p);
    defer g.deinit();
    _ = try lossAndGrads(std.testing.allocator, two_layers, p, a_tok, a_tgt, &g);
    const one = try std.testing.allocator.dupe(f32, g.tok_embed.data);
    defer std.testing.allocator.free(one);

    _ = try lossAndGrads(std.testing.allocator, two_layers, p, b_tok, b_tgt, &g);
    const two = try std.testing.allocator.dupe(f32, g.tok_embed.data);
    defer std.testing.allocator.free(two);

    // The second call has to leave the sum of the two batches, so a fresh
    // gradient for the second batch alone, added to the first, is what is there.
    // Compared with a relative tolerance and not bit for bit: the first call adds
    // its terms onto the existing value one at a time, while `one + two` adds
    // them onto a value that is already the whole second gradient, and f32 does
    // not reassociate.
    var g2 = try autograd.zeroGrads(std.testing.allocator, p);
    defer g2.deinit();
    _ = try lossAndGrads(std.testing.allocator, two_layers, p, b_tok, b_tgt, &g2);
    for (one, two, g2.tok_embed.data) |a, b, c| {
        // two is the sum, one is the first batch alone, c is the second alone.
        try std.testing.expectApproxEqRel(a + c, b, 1e-5);
    }
}

test "autograd: dLossDLogits on uniform logits is the closed form" {
    // loss_t = logsumexp - logit[target], so dlogits[t][v] = (p[t][v] - onehot) / T.
    // Uniform logits over V=4 give p = 1/4 everywhere, T = 2, targets {0, 1}:
    //     row 0 = (-0.375, 0.125, 0.125, 0.125)
    //     row 1 = ( 0.125, -0.375, 0.125, 0.125)
    var logits = try Tensor.init(std.testing.allocator, 2, 4);
    defer logits.deinit();
    logits.fill(0);

    var dl = try autograd.dLossDLogits(std.testing.allocator, logits, &.{ 0, 1 });
    defer dl.deinit();

    try std.testing.expectEqual(@as(usize, 2), dl.rows);
    try std.testing.expectEqual(@as(usize, 4), dl.cols);
    try std.testing.expectEqual(@as(f32, -0.375), dl.at(0, 0));
    try std.testing.expectEqual(@as(f32, 0.125), dl.at(0, 1));
    try std.testing.expectEqual(@as(f32, 0.125), dl.at(0, 2));
    try std.testing.expectEqual(@as(f32, 0.125), dl.at(0, 3));
    try std.testing.expectEqual(@as(f32, 0.125), dl.at(1, 0));
    try std.testing.expectEqual(@as(f32, -0.375), dl.at(1, 1));
    try std.testing.expectEqual(@as(f32, 0.125), dl.at(1, 2));
    try std.testing.expectEqual(@as(f32, 0.125), dl.at(1, 3));
}

test "autograd: dLossDLogits rejects a target it cannot use" {
    var logits = try Tensor.init(std.testing.allocator, 2, 4);
    defer logits.deinit();
    try std.testing.expectError(error.TargetOutOfRange, autograd.dLossDLogits(std.testing.allocator, logits, &.{ 0, 4 }));
    try std.testing.expectError(error.TargetOutOfRange, autograd.dLossDLogits(std.testing.allocator, logits, &.{ 0, 4294967295 }));
    try std.testing.expectError(error.TargetCountMismatch, autograd.dLossDLogits(std.testing.allocator, logits, &.{0}));
    var empty = try Tensor.init(std.testing.allocator, 0, 4);
    defer empty.deinit();
    try std.testing.expectError(error.EmptyBatch, autograd.dLossDLogits(std.testing.allocator, empty, &.{}));
}

test "autograd: a corrupt target is an error from loss to gradient, never a panic" {
    var p = try liveParams(std.testing.allocator, tiny);
    defer p.deinit();
    var g = try autograd.zeroGrads(std.testing.allocator, p);
    defer g.deinit();
    // The tokens are in range and the target is not, which is the shape a
    // corrupt corpus row takes. It has to surface as an error, and the gradient
    // buffer allocated for the attempt has to come back.
    try std.testing.expectError(error.TargetOutOfRange, lossAndGrads(std.testing.allocator, tiny, p, tok, &.{ 1, 4, 0, tiny.vocab_size }, &g));
    try expectAllZero(g.tok_embed.data);
}

test "autograd: backward rejects a token or a gradient it cannot use" {
    var p = try liveParams(std.testing.allocator, tiny);
    defer p.deinit();
    var g = try autograd.zeroGrads(std.testing.allocator, p);
    defer g.deinit();
    var dl = try dlogitsOf(std.testing.allocator, tiny, p, tok, tgt);
    defer dl.deinit();

    // The token index is a trust boundary: the embedding scatter reads it.
    try std.testing.expectError(error.TokenOutOfRange, autograd.backward(std.testing.allocator, p, &g, tiny, &.{ 0, 1, 4, tiny.vocab_size }, dl));
    // Longer than the context the config declares. The length is checked before
    // the gradient's shape, so a gradient built for four rows still reports the
    // context rather than the mismatch.
    const over = try std.testing.allocator.alloc(u32, tiny.n_ctx + 1);
    defer std.testing.allocator.free(over);
    for (over) |*t| t.* = 0;
    try std.testing.expectError(error.SequenceTooLong, autograd.backward(std.testing.allocator, p, &g, tiny, over, dl));
    // A gradient whose row count is not the token count.
    try std.testing.expectError(error.DimensionMismatch, autograd.backward(std.testing.allocator, p, &g, tiny, &.{ 0, 1 }, dl));
    // A gradient built for a wider vocabulary.
    var wide = try Tensor.init(std.testing.allocator, tok.len, tiny.vocab_size * 2);
    defer wide.deinit();
    try std.testing.expectError(error.DimensionMismatch, autograd.backward(std.testing.allocator, p, &g, tiny, tok, wide));
    // And the last good call still works after every one of those.
    try autograd.backward(std.testing.allocator, p, &g, tiny, tok, dl);
}

test "autograd: a full forward, backward and deinit leaks nothing" {
    var p = try liveParams(std.testing.allocator, two_layers);
    defer p.deinit();
    var g = try autograd.zeroGrads(std.testing.allocator, p);
    defer g.deinit();

    _ = try lossAndGrads(std.testing.allocator, two_layers, p, tok, tgt, &g);

    // And on the error path, which is where the per layer unwind has to work.
    try std.testing.expectError(error.TokenOutOfRange, lossAndGrads(std.testing.allocator, two_layers, p, &.{ 0, 1, tiny.vocab_size }, tgt, &g));
}

test "autograd: every allocation point on the loss to gradient path unwinds to nothing" {
    // backward allocates the layer inputs it replays, a block of eleven tensors
    // per layer twice over, and a scratch tensor per gradient. An unwind that
    // misses one of them leaks on every failing path, and only testing.allocator
    // can see it, so every allocation point is induced to fail in turn.
    var p = try liveParams(std.testing.allocator, two_layers);
    defer p.deinit();
    var g = try autograd.zeroGrads(std.testing.allocator, p);
    defer g.deinit();
    try std.testing.checkAllAllocationFailures(std.testing.allocator, lossAndGradsVoid, .{ two_layers, p, tok, tgt, &g });
}

fn lossAndGradsVoid(
    allocator: std.mem.Allocator,
    cfg: model.Config,
    p: model.Params,
    tokens: []const u32,
    targets: []const u32,
    g: *autograd.Grads,
) !void {
    _ = try lossAndGrads(allocator, cfg, p, tokens, targets, g);
}
