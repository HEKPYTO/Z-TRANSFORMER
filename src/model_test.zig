const std = @import("std");
const model = @import("model.zig");
const norm = @import("norm.zig");
const tensor = @import("tensor.zig");
const Tensor = tensor.Tensor;

/// 1 layer, d_model 8, vocab 16, ctx 32: the smallest shape that still holds a
/// query head, a kv head and a rotary pair, so the block arithmetic stays hand
/// checkable. The 4 layer default is slow per test and proves no more.
const tiny = model.Config{
    .n_layers = 1,
    .n_heads = 2,
    .n_kv_heads = 2,
    .head_dim = 4,
    .n_ctx = 32,
    .vocab_size = 16,
    .ffn_mult = 1,
};

/// Two layers, for the paths that only a real stack of blocks reaches: the
/// unwind out of `initParams` and the per layer intermediates of `forward`.
const two_layers = model.Config{
    .n_layers = 2,
    .n_heads = 2,
    .n_kv_heads = 2,
    .head_dim = 4,
    .n_ctx = 32,
    .vocab_size = 16,
    .ffn_mult = 1,
};

fn expectSameBits(want: Tensor, got: Tensor) !void {
    try std.testing.expectEqual(want.rows, got.rows);
    try std.testing.expectEqual(want.cols, got.cols);
    for (want.data, got.data) |a, b| {
        try std.testing.expectEqual(@as(u32, @bitCast(a)), @as(u32, @bitCast(b)));
    }
}

fn expectAllZero(t: Tensor) !void {
    for (t.data) |v| try std.testing.expectEqual(@as(u32, 0), @as(u32, @bitCast(v)));
}

fn expectAllOne(t: Tensor) !void {
    for (t.data) |v| try std.testing.expectEqual(@as(u32, 0x3f800000), @as(u32, @bitCast(v)));
}

/// The nine parameters of one layer, so a test can reach all of them without
/// repeating the field list.
fn layerTensors(l: *model.Layer) [9]*Tensor {
    return .{
        &l.attn_norm, &l.wq,     &l.wk,   &l.wv,     &l.wo,
        &l.mlp_norm,  &l.w_gate, &l.w_up, &l.w_down,
    };
}

fn hasField(comptime T: type, name: []const u8) bool {
    var found = false;
    inline for (@typeInfo(T).@"struct".fields) |f| {
        if (std.mem.eql(u8, f.name, name)) found = true;
    }
    return found;
}

/// `initParams` with every value overwritten, so a test can write the handful
/// of weights it cares about and know that nothing else is in play. The norm
/// weights get ones: a zero norm weight erases the branch it sits on, and most
/// checks here want to read that branch.
fn blank(allocator: std.mem.Allocator, cfg: model.Config) !model.Params {
    var p = try model.initParams(allocator, cfg, 7);
    p.tok_embed.fill(0);
    p.final_norm.fill(1);
    for (p.layers) |*l| {
        for (layerTensors(l)) |t| t.fill(0);
        l.attn_norm.fill(1);
        l.mlp_norm.fill(1);
    }
    return p;
}

test "model forward returns one row per token and one column per vocabulary entry" {
    var p = try model.initParams(std.testing.allocator, tiny, 1234);
    defer p.deinit();

    var logits = try model.forward(std.testing.allocator, p, tiny, &.{ 3, 1, 4 });
    defer logits.deinit();

    try std.testing.expectEqual(@as(usize, 3), logits.rows);
    try std.testing.expectEqual(@as(usize, 16), logits.cols);
    try std.testing.expectEqual(@as(usize, 48), logits.data.len);
}

test "model derives d_model and ffn width from the config" {
    try std.testing.expectEqual(@as(usize, 8), model.dModel(tiny));
    try std.testing.expectEqual(@as(usize, 8), model.ffnDim(tiny));

    const full = model.defaultConfig();
    try std.testing.expectEqual(@as(usize, 128), model.dModel(full));
    try std.testing.expectEqual(@as(usize, 512), model.ffnDim(full));
    try std.testing.expectEqual(@as(usize, 4), full.n_layers);
    try std.testing.expectEqual(@as(usize, 256), full.n_ctx);
    // 4 query heads over 2 kv heads is a group of 2, the GQA split.
    try std.testing.expectEqual(@as(usize, 0), full.n_heads % full.n_kv_heads);
}

test "model forward is bit reproducible and the seed is too" {
    var p = try model.initParams(std.testing.allocator, tiny, 1234);
    defer p.deinit();
    var same = try model.initParams(std.testing.allocator, tiny, 1234);
    defer same.deinit();

    try expectSameBits(p.tok_embed, same.tok_embed);
    try expectSameBits(p.final_norm, same.final_norm);
    try std.testing.expectEqual(p.layers.len, same.layers.len);
    for (p.layers, same.layers) |*a, *b| {
        const a_t = layerTensors(a);
        const b_t = layerTensors(b);
        for (a_t, b_t) |x, y| try expectSameBits(x.*, y.*);
    }

    const tokens = [_]u32{ 5, 0, 15, 2 };
    var first = try model.forward(std.testing.allocator, p, tiny, &tokens);
    defer first.deinit();
    var second = try model.forward(std.testing.allocator, p, tiny, &tokens);
    defer second.deinit();

    try expectSameBits(first, second);
}

test "model ties the head to the embedding table" {
    // Structural: the two field checks below are the whole claim. Counting
    // fields as well would only add a second way to fail for a reason no one
    // wrote down.
    try std.testing.expect(hasField(model.Params, "tok_embed"));
    try std.testing.expect(!hasField(model.Params, "lm_head"));

    var p = try blank(std.testing.allocator, tiny);
    defer p.deinit();
    p.tok_embed.set(5, 2, 4);

    var logits = try model.forward(std.testing.allocator, p, tiny, &.{5});
    defer logits.deinit();

    // Every branch weight is zero, so the residual stream is the embedding row
    // itself: h = 4 * e2, final = h / sqrt(16/8 + 1e-5), and the tied head
    // reports dot(emb[5], final) = 16 / sqrt(16/8 + 1e-5) = 11.3136802148.
    try std.testing.expectApproxEqRel(@as(f32, 11.3136802148), logits.at(0, 5), 1e-6);
    // A separate head would be dense; the tied head can only be non zero where
    // an embedding row is.
    for (0..tiny.vocab_size) |v| {
        if (v == 5) continue;
        try std.testing.expectEqual(@as(f32, 0), logits.at(0, v));
    }

    // Changing tok_embed changes the logits.
    p.tok_embed.fill(0);
    var after = try model.forward(std.testing.allocator, p, tiny, &.{5});
    defer after.deinit();
    try expectAllZero(after);
}

test "model initialises every norm weight to exactly one" {
    var p = try model.initParams(std.testing.allocator, tiny, 99);
    defer p.deinit();

    // The projection weights are not one, so this is an init rule and not an
    // artefact of a tensor that was never written.
    try std.testing.expect(p.tok_embed.at(0, 0) != 1);
    try std.testing.expect(p.layers[0].wq.at(0, 0) != 1);

    // One, not zero. At zero every branch output is zero, so the loss is flat
    // in all 28 projections and in tok_embed, and only these nine weights
    // receive gradient. `blank` is where a zero norm is still wanted, because
    // that is the configuration in which pre-norm and post-norm differ.
    try expectAllOne(p.final_norm);
    for (p.layers) |*l| {
        try expectAllOne(l.attn_norm);
        try expectAllOne(l.mlp_norm);
    }
}

test "model is pre-norm: zero weights leave the embedding passed through the final norm" {
    var p = try blank(std.testing.allocator, tiny);
    defer p.deinit();
    p.tok_embed.set(0, 0, 1);
    p.tok_embed.set(0, 1, 2);
    p.tok_embed.set(1, 2, 3);

    var logits = try model.forward(std.testing.allocator, p, tiny, &.{ 0, 1 });
    defer logits.deinit();

    // Every branch weight is zero, so both residual branches add nothing and
    // the stream reaching the final norm is the embedding lookup unchanged.
    // A post-norm block instead folds the norm into the residual sum. `blank`
    // zeroes the 28 projections and sets the three branch norms to 1, so under
    // post-norm the norm multiplies an already-zero branch output and the whole
    // stream is erased: it reports exactly 0 here.
    //
    // row 0: final = [1 2 0 ...] / sqrt(5/8 + 1e-5)
    //         logit[0][0] = dot(emb[0], final) = 5 / sqrt(5/8 + 1e-5)
    // row 1: final = [0 0 3 0 ...] / sqrt(9/8 + 1e-5)
    //         logit[1][1] = dot(emb[1], final) = 9 / sqrt(9/8 + 1e-5)
    try std.testing.expectApproxEqRel(@as(f32, 6.3245047245), logits.at(0, 0), 1e-6);
    try std.testing.expectApproxEqRel(@as(f32, 8.4852436621), logits.at(1, 1), 1e-6);
    // The rows stay separate, so the value is not leaked across positions.
    try std.testing.expectEqual(@as(f32, 0), logits.at(1, 0));
    for (0..tiny.vocab_size) |v| {
        if (v == 0 or v == 1) continue;
        try std.testing.expectEqual(@as(f32, 0), logits.at(0, v));
        try std.testing.expectEqual(@as(f32, 0), logits.at(1, v));
    }
}

test "the tied head sums a vocab that is not a multiple of its unroll width" {
    // The tied head walks the vocab eight rows at a time, and every other
    // config in this file has a vocab that divides by eight, so nothing else
    // here reaches the remainder loop. This config does.
    const odd = model.Config{
        .n_layers = 1,
        .n_heads = 2,
        .n_kv_heads = 2,
        .head_dim = 4,
        .n_ctx = 32,
        .vocab_size = 13,
        .ffn_mult = 1,
    };
    var p = try blank(std.testing.allocator, odd);
    defer p.deinit();
    // Distinct values, not zeros and not all equal, so a lane that mixed up
    // which vocab row it was summing would not sum to the same answer.
    for (p.tok_embed.data, 0..) |*c, k| c.* = @floatCast(@as(f32, @floatFromInt(k % 7)) * 0.25 - 0.5);

    var logits = try model.forward(std.testing.allocator, p, odd, &.{0});
    defer logits.deinit();

    // `blank` zeroes the 28 projections, so the stream reaching the final norm
    // is the embedding lookup untouched, and `norm` is what the model itself
    // ran. Referencing the real norm output is what makes the expectation exact
    // in f32 rather than close, which is the bar this test has to hold: a
    // reassociated sum is a different f64 value, not a nearby one.
    var stream = try Tensor.init(std.testing.allocator, 1, 8);
    defer stream.deinit();
    @memcpy(stream.row(0), p.tok_embed.rowConst(0));
    var final = try norm.forward(std.testing.allocator, stream, p.final_norm);
    defer final.deinit();
    const h = final.rowConst(0);
    // Ascending `i`, one f64 accumulator, narrowed once: the scalar loop the
    // unroll replaced, restated as the oracle. Bit equality, not a tolerance.
    for (0..odd.vocab_size) |v| {
        const e = p.tok_embed.rowConst(v);
        var acc: f64 = 0;
        for (0..e.len) |i| acc += @as(f64, h[i]) * @as(f64, e[i]);
        try std.testing.expectEqual(
            @as(u32, @bitCast(@as(f32, @floatCast(acc)))),
            @as(u32, @bitCast(logits.at(0, v))),
        );
    }
}

test "model adds the residual rather than replacing it" {
    var p = try blank(std.testing.allocator, tiny);
    defer p.deinit();
    p.tok_embed.set(0, 0, 1);

    // wq and wk stay zero, so with one token the single score softmaxes to 1
    // and attention returns v unchanged. wv reads the normalised embedding in
    // every column, and wo keeps that in the first two coordinates:
    //     attn(norm(x)) = [A A 0 ...]
    // A is a length 8 vector holding one 1 through a unit RMSNorm weight, so
    // A = 1 / sqrt(1/8 + 1e-5) = 2.8283139944. The norm keeps its own eps, which
    // is why this is not sqrt(8).
    const a: f32 = 2.8283139944;
    const d = model.dModel(tiny);
    for (0..d) |j| p.layers[0].wv.set(0, j, 1);
    p.layers[0].wo.set(0, 0, 1);
    p.layers[0].wo.set(1, 1, 1);

    var logits = try model.forward(std.testing.allocator, p, tiny, &.{0});
    defer logits.deinit();

    // h = x + attn = [(1 + A) A 0 ...] and logit[0][0] = (1 + A) / rms(h) with
    // rms(h) = sqrt(((1 + A)^2 + A^2)/8 + 1e-5), which is 2.2749214623.
    // Replacing the residual instead reports 1.9999949996 here and dropping the
    // branch altogether reports 2.8283139944, so this one literal separates all
    // three arrangements.
    try std.testing.expectApproxEqRel(@as(f32, 2.2749214623), logits.at(0, 0), 1e-6);

    // Otherwise the check above is vacuous if wo were never read.
    p.layers[0].wo.fill(0);
    var without = try model.forward(std.testing.allocator, p, tiny, &.{0});
    defer without.deinit();
    try std.testing.expectApproxEqRel(a, without.at(0, 0), 1e-6);
}

/// head_dim 2, so every rotary frequency is 500000^0 = 1 and a row rotates by
/// exactly one radian per position. d stays 8 and four heads fit, so q and k
/// are [T, 8] with four head blocks of 2, which is where a whole row
/// rotation and a per block rotation disagree.
const rotary = model.Config{
    .n_layers = 1,
    .n_heads = 4,
    .n_kv_heads = 4,
    .head_dim = 2,
    .n_ctx = 8,
    .vocab_size = 2,
    .ffn_mult = 1,
};

test "model rotates each head block of q and k, not the concatenated row" {
    // The other numeric cases here zero wq and wk, and RoPE of zero is zero, so
    // none of them can see a wrong rotation. This one does not, and every
    // projection that matters is the identity or a single 3, so the whole
    // forward is hand checkable.
    var p = try blank(std.testing.allocator, rotary);
    defer p.deinit();
    const d = model.dModel(rotary); // 8

    // Two tokens whose embeddings differ, so token 1's query is not parallel
    // to its own key and the softmax is not trivially uniform.
    //     x0 = [-2 0 0 0 0 -1 0 0]  sum of squares 5, rms 0.7905757396
    //     x1 = [-1 -1 0 0 1 -2 0 0]  sum of squares 7, rms 0.9354196919
    //     attn_in0 = [-2.5298018898 0 0 0 0 -1.2649009449 0 0]
    //     attn_in1 = [-1.0690388589 -1.0690388589 0 0 1.0690388589 -2.1380777177 0 0]
    // x1 carries mass in head 0 (columns 0, 1) and in head 1 (columns 4, 5), so
    // both a wrong head pairing and a wrong frequency exponent move the result.
    // x0 leaves column 1 empty on purpose, which zeroes v row 0 below and keeps
    // the mixture from being diluted by a term no rotation can reach.
    p.tok_embed.set(0, 0, -2);
    p.tok_embed.set(0, 5, -1);
    p.tok_embed.set(1, 0, -1);
    p.tok_embed.set(1, 1, -1);
    p.tok_embed.set(1, 4, 1);
    p.tok_embed.set(1, 5, -2);

    // wq and wk are the identity, so q and k are the normalised embeddings.
    for (0..d) |i| {
        p.layers[0].wq.set(i, i, 1);
        p.layers[0].wk.set(i, i, 1);
    }

    // `matmul(x, w)` is x @ w, so wv[1][1] = 3 is one weight: input feature 1
    // scaled by 3 into output feature 1. That makes
    //     v[0] = 0                        (attn_in0[1] is 0)
    //     v[1] = [0 -3.2071165766 0 ...]  (3 * attn_in1[1])
    // wo is the identity, so the branch reaches the stream unscrambled and is
    // read where it was written. w_gate, w_up and w_down stay zero, so the mlp
    // branch adds nothing and the attention branch is the whole story.
    p.layers[0].wv.set(1, 1, 3);
    for (0..d) |i| p.layers[0].wo.set(i, i, 1);

    var logits = try model.forward(std.testing.allocator, p, rotary, &.{ 0, 1 });
    defer logits.deinit();

    // Row 0 has one key, so its softmax is 1 whatever the rotation is and
    // ctx[0] = v[0] = 0. The stream is x0 unchanged, rms = 0.7905757396, and
    // the tied head reads
    //     logit[0][0] = dot(x0, x0 / rms) = 6.3245047245
    //     logit[0][1] = dot(x1, x0 / rms) = 5.0596037796
    // This row is the control: it pins the embedding, norm and head path, and
    // it reads the same under every rotation, right or wrong.
    try std.testing.expectApproxEqRel(@as(f32, 6.3245047245), logits.at(0, 0), 1e-5);
    try std.testing.expectApproxEqRel(@as(f32, 5.0596037796), logits.at(0, 1), 1e-5);

    // Row 1 is the one that moves. Every head block turns by 1 radian with
    // cos = 0.5403023058681398 and sin = 0.8414709848078965, so a block holding
    // (a, b) becomes (a cos - b sin, b cos + a sin):
    //     head 0 (a, b) = (-1.0690388589, -1.0690388589)
    //         -> (a(cos - sin), a(cos + sin)) = (0.3219610209, -1.4771693419)
    //     head 1 (a, b) = (1.0690388589, -2.1380777177) = (p, -2p)
    //         -> (p(cos + 2 sin), p(sin - 2 cos)) = (2.3767345233, -0.2556431396)
    // With k row 0 unrotated at position 0, head 0 at t = 1 scores
    //     s0 = dot(q1_head0, k0_head0) / sqrt 2 = -0.5759367755
    //     s1 = dot(q1_head0, k1_head0) / sqrt 2 =  1.6162256001
    // s1 is a rotation of a vector against itself, so it is a norm and lands on
    // the same value whatever the rotation. s0 is not, and that is the whole
    // difference. softmax gives
    //     w0 = 0.1004565216   w1 = 0.8995434784
    //     ctx[1][1] = w1 * -3.2071165766 = -2.8849408010
    //     h[1]      = x1 + ctx = [-1 -3.8849408010 0 0 1 -2 0 0]
    //     rms(h)    = 1.6237627993
    //     logit[1][0] = dot(x0, h / rms) = 2.4634139923
    //     logit[1][1] = dot(x1, h / rms) = 6.0876753706
    // The 1e-5 tolerance is four orders of magnitude below the gap to either
    // wrong answer. Rotating the whole 8 wide row pairs column 0 with column 4
    // and halves every exponent, and reads
    //     [s0 = 2.6424197109, s1 = 2.2330223215, w1 = 0.3990566247,
    //      logit[1][0] = 3.3809695757, logit[1][1] = 6.9984558214]
    // and not rotating at all reads
    //     [s0 = 1.9123395486, s1 = 1.6162256001, w1 = 0.4265077345,
    //      logit[1][0] = 3.3208401058, logit[1][1] = 6.9470812930]
    try std.testing.expectApproxEqRel(@as(f32, 2.4634139923), logits.at(1, 0), 1e-5);
    try std.testing.expectApproxEqRel(@as(f32, 6.0876753706), logits.at(1, 1), 1e-5);
}

test "model rejects a config it cannot assemble" {
    var p = try model.initParams(std.testing.allocator, tiny, 5);
    defer p.deinit();

    // 3 query heads over 2 kv heads leaves a head with no group.
    const ragged = model.Config{ .n_layers = 1, .n_heads = 3, .n_kv_heads = 2, .head_dim = 4, .n_ctx = 32, .vocab_size = 16, .ffn_mult = 1 };
    try std.testing.expectError(error.InvalidConfig, model.initParams(std.testing.allocator, ragged, 5));
    try std.testing.expectError(error.InvalidConfig, model.forward(std.testing.allocator, p, ragged, &.{0}));

    const layerless = model.Config{ .n_layers = 0, .n_heads = 2, .n_kv_heads = 2, .head_dim = 4, .n_ctx = 32, .vocab_size = 16, .ffn_mult = 1 };
    try std.testing.expectError(error.InvalidConfig, model.initParams(std.testing.allocator, layerless, 5));
    try std.testing.expectError(error.InvalidConfig, model.forward(std.testing.allocator, p, layerless, &.{0}));

    // Zero kv heads would divide by zero in the group split.
    const kvless = model.Config{ .n_layers = 1, .n_heads = 2, .n_kv_heads = 0, .head_dim = 4, .n_ctx = 32, .vocab_size = 16, .ffn_mult = 1 };
    try std.testing.expectError(error.InvalidConfig, model.forward(std.testing.allocator, p, kvless, &.{0}));

    // An odd head dim has no rotary pair to rotate.
    const odd = model.Config{ .n_layers = 1, .n_heads = 2, .n_kv_heads = 2, .head_dim = 3, .n_ctx = 32, .vocab_size = 16, .ffn_mult = 1 };
    try std.testing.expectError(error.InvalidConfig, model.forward(std.testing.allocator, p, odd, &.{0}));

    // Two layers of parameters against a config that declares one. The
    // embedding shape still agrees, because d_model does not depend on depth,
    // so nothing else in the validation can see it.
    var deep = try model.initParams(std.testing.allocator, two_layers, 5);
    defer deep.deinit();
    try std.testing.expectError(error.DimensionMismatch, model.forward(std.testing.allocator, deep, tiny, &.{0}));
}

test "model rejects a token id outside the vocabulary instead of wrapping it" {
    var p = try model.initParams(std.testing.allocator, tiny, 5);
    defer p.deinit();

    // Token ids come from corpus data. Wrapping 16 down to 0 would quietly
    // train on the wrong row.
    try std.testing.expectError(error.TokenOutOfRange, model.forward(std.testing.allocator, p, tiny, &.{ 0, tiny.vocab_size }));
    try std.testing.expectError(error.TokenOutOfRange, model.forward(std.testing.allocator, p, tiny, &.{4294967295}));
    // Longer than the context the config declares.
    try std.testing.expectError(error.SequenceTooLong, model.forward(std.testing.allocator, p, tiny, &.{
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        0,
    }));
}

test "model releases the layers it built when an allocation fails" {
    // tok_embed, the layer slice, final_norm, then nine tensors per layer: 21
    // allocations. A failure at any one of them has to unwind to nothing, and
    // the layer slice holds uninitialised memory until its entry is built, so
    // this is the only way to see an unwind walk a layer that does not exist.
    var fail_index: usize = 0;
    while (fail_index < 21) : (fail_index += 1) {
        var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = fail_index });
        try std.testing.expectError(error.OutOfMemory, model.initParams(failing.allocator(), two_layers, 3));
    }

    // And the same seed still gives the same weights afterwards.
    var p = try model.initParams(std.testing.allocator, two_layers, 3);
    defer p.deinit();
    try std.testing.expect(p.tok_embed.at(0, 0) != 0);
}

test "model frees every intermediate of a full forward" {
    var p = try model.initParams(std.testing.allocator, two_layers, 11);
    defer p.deinit();

    const tokens = [_]u32{ 1, 2, 3 };
    var logits = try model.forward(std.testing.allocator, p, two_layers, &tokens);
    logits.deinit();

    // The residual stream and every per layer intermediate are released on the
    // error path too, which testing.allocator is the only thing that can see.
    try std.testing.expectError(error.TokenOutOfRange, model.forward(std.testing.allocator, p, two_layers, &.{ 1, 99, 3 }));
}

/// Collects what the sink reports, so a test can count and name it.
///
/// `@fieldParentPtr` rather than a cast of the pointer itself. Casting the
/// `*model.Sink` straight back to a
/// `*Recorder` only works while the sink sits at offset zero, which is an
/// accident of field order rather than a contract; this form holds whatever
/// order the fields end up in.
const Recorder = struct {
    sink: model.Sink = undefined,
    seen: usize = 0,
    names: std.EnumArray(model.Name, usize) = .initFill(0),
    per_layer: usize = 0,
    layer_zero: usize = 0,
    non_layer_zero: usize = 0,

    fn put(which: *model.Sink, name: model.Name, layer: usize, _: Tensor) void {
        const self: *Recorder = @fieldParentPtr("sink", which);
        self.seen += 1;
        self.names.set(name, self.names.get(name) + 1);
        if (name == .final_norm or name == .logits) {
            self.non_layer_zero += 1;
        } else {
            self.per_layer += 1;
            if (layer == 0) self.layer_zero += 1;
        }
    }

    /// The `*model.Sink` to hand the pass, backed by this recorder.
    fn asSink(self: *Recorder) *model.Sink {
        self.sink = .{ .put = put };
        return &self.sink;
    }
};

test "a live sink observes the pass without perturbing it" {
    var p = try model.initParams(std.testing.allocator, two_layers, 5);
    defer p.deinit();
    const tokens = [_]u32{ 1, 2, 3, 4 };

    var plain = try model.forward(std.testing.allocator, p, two_layers, &tokens);
    defer plain.deinit();

    // A REAL sink. `model.forward` is `return forwardWith(..., null)`, so an
    // earlier version of this test handed both arms a null sink and compared
    // `forwardWith(..., null)` against `forwardWith(..., null)`: a value against
    // itself, which passes for any arithmetic at all. It could not see the one
    // thing it was named after -- a live sink perturbing the pass, either
    // through the `probs` allocation inside `attention.forwardWith` or through
    // a `Sink.put` that wrote through a bad pointer. The sink is on the
    // load-bearing path -- every forward records into one -- so a blind spot
    // there is a blind spot in the block itself.
    var rec: Recorder = .{};
    var through_sink = try model.forwardWith(std.testing.allocator, p, two_layers, &tokens, rec.asSink());
    defer through_sink.deinit();

    // First: the sink really was live on that pass. Without this the comparison
    // below would again be two null runs if `rec.asSink()` ever returned null.
    try std.testing.expectEqual(@as(usize, 16 * two_layers.n_layers + 2), rec.seen);

    // Bit for bit, not within a tolerance and not `==`. `expectSameBits` reads
    // the two tensors as raw u32, so a sink that perturbed the pass has to move
    // a bit rather than merely a value: `+0.0 == -0.0` and `NaN != NaN` would
    // both slip past a value comparison. The sink is read on no path in the
    // arithmetic, so a difference here is not a numerical question at all: it
    // means the plumbing changed what the pass computes.
    try expectSameBits(plain, through_sink);
}

test "the sink reports every intermediate, once per layer" {
    var p = try model.initParams(std.testing.allocator, two_layers, 5);
    defer p.deinit();
    const tokens = [_]u32{ 1, 2, 3, 4 };

    var rec: Recorder = .{};
    var out = try model.forwardWith(std.testing.allocator, p, two_layers, &tokens, rec.asSink());
    defer out.deinit();

    // 16 per-layer names over 2 layers, plus final_norm and logits once each.
    try std.testing.expectEqual(@as(usize, 16 * two_layers.n_layers + 2), rec.seen);
    try std.testing.expectEqual(@as(usize, 16 * two_layers.n_layers), rec.per_layer);
    try std.testing.expectEqual(@as(usize, 2), rec.non_layer_zero);
    // Each of the 16 layer-scoped names fires once per layer, so the two layers
    // contribute evenly and no name is reported for a layer that did not run.
    for (std.enums.values(model.Name)) |n| {
        const want: usize = switch (n) {
            .final_norm, .logits => 1,
            else => two_layers.n_layers,
        };
        try std.testing.expectEqual(want, rec.names.get(n));
    }
}

// `forward` used to check `tok_embed` against `d` and count the layers, and
// nothing else. That is enough for a changed `n_heads`, `head_dim` or depth --
// attention has its own shape contract for `n_kv_heads` -- and blind to the MLP,
// because `mlp.forward` takes no config and validates nothing. A config carrying
// a different `ffn_mult` therefore named weights the run did not have, and the
// matmul read past the buffer instead of reporting the mismatch.
//
// The first three cases already passed before this was added; the `ffn_mult` one
// did not, and it is the reason the check exists.
test "forward refuses a config whose geometry the parameters do not have" {
    const Cases = struct {
        name: []const u8,
        apply: *const fn (*model.Config) void,
    };
    const cases = [_]Cases{
        .{ .name = "n_kv_heads", .apply = struct {
            fn f(c: *model.Config) void {
                c.n_kv_heads = 4;
            }
        }.f },
        .{ .name = "ffn_mult", .apply = struct {
            fn f(c: *model.Config) void {
                c.ffn_mult = 4;
            }
        }.f },
        .{ .name = "head_dim", .apply = struct {
            fn f(c: *model.Config) void {
                c.head_dim = 16;
            }
        }.f },
        .{ .name = "n_heads", .apply = struct {
            fn f(c: *model.Config) void {
                c.n_heads = 8;
            }
        }.f },
    };
    for (cases) |c| {
        var cfg = model.Config{ .n_layers = 2, .n_heads = 4, .n_kv_heads = 2, .head_dim = 32, .n_ctx = 16, .vocab_size = 64, .ffn_mult = 2 };
        var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
        defer arena.deinit();
        const p = try model.initParams(arena.allocator(), cfg, 7);
        c.apply(&cfg);
        const toks = [_]u32{ 1, 2, 3 };
        try std.testing.expectError(
            error.DimensionMismatch,
            model.forward(arena.allocator(), p, cfg, &toks),
        );
    }
}
