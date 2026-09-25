const std = @import("std");
const model = @import("model.zig");
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
