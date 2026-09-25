const std = @import("std");
const attention = @import("attention.zig");
const tensor = @import("tensor.zig");

const Tensor = tensor.Tensor;

/// Row-major input builder. `vals` must hold exactly `rows * cols` elements, so a
/// mistyped literal trips the bounds check instead of silently padding.
fn filled(allocator: std.mem.Allocator, rows: usize, cols: usize, vals: []const f32) !Tensor {
    var t = try Tensor.init(allocator, rows, cols);
    for (vals, 0..) |val, i| t.data[i] = val;
    return t;
}

test "attention one head one token returns v exactly" {
    var q = try filled(std.testing.allocator, 1, 2, &.{ 1, 2 });
    defer q.deinit();
    var k = try filled(std.testing.allocator, 1, 2, &.{ 3, 4 });
    defer k.deinit();
    var v = try filled(std.testing.allocator, 1, 2, &.{ 5, 6 });
    defer v.deinit();

    var out = try attention.forward(std.testing.allocator, q, k, v, .{ .n_heads = 1, .n_kv_heads = 1, .head_dim = 2 });
    defer out.deinit();

    // One score softmaxes to exactly 1, so the output is v with no rounding.
    try std.testing.expectEqualSlices(f32, v.data, out.data);
}

test "attention two tokens one head matches hand computed values" {
    // q [1 0]  k [1 0]  v [1 0]   t=0: score 1/sqrt(2) over one term -> p = 1 -> v0
    //  [0 0]   [0 0]    [3 4]   t=1: scores [0, 0] -> p = [1/2, 1/2] -> (v0 + v1)/2
    var q = try filled(std.testing.allocator, 2, 2, &.{ 1, 0, 0, 0 });
    defer q.deinit();
    var k = try filled(std.testing.allocator, 2, 2, &.{ 1, 0, 0, 0 });
    defer k.deinit();
    var v = try filled(std.testing.allocator, 2, 2, &.{ 1, 0, 3, 4 });
    defer v.deinit();

    var out = try attention.forward(std.testing.allocator, q, k, v, .{ .n_heads = 1, .n_kv_heads = 1, .head_dim = 2 });
    defer out.deinit();

    try std.testing.expectEqualSlices(f32, &.{ 1, 0, 2, 2 }, out.data);
}

test "attention ignores keys and values after the query position" {
    const cfg = attention.Config{ .n_heads = 2, .n_kv_heads = 1, .head_dim = 2 };
    var q = try filled(std.testing.allocator, 3, 4, &.{ 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, -1 });
    defer q.deinit();
    var k = try filled(std.testing.allocator, 3, 2, &.{ 1, 0, 0, 1, 1, 1 });
    defer k.deinit();
    var v = try filled(std.testing.allocator, 3, 2, &.{ 1, 2, 3, 4, 5, 6 });
    defer v.deinit();

    var before = try attention.forward(std.testing.allocator, q, k, v, cfg);
    defer before.deinit();

    k.set(1, 0, 99);
    k.set(1, 1, 98);
    v.set(1, 0, 97);
    v.set(1, 1, 96);

    var after = try attention.forward(std.testing.allocator, q, k, v, cfg);
    defer after.deinit();

    // Row 0 attends to row 0 alone, so no edit at row 1 can move it.
    for (0..before.cols) |c| {
        try std.testing.expectEqual(@as(u32, @bitCast(before.at(0, c))), @as(u32, @bitCast(after.at(0, c))));
    }
    // Otherwise the pass above is vacuous: the edit must reach a later position.
    try std.testing.expect(after.at(1, 0) != before.at(1, 0));
}

test "attention groups query heads onto their kv head" {
    const cfg = attention.Config{ .n_heads = 4, .n_kv_heads = 2, .head_dim = 2 };
    // Heads 0 and 1 are the same query vector, heads 2 and 3 the other one.
    var q = try filled(std.testing.allocator, 2, 8, &.{ 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0 });
    defer q.deinit();
    var k = try filled(std.testing.allocator, 2, 4, &.{ 1, 0, 2, 0, 0, 1, 0, 2 });
    defer k.deinit();
    var v = try filled(std.testing.allocator, 2, 4, &.{ 1, 2, 3, 4, 5, 6, 7, 8 });
    defer v.deinit();

    var out = try attention.forward(std.testing.allocator, q, k, v, cfg);
    defer out.deinit();

    // Position 0 sees one key, so it returns v[0] of that group's own kv head.
    try std.testing.expectEqualSlices(f32, &.{ 1, 2 }, out.rowConst(0)[0..2]);
    try std.testing.expectEqualSlices(f32, &.{ 3, 4 }, out.rowConst(0)[4..6]);

    for (0..2) |t| {
        try std.testing.expectEqualSlices(f32, out.rowConst(t)[0..2], out.rowConst(t)[2..4]);
        try std.testing.expect(out.at(t, 0) != out.at(t, 4));
    }
}

test "attention softmax weights sum to one per query position" {
    // v is all ones, so every output element is the weight sum of that position.
    var q = try filled(std.testing.allocator, 4, 2, &.{ 1, 0, 0, 1, 1, 1, 0, -1 });
    defer q.deinit();
    var k = try filled(std.testing.allocator, 4, 2, &.{ 1, 1, 0, 1, -1, 0, 2, 1 });
    defer k.deinit();
    var v = try filled(std.testing.allocator, 4, 2, &.{ 1, 1, 1, 1, 1, 1, 1, 1 });
    defer v.deinit();

    var out = try attention.forward(std.testing.allocator, q, k, v, .{ .n_heads = 1, .n_kv_heads = 1, .head_dim = 2 });
    defer out.deinit();

    // An unnormalised softmax would report 1, 2, 3 and 4 here.
    for (out.data) |got| try std.testing.expectApproxEqAbs(@as(f32, 1), got, 1e-5);
}

test "attention averages exactly the causal prefix" {
    // Every dot is 0, so position t weights rows 0..t equally.
    var q = try filled(std.testing.allocator, 4, 2, &.{ 1, 0, 1, 0, 1, 0, 1, 0 });
    defer q.deinit();
    var k = try filled(std.testing.allocator, 4, 2, &.{ 0, 0, 0, 0, 0, 0, 0, 0 });
    defer k.deinit();
    var v = try filled(std.testing.allocator, 4, 2, &.{ 1, 2, 3, 4, 5, 6, 7, 8 });
    defer v.deinit();

    var out = try attention.forward(std.testing.allocator, q, k, v, .{ .n_heads = 1, .n_kv_heads = 1, .head_dim = 2 });
    defer out.deinit();

    // means of [1 2] [1 3] [1 5] [1 7] and of [2 4] [2 6] [2 8] [2 10]
    const want = [_]f32{ 1, 2, 2, 3, 3, 4, 4, 5 };
    for (want, out.data) |w, got| try std.testing.expectApproxEqAbs(w, got, 1e-6);
}

test "attention scales scores by the inverse square root of head dim" {
    // head_dim 4 makes the scale exactly 0.5, so the two scores are exactly 0
    // and 2 and the weights are 1/(1 + e^2) and e^2/(1 + e^2). Dropping the
    // scale scores them 0 and 4, which weighs row 0 at 1/(1 + e^4) =
    // 0.01798621 instead of 0.11920292.
    var q = try filled(std.testing.allocator, 2, 4, &.{ 1, 0, 0, 0, 0, 1, 0, 0 });
    defer q.deinit();
    var k = try filled(std.testing.allocator, 2, 4, &.{ 2, 0, 0, 0, 0, 4, 0, 0 });
    defer k.deinit();
    var v = try filled(std.testing.allocator, 2, 4, &.{ 1, 0, 0, 0, 0, 1, 0, 0 });
    defer v.deinit();

    var out = try attention.forward(std.testing.allocator, q, k, v, .{ .n_heads = 1, .n_kv_heads = 1, .head_dim = 4 });
    defer out.deinit();

    const want = [_]f32{ 1, 0, 0, 0, 0.11920292202211755, 0.88079707797788245, 0, 0 };
    for (want, out.data) |w, got| try std.testing.expectApproxEqAbs(w, got, 1e-6);
}

test "attention survives scores that would overflow exp" {
    // 100 * 100 / sqrt(2) = 7071, and exp(7071) is inf. Exponentiating before
    // subtracting the row max gives inf / inf = NaN here.
    var q = try filled(std.testing.allocator, 2, 2, &.{ 100, 0, 100, 0 });
    defer q.deinit();
    var k = try filled(std.testing.allocator, 2, 2, &.{ 100, 0, 0, 0 });
    defer k.deinit();
    var v = try filled(std.testing.allocator, 2, 2, &.{ 3, 5, 7, 9 });
    defer v.deinit();

    var out = try attention.forward(std.testing.allocator, q, k, v, .{ .n_heads = 1, .n_kv_heads = 1, .head_dim = 2 });
    defer out.deinit();

    // The far weight underflows to 0, so both rows land on v0.
    try std.testing.expectEqualSlices(f32, &.{ 3, 5, 3, 5 }, out.data);
}

test "attention rejects a head config with no valid group split" {
    var q = try filled(std.testing.allocator, 1, 2, &.{ 1, 2 });
    defer q.deinit();
    var k = try filled(std.testing.allocator, 1, 2, &.{ 3, 4 });
    defer k.deinit();
    var v = try filled(std.testing.allocator, 1, 2, &.{ 5, 6 });
    defer v.deinit();

    // Zero kv heads divides by zero, a zero head dim makes the scale infinite,
    // and 3 query heads over 2 kv heads leaves a head with no group.
    try std.testing.expectError(error.InvalidHeadConfig, attention.forward(std.testing.allocator, q, k, v, .{ .n_heads = 1, .n_kv_heads = 0, .head_dim = 2 }));
    try std.testing.expectError(error.InvalidHeadConfig, attention.forward(std.testing.allocator, q, k, v, .{ .n_heads = 1, .n_kv_heads = 1, .head_dim = 0 }));
    try std.testing.expectError(error.InvalidHeadConfig, attention.forward(std.testing.allocator, q, k, v, .{ .n_heads = 3, .n_kv_heads = 2, .head_dim = 2 }));
}

test "attention rejects k and v with different shapes" {
    const cfg = attention.Config{ .n_heads = 1, .n_kv_heads = 1, .head_dim = 2 };
    var q = try filled(std.testing.allocator, 2, 2, &.{ 1, 0, 0, 1 });
    defer q.deinit();
    var wide = try filled(std.testing.allocator, 2, 2, &.{ 1, 0, 0, 1 });
    defer wide.deinit();
    var wide_v = try filled(std.testing.allocator, 2, 4, &.{ 1, 2, 3, 4, 5, 6, 7, 8 });
    defer wide_v.deinit();
    var one = try filled(std.testing.allocator, 1, 2, &.{ 1, 0 });
    defer one.deinit();

    try std.testing.expectError(error.DimensionMismatch, attention.forward(std.testing.allocator, q, wide, wide_v, cfg));
    try std.testing.expectError(error.DimensionMismatch, attention.forward(std.testing.allocator, q, one, wide, cfg));
    try std.testing.expectError(error.DimensionMismatch, attention.forward(std.testing.allocator, q, one, one, cfg));
}

test "attention rejects a kv width the config disagrees with" {
    var q = try filled(std.testing.allocator, 1, 2, &.{ 1, 0 });
    defer q.deinit();
    var kv = try filled(std.testing.allocator, 1, 2, &.{ 3, 4 });
    defer kv.deinit();
    var v = try filled(std.testing.allocator, 1, 2, &.{ 5, 6 });
    defer v.deinit();
    var fat_q = try filled(std.testing.allocator, 1, 3, &.{ 1, 2, 3 });
    defer fat_q.deinit();
    var fat_kv = try filled(std.testing.allocator, 1, 3, &.{ 3, 4, 5 });
    defer fat_kv.deinit();

    // Two heads of 2 dims need 4 columns of q, and one kv head needs 2 of k.
    try std.testing.expectError(error.DimensionMismatch, attention.forward(std.testing.allocator, fat_q, kv, v, .{ .n_heads = 2, .n_kv_heads = 1, .head_dim = 2 }));
    try std.testing.expectError(error.DimensionMismatch, attention.forward(std.testing.allocator, q, fat_kv, v, .{ .n_heads = 1, .n_kv_heads = 1, .head_dim = 2 }));
}

test "attention on zero tokens returns an empty output" {
    var q = try filled(std.testing.allocator, 0, 2, &.{});
    defer q.deinit();
    var k = try filled(std.testing.allocator, 0, 2, &.{});
    defer k.deinit();
    var v = try filled(std.testing.allocator, 0, 2, &.{});
    defer v.deinit();

    var out = try attention.forward(std.testing.allocator, q, k, v, .{ .n_heads = 1, .n_kv_heads = 1, .head_dim = 2 });
    defer out.deinit();

    try std.testing.expectEqual(@as(usize, 0), out.rows);
    try std.testing.expectEqual(@as(usize, 2), out.cols);
    try std.testing.expectEqual(@as(usize, 0), out.data.len);
}

test "attention keeps an f64 accumulator over a 512 token context" {
    // Every other test here is small, and its values are dyadic, so f32 and
    // f64 arithmetic agree bit for bit on them and nothing above can tell an
    // f64 accumulator from an f32 one. This is the test that can.
    //
    // One head, head_dim 8, T 512, and q, k and v filled with 1/3 and 1/7.
    // Both are inexact in f32, so every product and every partial sum is
    // rounded, and a 512 term reduction is long enough for those roundings to
    // accumulate instead of cancelling. Measured over the 4096 outputs of this
    // configuration, an f32 accumulator is up to 2.21e-6 absolute and 6.62e-6
    // relative away from the f64 answer, and the largest gap sits at t = 492.
    //
    // The tolerance below is 1e-6 relative: 6.6x under the f32 drift, so
    // narrowing the accumulator fails, and still 14x over the f32 storage
    // rounding the output genuinely carries, which is 4.6e-7 relative at 0.33.
    //
    // The reference is the same formula computed here in f64 from the same
    // inputs, written out rather than called, so narrowing attention.zig's
    // accumulator changes the code under test and not the expected answer.
    const t_len: usize = 512;
    const dim: usize = 8;
    const cfg = attention.Config{ .n_heads = 1, .n_kv_heads = 1, .head_dim = dim };

    var q = try Tensor.init(std.testing.allocator, t_len, dim);
    defer q.deinit();
    var k = try Tensor.init(std.testing.allocator, t_len, dim);
    defer k.deinit();
    var v = try Tensor.init(std.testing.allocator, t_len, dim);
    defer v.deinit();
    // Even indices take 1/3, odd take 1/7, so a dot product sums a mix of
    // 1/9 and 1/21 terms and no two consecutive positions repeat.
    for (0..t_len) |t| {
        for (0..dim) |j| {
            const val: f32 = if ((t + j) % 2 == 0) 1.0 / 3.0 else 1.0 / 7.0;
            q.set(t, j, val);
            k.set(t, j, val);
            v.set(t, j, val);
        }
    }

    var out = try attention.forward(std.testing.allocator, q, k, v, cfg);
    defer out.deinit();

    const scale = 1.0 / @sqrt(@as(f64, @floatFromInt(dim)));
    var scores: [t_len]f64 = undefined;
    for (0..t_len) |t| {
        var row_max: f64 = -std.math.inf(f64);
        for (0..t + 1) |s| {
            var dot: f64 = 0;
            for (0..dim) |j| dot += @as(f64, q.at(t, j)) * @as(f64, k.at(s, j));
            scores[s] = dot * scale;
            row_max = @max(row_max, scores[s]);
        }
        var denom: f64 = 0;
        for (0..t + 1) |s| {
            scores[s] = @exp(scores[s] - row_max);
            denom += scores[s];
        }
        for (0..t + 1) |s| scores[s] /= denom;
        for (0..dim) |j| {
            var acc: f64 = 0;
            for (0..t + 1) |s| acc += scores[s] * @as(f64, v.at(s, j));
            try std.testing.expectApproxEqRel(acc, @as(f64, out.at(t, j)), 1e-6);
        }
    }
}

test "attention is bit reproducible run to run" {
    const cfg = attention.Config{ .n_heads = 4, .n_kv_heads = 2, .head_dim = 4 };
    var q = try filled(std.testing.allocator, 3, 16, &.{
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
        1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2,
        3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8,
    });
    defer q.deinit();
    var k = try filled(std.testing.allocator, 3, 8, &.{ 0.5, 0.25, 0.125, 0.0625, 0.5, 0.25, 0.125, 0.0625, 1, 0.5, 0.25, 0.125, 1, 0.5, 0.25, 0.125, 2, 1, 0.5, 0.25, 2, 1, 0.5, 0.25 });
    defer k.deinit();
    var v = try filled(std.testing.allocator, 3, 8, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 });
    defer v.deinit();

    var first = try attention.forward(std.testing.allocator, q, k, v, cfg);
    defer first.deinit();
    var second = try attention.forward(std.testing.allocator, q, k, v, cfg);
    defer second.deinit();

    for (first.data, second.data) |a, b| {
        try std.testing.expectEqual(@as(u32, @bitCast(a)), @as(u32, @bitCast(b)));
    }
}
