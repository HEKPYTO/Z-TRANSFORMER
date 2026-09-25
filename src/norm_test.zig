const std = @import("std");
const norm = @import("norm.zig");
const Tensor = @import("tensor.zig").Tensor;

/// Tolerance for every literal below. Phase 1 exit criterion is 1e-6 against a
/// hand-computed fp64 answer.
const tol: f32 = 1e-6;

test "norm forward matches a hand computed 1x4 row" {
    // x = [1 2 3 4], w = [0.5 1 2 0.25]
    // sum sq = 1+4+9+16 = 30, mean = 30/4 = 7.5, rms = sqrt(7.50001) = 2.7386146
    // y = x/rms*w = [0.18257406, 0.73029626, 2.19088877, 0.36514813]
    var x = try Tensor.init(std.testing.allocator, 1, 4);
    defer x.deinit();
    var w = try Tensor.init(std.testing.allocator, 1, 4);
    defer w.deinit();
    for ([_]f32{ 1, 2, 3, 4 }, 0..) |v, i| x.set(0, i, v);
    for ([_]f32{ 0.5, 1, 2, 0.25 }, 0..) |v, i| w.set(0, i, v);

    var got = try norm.forward(std.testing.allocator, x, w);
    defer got.deinit();

    try std.testing.expectEqual(@as(usize, 1), got.rows);
    try std.testing.expectEqual(@as(usize, 4), got.cols);
    try std.testing.expectApproxEqAbs(@as(f32, 0.18257406), got.at(0, 0), tol);
    try std.testing.expectApproxEqAbs(@as(f32, 0.73029626), got.at(0, 1), tol);
    try std.testing.expectApproxEqAbs(@as(f32, 2.19088877), got.at(0, 2), tol);
    try std.testing.expectApproxEqAbs(@as(f32, 0.36514813), got.at(0, 3), tol);
}

test "norm forward reduces over the last axis, per row" {
    // x = [ 1  2  3]
    //     [10 20 30],  w = [0.5 1 2]
    //
    // Per row: row 0 sum sq = 14, mean = 14/3, rms = 2.1602492;
    //          row 1 sum sq = 1400, mean = 1400/3, rms = 21.6024692.
    // A single global rms over all 6 elements is sqrt(1414/6) = 15.3514389,
    // which would give [0.0325702, 0.1302809, 0.3908428] and
    // [0.3257024, 1.3028095, 3.9084284]. Those are the wrong answers.
    var x = try Tensor.init(std.testing.allocator, 2, 3);
    defer x.deinit();
    var w = try Tensor.init(std.testing.allocator, 1, 3);
    defer w.deinit();
    for ([_]f32{ 1, 2, 3, 10, 20, 30 }, 0..) |v, i| x.set(i / 3, i % 3, v);
    for ([_]f32{ 0.5, 1, 2 }, 0..) |v, i| w.set(0, i, v);

    var got = try norm.forward(std.testing.allocator, x, w);
    defer got.deinit();

    try std.testing.expectApproxEqAbs(@as(f32, 0.23145478), got.at(0, 0), tol);
    try std.testing.expectApproxEqAbs(@as(f32, 0.92581911), got.at(0, 1), tol);
    try std.testing.expectApproxEqAbs(@as(f32, 2.77745732), got.at(0, 2), tol);
    try std.testing.expectApproxEqAbs(@as(f32, 0.23145502), got.at(1, 0), tol);
    try std.testing.expectApproxEqAbs(@as(f32, 0.92582009), got.at(1, 1), tol);
    try std.testing.expectApproxEqAbs(@as(f32, 2.77746027), got.at(1, 2), tol);
}

test "norm forward with an all ones weight is pure normalization" {
    // x = [3 4], sum sq = 25, mean = 12.5, rms = sqrt(12.50001) = 3.5355353
    // y = [3/rms, 4/rms] = [0.84852780, 1.13137040]
    var x = try Tensor.init(std.testing.allocator, 1, 2);
    defer x.deinit();
    var w = try Tensor.init(std.testing.allocator, 1, 2);
    defer w.deinit();
    x.set(0, 0, 3);
    x.set(0, 1, 4);
    w.fill(1);

    var got = try norm.forward(std.testing.allocator, x, w);
    defer got.deinit();

    try std.testing.expectApproxEqAbs(@as(f32, 0.8485278), got.at(0, 0), tol);
    try std.testing.expectApproxEqAbs(@as(f32, 1.1313704), got.at(0, 1), tol);
}

test "norm forward on an all zero row returns zeros, not NaN" {
    // rms = sqrt(0 + 1e-5) = 0.0031623, so 0/rms is 0 and the row stays finite.
    var x = try Tensor.init(std.testing.allocator, 2, 3);
    defer x.deinit();
    var w = try Tensor.init(std.testing.allocator, 1, 3);
    defer w.deinit();
    for ([_]f32{ 1, 2, 3 }, 0..) |v, i| w.set(0, i, v);

    var got = try norm.forward(std.testing.allocator, x, w);
    defer got.deinit();

    try std.testing.expectEqual(@as(usize, 6), got.data.len);
    for (got.data) |v| {
        try std.testing.expect(std.math.isFinite(v));
        try std.testing.expectEqual(@as(f32, 0), v);
    }
}

test "norm forward keeps an f64 accumulator over a 512 element row" {
    // 512 is ffn_dim of the default model config, so this is the widest row the
    // model actually builds. 4096, which this used to use, is a shape nothing
    // here ever produces, and a tolerance picked for it says nothing about the
    // rows that run.
    //
    // 512 copies of 1/3. Measured on this row, in f32 and in f64:
    //     sum of squares   f32 56.888706        f64 56.88889227973056
    //     mean             f32 0.111110754      f64 0.11111111773384875
    //     rms              f32 0.3333478        f64 0.33334833
    //     y = (1/3)/rms    f32 0.99995667       f64 0.99995506
    //     exact, f64 throughout, 0.999955003039954
    // So the f64 accumulator lands 5.7e-8 from the exact value, which is the
    // f32 storage rounding of the one division, and the f32 accumulator lands
    // 1.667e-6 from it. The tolerance below is 1e-6: 17x the error norm.zig
    // actually has, and still 1.6x tighter than the f32 drift, so an
    // accumulator narrowed to f32 fails here. The last assertion is that same
    // claim stated as a check, so this test cannot quietly stop discriminating.
    const d: usize = 512;
    const third: f32 = 1.0 / 3.0;
    const exact: f64 = 0.999955003039954;

    var sum32: f32 = 0;
    for (0..d) |_| sum32 += third * third;
    const mean32: f64 = @as(f64, sum32) / @as(f64, @floatFromInt(d));
    const rms32: f32 = @floatCast(@sqrt(mean32 + 1e-5));
    const y32: f64 = @as(f64, third) / @as(f64, rms32);

    var x = try Tensor.init(std.testing.allocator, 1, d);
    defer x.deinit();
    var w = try Tensor.init(std.testing.allocator, 1, d);
    defer w.deinit();
    x.fill(third);
    w.fill(1);

    var got = try norm.forward(std.testing.allocator, x, w);
    defer got.deinit();

    for (got.data) |v| try std.testing.expectApproxEqAbs(@as(f32, 0.999955), v, tol);
    // Stated in f64 against the exact value, so the margin is the measured one
    // and not the f32 rounding of the literal on the line above.
    try std.testing.expectApproxEqAbs(exact, @as(f64, got.at(0, 0)), @as(f64, tol));
    // An f32 accumulator misses by more than the tolerance, which is what makes
    // the loop above a test of the accumulator rather than of the formula.
    try std.testing.expect(@abs(y32 - exact) > @as(f64, tol));
}

test "norm forward rejects a weight of the wrong length" {
    // d = 3, so a weight must hold exactly 3 elements. A [2 2] weight holds 4.
    var x = try Tensor.init(std.testing.allocator, 2, 3);
    defer x.deinit();
    var w = try Tensor.init(std.testing.allocator, 2, 2);
    defer w.deinit();
    w.fill(1);

    try std.testing.expectError(error.DimensionMismatch, norm.forward(std.testing.allocator, x, w));
}

test "norm forward accepts a column weight of shape d by 1" {
    // Same 1x4 row as the hand computed case, weight stored as [d 1].
    var x = try Tensor.init(std.testing.allocator, 1, 4);
    defer x.deinit();
    var w = try Tensor.init(std.testing.allocator, 4, 1);
    defer w.deinit();
    for ([_]f32{ 1, 2, 3, 4 }, 0..) |v, i| x.set(0, i, v);
    for ([_]f32{ 0.5, 1, 2, 0.25 }, 0..) |v, i| w.set(i, 0, v);

    var got = try norm.forward(std.testing.allocator, x, w);
    defer got.deinit();

    try std.testing.expectApproxEqAbs(@as(f32, 0.18257406), got.at(0, 0), tol);
    try std.testing.expectApproxEqAbs(@as(f32, 0.73029626), got.at(0, 1), tol);
    try std.testing.expectApproxEqAbs(@as(f32, 2.19088877), got.at(0, 2), tol);
    try std.testing.expectApproxEqAbs(@as(f32, 0.36514813), got.at(0, 3), tol);
}
