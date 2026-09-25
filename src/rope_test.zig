const std = @import("std");
const rope = @import("rope.zig");
const tensor = @import("tensor.zig");

const theta: f64 = 500000;

fn fromLiterals(a: std.mem.Allocator, rows: usize, cols: usize, vals: []const f32) !tensor.Tensor {
    var t = try tensor.Tensor.init(a, rows, cols);
    errdefer t.deinit();
    @memcpy(t.data, vals);
    return t;
}

fn l2(t: tensor.Tensor) f64 {
    var acc: f64 = 0;
    for (t.data) |v| {
        const w: f64 = @floatCast(v);
        acc += w * w;
    }
    return @sqrt(acc);
}

test "position 0 is the identity, bit for bit" {
    // One row, so every element really does sit at absolute position 0 and not
    // at pos+r. cos(0) is exactly 1 and sin(0) is exactly 0, so every rotation
    // degenerates to x*1 - y*0. -0.0 is kept out of the input because y*1 + x*0
    // turns -0.0 into +0.0, a real bit change even though it is the same number.
    var x = try fromLiterals(std.testing.allocator, 1, 8, &.{ 1, -2, 0.5, -0.25, 3, -4, 0.125, 7 });
    defer x.deinit();

    var y = try rope.forward(std.testing.allocator, x, 0, theta);
    defer y.deinit();

    for (x.data, y.data) |a, b| {
        const ab: u32 = @bitCast(a);
        const bb: u32 = @bitCast(b);
        try std.testing.expectEqual(ab, bb);
    }
}

test "d=2 pos=1 rotates the single pair by one radian" {
    // d=2 so half=1 and freq_0 = theta^0 = 1, hence angle = 1 rad exactly.
    // cos 1 = 0.5403023058681397, sin 1 = 0.8414709848078965.
    // 3 cos 1 - 4 sin 1 = 1.6209069176044192 - 3.3658839392315860
    // 4 cos 1 + 3 sin 1 = 2.1612092234725589 + 2.5244129544236895
    var x = try fromLiterals(std.testing.allocator, 1, 2, &.{ 3, 4 });
    defer x.deinit();

    var y = try rope.forward(std.testing.allocator, x, 1, theta);
    defer y.deinit();

    try std.testing.expectApproxEqAbs(@as(f32, -1.7449770216271669), y.at(0, 0), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 4.6856221778962484), y.at(0, 1), 1e-6);
}

test "d=4 pos=1 turns both halves of the split" {
    // d=4 so half=2. freq_0 = 1, freq_1 = 500000^(-2/4) = 0.0014142135623730951.
    // Pair 0 is (x[0], x[2]) = (1, 3) at 1 rad:
    //   1 cos - 3 sin = 0.5403023058681397 - 2.5244129544236895 = -1.9841106485555498
    //   3 cos + 1 sin = 1.6209069176044192 + 0.8414709848078965 = 2.4623779024123157
    // Pair 1 is (x[1], x[3]) = (2, 4) at 0.0014142135623730951 rad, where
    // cos = 0.9999990000000002 and sin = 0.0014142130909685744:
    //   2 cos - 4 sin = 1.9999980000000004 - 0.0056568523638743 = 1.9943411476361261
    //   4 cos + 2 sin = 3.9999960000000008 + 0.0028284261819371 = 4.0028244261821879
    var x = try fromLiterals(std.testing.allocator, 1, 4, &.{ 1, 2, 3, 4 });
    defer x.deinit();

    var y = try rope.forward(std.testing.allocator, x, 1, theta);
    defer y.deinit();

    try std.testing.expectApproxEqAbs(@as(f32, -1.9841106485555498), y.at(0, 0), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.9943411476361261), y.at(0, 1), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.4623779024123157), y.at(0, 2), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0028244261821879), y.at(0, 3), 1e-6);
}

test "llama-3 half-split layout, not the interleaved pairs of the original rope" {
    // A lone unit in x[0] is the whole discriminator. Half-split pairs x[0]
    // with x[d/2] = x[2], so all the mass lands in y[2] and y[1] stays exactly 0.
    // Interleaved layout pairs x[0] with x[1] and answers
    // [0.5403023058681397, 0.8414709848078965, 0, 0] instead, failing both
    // assertions below.
    var x = try fromLiterals(std.testing.allocator, 1, 4, &.{ 1, 0, 0, 0 });
    defer x.deinit();

    var y = try rope.forward(std.testing.allocator, x, 1, theta);
    defer y.deinit();

    try std.testing.expectApproxEqAbs(@as(f32, 0.5403023058681397), y.at(0, 0), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0), y.at(0, 1), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.8414709848078965), y.at(0, 2), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0), y.at(0, 3), 1e-6);
}

test "rotation preserves the euclidean norm" {
    // d=6 so half=3: a partner offset of 2 or 4 moves the mass between pairs and
    // breaks the norm, which an even power of two head dim can hide.
    // ||x|| = sqrt(1+4+9+16+25+36) = sqrt(91) = 9.5393920142
    var x = try fromLiterals(std.testing.allocator, 1, 6, &.{ 1, -2, 3, -4, 5, -6 });
    defer x.deinit();

    var y = try rope.forward(std.testing.allocator, x, 3, theta);
    defer y.deinit();

    try std.testing.expectApproxEqAbs(9.5393920142, l2(x), 1e-5);
    try std.testing.expectApproxEqAbs(9.5393920142, l2(y), 1e-5);
    try std.testing.expectApproxEqAbs(l2(x), l2(y), 1e-5);
}

test "an odd head dim is an error, never a silent truncate" {
    var x = try fromLiterals(std.testing.allocator, 1, 3, &.{ 1, 2, 3 });
    defer x.deinit();

    try std.testing.expectError(error.OddHeadDim, rope.forward(std.testing.allocator, x, 0, theta));
}

test "row r rotates at absolute position pos plus r" {
    // Three rows at pos=2 turn at positions 2, 3 and 4. Pair 0 uses
    // freq_0 = 1 so the angles are 2, 3 and 4 rad; pair 1 uses
    // freq_1 = 0.0014142135623730951, giving 0.0028284271247461902,
    // 0.0042426406871192853 and 0.0056568542494923804 rad with cos/sin of
    // 0.9999960000000027/0.0028284233535115,
    // 0.9999910000135/0.0042426279592087 and
    // 0.9999840000426667/0.0056568240796513.
    var x = try fromLiterals(std.testing.allocator, 3, 4, &.{
        1,   2,    3, 4,
        0.5, -0.5, 1, -1,
        1,   -1,   0, 0,
    });
    defer x.deinit();

    var y = try rope.forward(std.testing.allocator, x, 2, theta);
    defer y.deinit();

    try std.testing.expectApproxEqAbs(@as(f32, -3.1440391170241875), y.at(0, 0), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.9886783065859592), y.at(0, 1), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -0.3391430828157455), y.at(0, 2), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0056408467070337), y.at(0, 3), 1e-6);

    try std.testing.expectApproxEqAbs(@as(f32, -0.6361162563600899), y.at(1, 0), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -0.4957528720475413), y.at(1, 1), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -0.9194324925705119), y.at(1, 2), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0021123139931044), y.at(1, 3), 1e-6);

    try std.testing.expectApproxEqAbs(@as(f32, -0.6536436208636119), y.at(2, 0), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -0.9999840000426667), y.at(2, 1), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -0.7568024953079283), y.at(2, 2), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -0.0056568240796513), y.at(2, 3), 1e-6);
}
