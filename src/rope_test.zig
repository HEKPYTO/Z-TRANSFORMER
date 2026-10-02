const std = @import("std");
const model = @import("model.zig");

// Llama-3's RoPE base is 500000, not the textbook 10000, and NOTHING in the Zig
// suite would otherwise notice a change to it. The exporter writes
// `model.rope_theta` into the shape it hands the oracle and `removed_test.zig`
// once compared that field against `model.rope_theta` -- a value against itself, which
// passes for any value at all, and a silent edit to 10000 left `zig build verify`
// green, `zig build test` green, and CI green. It now reads the field back out of the
// exported `config.txt` and compares THAT against the constant, so the drift is caught
// in CI rather than by a manual parity run. What no longer holds is the older claim
// that the only thing which notices is
// the Python parity run, which no build gate invokes and which is a manual
// command on one host. An audit found this by asking what the parity harness
// depends on that nothing checks.
//
// The assertion is here rather than in `removed_test.zig` because this is the file
// that owns RoPE, and because a failure should name the constant rather than a
// field of an exported struct.
test "llama3 rope base is 500000, not the textbook 10000" {
    try std.testing.expectEqual(@as(f64, 500000), model.rope_theta);
}
const rope = @import("rope.zig");
const tensor = @import("tensor.zig");

const theta: f64 = 500000;

fn fromLiterals(a: std.mem.Allocator, rows: usize, cols: usize, vals: []const f32) !tensor.Tensor {
    var t = try tensor.Tensor.init(a, rows, cols);
    errdefer t.deinit();
    @memcpy(t.data, vals);
    return t;
}

/// The plain L2 norm of `t`, as `@sqrt(sum of squares)`.
///
/// This stays a direct sum. It cannot be `norm.forward` with an all ones
/// weight, because that returns x / rms(x) per element, which is not the norm
/// of anything: the reciprocal is still in it. Measured on the row this file
/// checks, x = [1 -2 3 -4 5 -6], the exact norm is 9.5393920142 and
/// x[0] / rms(x) is 0.2567762109, a factor of 37.15 apart, and the ratio is
/// not a constant of the row length either because of the eps inside the rms.
/// The literals asserted below still hold: 9.5393920142 is the hand computed
/// sqrt(91) and both calls to this helper return it within 1e-5.
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

    var y = try rope.forward(std.testing.allocator, x, 0, theta, 8);
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

    var y = try rope.forward(std.testing.allocator, x, 1, theta, 2);
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

    var y = try rope.forward(std.testing.allocator, x, 1, theta, 4);
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

    var y = try rope.forward(std.testing.allocator, x, 1, theta, 4);
    defer y.deinit();

    try std.testing.expectApproxEqAbs(@as(f32, 0.5403023058681397), y.at(0, 0), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0), y.at(0, 1), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.8414709848078965), y.at(0, 2), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0), y.at(0, 3), 1e-6);
}

test "each head block rotates on its own, never the whole concatenated row" {
    // head_dim 4 across 8 columns is two heads, which is the shape the model
    // hands over: q is [T, n_heads * head_dim] and k is
    // [T, n_kv_heads * head_dim], so x.cols is never the head dim.
    //
    // Per block of 4 the half is 2, so the two frequencies are
    // 500000^-(0/4) = 1 and 500000^-(2/4) = 0.001414213562373095. At pos 1
    // the angles are therefore 1 and 0.001414213562373095 radians, with
    //     cos 1   = 0.5403023058681398   sin 1   = 0.8414709848078965
    //     cos     = 0.9999990000001666   sin     = 0.0014142130909686214
    // Head 0 holds x[0..4] and pairs (1, 3) at the first angle and (2, 4) at
    // the second:
    //     1 cos - 3 sin =  0.5403023058681398 - 2.5244129544236895 = -1.9841106485555497
    //     3 cos + 1 sin =  1.6209069176044194 + 0.8414709848078965 =  2.4623779024123159
    //     2 cos - 4 sin =  1.9999980000003332 - 0.0056568523638745 =  1.9943411476364587
    //     4 cos + 2 sin =  3.9999960000006664 + 0.0028284261819372 =  4.0028244261826036
    // Head 1 holds x[4..8] and pairs (5, 7) and (6, 8) on the same two angles:
    //     5 cos - 7 sin =  2.7015115293406990 - 5.8902968936552755 = -3.1887853643145765
    //     7 cos + 5 sin =  3.7821161410769786 + 4.2073549240394825 =  7.9894710651164611
    //     6 cos - 8 sin =  5.9999940000009996 - 0.0113137047277490 =  5.9886802952732506
    //     8 cos + 6 sin =  7.9999920000013328 + 0.0084852785458117 =  8.0084772785471445
    //
    // Taking x.cols as one head of dim 8 instead pairs (0,4) (1,5) (2,6) (3,7)
    // and divides every exponent by 8 rather than 4, so all eight literals are
    // wrong there. The test is a unit mass moved across the head boundary on
    // top of that, so the second half of it is a separate case.
    var x = try fromLiterals(std.testing.allocator, 1, 8, &.{ 1, 2, 3, 4, 5, 6, 7, 8 });
    defer x.deinit();

    var y = try rope.forward(std.testing.allocator, x, 1, theta, 4);
    defer y.deinit();

    const want = [8]f32{
        -1.9841106485555497, 1.9943411476364587, 2.4623779024123159, 4.0028244261826036,
        -3.1887853643145765, 5.9886802952732506, 7.9894710651164611, 8.0084772785471445,
    };
    for (want, 0..) |v, i| try std.testing.expectApproxEqAbs(v, y.data[i], 1e-6);
}

test "rotation preserves the euclidean norm" {
    // d=6 so half=3: a partner offset of 2 or 4 moves the mass between pairs and
    // breaks the norm, which an even power of two head dim can hide.
    // ||x|| = sqrt(1+4+9+16+25+36) = sqrt(91) = 9.5393920142
    var x = try fromLiterals(std.testing.allocator, 1, 6, &.{ 1, -2, 3, -4, 5, -6 });
    defer x.deinit();

    var y = try rope.forward(std.testing.allocator, x, 3, theta, 6);
    defer y.deinit();

    try std.testing.expectApproxEqAbs(9.5393920142, l2(x), 1e-5);
    try std.testing.expectApproxEqAbs(9.5393920142, l2(y), 1e-5);
    try std.testing.expectApproxEqAbs(l2(x), l2(y), 1e-5);
}

test "an odd head dim is an error, never a silent truncate" {
    var x = try fromLiterals(std.testing.allocator, 1, 3, &.{ 1, 2, 3 });
    defer x.deinit();

    try std.testing.expectError(error.OddHeadDim, rope.forward(std.testing.allocator, x, 0, theta, 3));
    // Zero has no pair to rotate either, and `half` would be a loop that never
    // runs, which would report the row back unchanged as a silent success.
    try std.testing.expectError(error.OddHeadDim, rope.forward(std.testing.allocator, x, 0, theta, 0));
}

test "a row that is not a whole number of head blocks is a mismatch" {
    // 6 columns over head_dim 4 leaves a two wide remainder with no partner,
    // so reporting it is the only honest answer. Reading it as a head of dim 6
    // instead would rotate across a head boundary that does not exist.
    var x = try fromLiterals(std.testing.allocator, 1, 6, &.{ 1, 2, 3, 4, 5, 6 });
    defer x.deinit();

    try std.testing.expectError(error.DimensionMismatch, rope.forward(std.testing.allocator, x, 0, theta, 4));
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

    var y = try rope.forward(std.testing.allocator, x, 2, theta, 4);
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
