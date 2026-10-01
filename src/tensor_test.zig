const std = @import("std");
const tensor = @import("tensor.zig");

test "tensor init allocates rows times cols zeroed" {
    var t = try tensor.Tensor.init(std.testing.allocator, 3, 4);
    defer t.deinit();

    try std.testing.expectEqual(@as(usize, 12), t.data.len);
    try std.testing.expectEqual(@as(usize, 3), t.rows);
    try std.testing.expectEqual(@as(usize, 4), t.cols);
    for (t.data) |v| try std.testing.expectEqual(@as(f32, 0), v);
}

test "tensor init rejects a shape whose element count overflows" {
    // 2^40 * 2^40 is 2^80, which does not fit a usize, and the shapes are run
    // values rather than literals, so nothing at comptime stops the call. Left
    // unchecked the product wraps to 0 in a release build and `init` hands back
    // an empty buffer that still reports rows = cols = 2^40, so every later
    // bounds guard passes and every write lands past the end of the allocation.
    try std.testing.expectError(
        error.DimensionOverflow,
        tensor.Tensor.init(std.testing.allocator, 1 << 40, 1 << 40),
    );
    // The largest shape that does fit still allocates, so the check is on the
    // overflow and not on the size.
    var ok = try tensor.Tensor.init(std.testing.allocator, 0, 1 << 40);
    defer ok.deinit();
    try std.testing.expectEqual(@as(usize, 0), ok.data.len);
}

test "tensor init accepts a zero sized shape" {
    var t = try tensor.Tensor.init(std.testing.allocator, 0, 0);
    defer t.deinit();

    try std.testing.expectEqual(@as(usize, 0), t.data.len);
}

test "tensor at and set round trip every corner" {
    var t = try tensor.Tensor.init(std.testing.allocator, 3, 4);
    defer t.deinit();

    t.set(0, 0, 1);
    t.set(0, 3, 2);
    t.set(2, 0, 3);
    t.set(2, 3, 4);
    t.set(1, 1, 5);
    t.set(1, 2, 6);

    try std.testing.expectEqual(@as(f32, 1), t.at(0, 0));
    try std.testing.expectEqual(@as(f32, 2), t.at(0, 3));
    try std.testing.expectEqual(@as(f32, 3), t.at(2, 0));
    try std.testing.expectEqual(@as(f32, 4), t.at(2, 3));
    try std.testing.expectEqual(@as(f32, 5), t.at(1, 1));
    try std.testing.expectEqual(@as(f32, 6), t.at(1, 2));
}

test "tensor at reads row major, not column major" {
    var t = try tensor.Tensor.init(std.testing.allocator, 2, 3);
    defer t.deinit();

    t.set(1, 0, 7);
    try std.testing.expectEqual(@as(f32, 7), t.at(1, 0));
    try std.testing.expectEqual(@as(f32, 0), t.at(0, 0));
    try std.testing.expectEqual(@as(f32, 7), t.data[3]);
}

test "tensor row and rowConst expose exactly cols elements" {
    var t = try tensor.Tensor.init(std.testing.allocator, 2, 3);
    defer t.deinit();

    try std.testing.expectEqual(@as(usize, 3), t.row(0).len);
    try std.testing.expectEqual(@as(usize, 3), t.row(1).len);
    try std.testing.expectEqual(@as(usize, 3), t.rowConst(0).len);
    try std.testing.expectEqual(@as(usize, 3), t.rowConst(1).len);
}

test "tensor row write is visible through at" {
    var t = try tensor.Tensor.init(std.testing.allocator, 2, 3);
    defer t.deinit();

    t.row(1)[2] = 42;

    try std.testing.expectEqual(@as(f32, 42), t.at(1, 2));
    try std.testing.expectEqual(@as(f32, 42), t.rowConst(1)[2]);
    try std.testing.expectEqual(@as(f32, 0), t.at(0, 2));
}

test "tensor fill writes every element" {
    var t = try tensor.Tensor.init(std.testing.allocator, 3, 5);
    defer t.deinit();

    t.fill(-1.5);

    for (t.data) |v| try std.testing.expectEqual(@as(f32, -1.5), v);
}

test "tensor matmul 1x1" {
    var a = try tensor.Tensor.init(std.testing.allocator, 1, 1);
    defer a.deinit();
    var b = try tensor.Tensor.init(std.testing.allocator, 1, 1);
    defer b.deinit();
    a.set(0, 0, 3);
    b.set(0, 0, 4);

    var got = try tensor.matmul(a, b);
    defer got.deinit();

    try std.testing.expectEqual(@as(usize, 1), got.rows);
    try std.testing.expectEqual(@as(usize, 1), got.cols);
    try std.testing.expectEqual(@as(f32, 12), got.at(0, 0));
}

test "tensor matmul 2x3 by 3x2 against hand computed values" {
    // [1 2 3]   [ 7  8]   [ 58  64]
    // [4 5 6] @ [ 9 10] = [139 154]
    //          [11 12]
    var a = try tensor.Tensor.init(std.testing.allocator, 2, 3);
    defer a.deinit();
    var b = try tensor.Tensor.init(std.testing.allocator, 3, 2);
    defer b.deinit();
    const a_vals = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const b_vals = [_]f32{ 7, 8, 9, 10, 11, 12 };
    for (a_vals, 0..) |v, i| a.set(i / 3, i % 3, v);
    for (b_vals, 0..) |v, i| b.set(i / 2, i % 2, v);

    var got = try tensor.matmul(a, b);
    defer got.deinit();

    try std.testing.expectEqual(@as(usize, 2), got.rows);
    try std.testing.expectEqual(@as(usize, 2), got.cols);
    try std.testing.expectEqual(@as(f32, 58), got.at(0, 0));
    try std.testing.expectEqual(@as(f32, 64), got.at(0, 1));
    try std.testing.expectEqual(@as(f32, 139), got.at(1, 0));
    try std.testing.expectEqual(@as(f32, 154), got.at(1, 1));
}

test "tensor matmul 3x2 by 2x2 writes every output element" {
    // [1 2]   [2 0]   [ 2  6]
    // [3 4] @ [0 3] = [ 6 12]
    // [5 6]             [10 18]
    var a = try tensor.Tensor.init(std.testing.allocator, 3, 2);
    defer a.deinit();
    var b = try tensor.Tensor.init(std.testing.allocator, 2, 2);
    defer b.deinit();
    const a_vals = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const b_vals = [_]f32{ 2, 0, 0, 3 };
    for (a_vals, 0..) |v, i| a.set(i / 2, i % 2, v);
    for (b_vals, 0..) |v, i| b.set(i / 2, i % 2, v);

    var got = try tensor.matmul(a, b);
    defer got.deinit();

    try std.testing.expectEqual(@as(usize, 3), got.rows);
    try std.testing.expectEqual(@as(usize, 2), got.cols);
    try std.testing.expectEqualSlices(f32, &.{ 2, 6, 6, 12, 10, 18 }, got.data);
}

test "tensor matmul on a zero sized dimension returns zeros" {
    var a = try tensor.Tensor.init(std.testing.allocator, 2, 0);
    defer a.deinit();
    var b = try tensor.Tensor.init(std.testing.allocator, 0, 3);
    defer b.deinit();
    b.fill(2);

    var got = try tensor.matmul(a, b);
    defer got.deinit();

    try std.testing.expectEqual(@as(usize, 6), got.data.len);
    for (got.data) |v| try std.testing.expectEqual(@as(f32, 0), v);
}

test "tensor matmul on empty rows returns an empty result" {
    var a = try tensor.Tensor.init(std.testing.allocator, 0, 3);
    defer a.deinit();
    var b = try tensor.Tensor.init(std.testing.allocator, 3, 2);
    defer b.deinit();
    b.fill(2);

    var got = try tensor.matmul(a, b);
    defer got.deinit();

    try std.testing.expectEqual(@as(usize, 0), got.data.len);
}

test "tensor matmul reduces in f32, not f64" {
    // The accumulator's width is a choice tensor.zig makes on the line
    // `out_row[j] += scale * b_row[j]`, and nothing in the suite read it back:
    // the five matmul tests above reduce k <= 3 terms, where an f32 and an f64
    // reduction agree to the last bit, so `tools/mutation`'s `matmul-f64-acc`
    // survives all of them. This is the test that reads it back, and it reads a
    // shape the model actually multiplies: [ctx, ffn] @ [ffn, d_model] is
    // `w_down` at the shipped config, 256 rows of a 512-long reduction into 128
    // columns. k = 512 is the widest reduction the shipped model has, and the
    // error of a reduction grows with it, so this is the shape where the two
    // accumulator widths are furthest apart.
    //
    // The two gates are the same formula with the machine epsilon of the
    // accumulator swapped in, and nothing is picked. A reduction of k products
    // in a type of precision `e` is off by at most (k - 1) * e relative to the
    // sum of the absolute products it adds up, so:
    //
    //     f64_reduction_max = 511 * 2.22e-16 = 1.14e-13
    //     f32_reduction_max = 511 * 1.19e-7  = 6.09e-5
    //
    // The measured divergence of this implementation from the exact reduction is
    // 1.9e-7 over six seeds, so it sits 1.7 million times above the f64 gate and
    // at 0.3% of the f32 one. Six orders of clear air on the side that has to
    // fail the mutant and two on the side that has to pass, which is what makes
    // the test a statement about the accumulator rather than a fingerprint of
    // one build: the six numbers are byte-identical in Debug, ReleaseSafe and
    // ReleaseFast, because f32 multiply and add are correctly rounded and
    // nothing here goes near a libm.
    //
    // The reference is narrowed to f32 before it is compared, because that is
    // what `matmul` returns. Both spellings store f32, so comparing an f32
    // against an unrounded f64 sum leaves the f32 store rounding on one side
    // only, and that term alone is 9.3e-9 here, which is 8.2e4 times the 1.14e-13
    // f64 gate -- nearly five orders, and a test that cannot tell an f64
    // accumulator from an f32 one.
    //
    // The normalisation is the sum of the absolute products rather than the
    // result, because the result of a cancelling dot product is near zero and a
    // relative error against near zero is a large number that means nothing.
    const m: usize = 256; // the shipped ctx, so one batch of windows
    const k: usize = 512; // ffn_dim, the widest reduction the model has
    const n: usize = 128; // d_model
    const f64_gate = @as(f64, @floatFromInt(k - 1)) * std.math.floatEps(f64);
    const f32_gate = @as(f64, @floatFromInt(k - 1)) * std.math.floatEps(f32);

    // Xoshiro256 named, as model.zig names it: every published number in this
    // project descends from that stream, so the alias is not a promise. The
    // weights are drawn uniform rather than normal because the gate above is
    // about how many roundings a reduction makes, not about their distribution,
    // and a uniform draw is two lines where a normal one is a log and a cos.
    var prng = std.Random.Xoshiro256.init(7);
    const rnd = prng.random();
    var a = try tensor.Tensor.init(std.testing.allocator, m, k);
    defer a.deinit();
    var b = try tensor.Tensor.init(std.testing.allocator, k, n);
    defer b.deinit();
    for (a.data) |*v| v.* = (rnd.float(f32) - 0.5) * 0.04;
    for (b.data) |*v| v.* = (rnd.float(f32) - 0.5) * 0.04;

    var got = try tensor.matmul(a, b);
    defer got.deinit();

    var worst: f64 = 0;
    for (0..m) |i| {
        for (0..n) |j| {
            var exact: f64 = 0;
            var abs_sum: f64 = 0;
            for (0..k) |kk| {
                const product = @as(f64, a.at(i, kk)) * @as(f64, b.at(kk, j));
                exact += product;
                abs_sum += @abs(product);
            }
            // What a reduction in f64 would have stored, in the same f32 the
            // result is stored in, so the only thing left in the difference is
            // the width of the accumulator.
            const wide: f32 = @floatCast(exact);
            const divergence = @abs(@as(f64, got.at(i, j)) - @as(f64, wide)) / abs_sum;
            worst = @max(worst, divergence);
        }
    }

    // Below the f64 gate means the reduction was carried in f64, whatever order
    // it walked k in: the answer then IS the exact reduction and this is zero.
    try std.testing.expect(worst > f64_gate);
    // Above the f32 gate means it is not carrying something wider than f32, or
    // not adding up the terms the loop claims to. Without this half the first
    // one would also pass a matmul that returns noise.
    try std.testing.expect(worst < f32_gate);
}

test "tensor matmul rejects a dimension mismatch" {
    var a = try tensor.Tensor.init(std.testing.allocator, 2, 3);
    defer a.deinit();
    var b = try tensor.Tensor.init(std.testing.allocator, 2, 2);
    defer b.deinit();

    try std.testing.expectError(error.DimensionMismatch, tensor.matmul(a, b));
}
