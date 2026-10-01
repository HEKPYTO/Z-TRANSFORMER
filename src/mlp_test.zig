const std = @import("std");
const mlp = @import("mlp.zig");
const tensor = @import("tensor.zig");
const Tensor = tensor.Tensor;

// f32 keeps about seven significant digits, so a hand-written fp64 literal
// cannot be asserted exactly. 1e-6 is the tolerance this file uses throughout.
const tol: f32 = 1e-6;

fn make(allocator: std.mem.Allocator, rows: usize, cols: usize, vals: []const f32) !Tensor {
    var t = try Tensor.init(allocator, rows, cols);
    errdefer t.deinit();
    for (vals, 0..) |v, i| t.set(i / cols, i % cols, v);
    return t;
}

test "mlp silu of one is the hand computed activation" {
    var x = try make(std.testing.allocator, 1, 1, &[_]f32{1});
    defer x.deinit();
    var w_gate = try make(std.testing.allocator, 1, 1, &[_]f32{1});
    defer w_gate.deinit();
    var w_up = try make(std.testing.allocator, 1, 1, &[_]f32{1});
    defer w_up.deinit();
    var w_down = try make(std.testing.allocator, 1, 1, &[_]f32{1});
    defer w_down.deinit();

    var got = try mlp.forward(std.testing.allocator, x, w_gate, w_up, w_down);
    defer got.deinit();

    try std.testing.expectEqual(@as(usize, 1), got.rows);
    try std.testing.expectEqual(@as(usize, 1), got.cols);
    // silu(1) = 1 * 1 / (1 + exp(-1)) = 1 / 1.3678794411714423
    try std.testing.expectApproxEqAbs(@as(f32, 0.7310585786300049), got.at(0, 0), tol);
}

test "mlp silu of zero is exactly zero" {
    try std.testing.expectEqual(@as(f32, 0), mlp.silu(0));
}

test "mlp silu stays a small negative number where exp(-z) overflows" {
    // exp(-z) leaves f32 range just past z = -89, and 1 / inf would swallow the
    // whole negative tail. The answer here is about -3.7e-42.
    const got = mlp.silu(-100);

    try std.testing.expect(std.math.isFinite(got));
    try std.testing.expect(got < 0);
    try std.testing.expect(got > -1e-30);
}

test "mlp silu of a huge negative input is not NaN or infinite" {
    const got = mlp.silu(-1e30);

    try std.testing.expect(std.math.isFinite(got));
    // exp(-1e30) is underflowed to zero, so zero is the closest f32 answer.
    try std.testing.expectApproxEqAbs(@as(f32, 0), got, 1e-30);
}

test "mlp forward 2x2 with a hidden width of three matches hand computed values" {
    // x       w_gate   w_up     w_down
    // [1 1]   [1 0 1]  [2 0 0]  [1 0]
    // [1 0] @ [0 1 0]  [0 0 3]  [0 1]
    //         [-----]  [-----]  [0 1]
    //
    // gate = [[1 1 1]  up = [[2 0 3]   a = silu(gate) * up
    //        [1 0 1]       [2 0 0]      = [[2s 0 3s]
    //                                            [2s 0 0]]      s = silu(1)
    // out = a @ w_down = [[2s 3s] = [[1.4621171573  2.1931757359]
    //                       [2s 0]    [1.4621171573  0          ]]
    var x = try make(std.testing.allocator, 2, 2, &[_]f32{ 1, 1, 1, 0 });
    defer x.deinit();
    var w_gate = try make(std.testing.allocator, 2, 3, &[_]f32{ 1, 0, 1, 0, 1, 0 });
    defer w_gate.deinit();
    var w_up = try make(std.testing.allocator, 2, 3, &[_]f32{ 2, 0, 0, 0, 0, 3 });
    defer w_up.deinit();
    var w_down = try make(std.testing.allocator, 3, 2, &[_]f32{ 1, 0, 0, 1, 0, 1 });
    defer w_down.deinit();

    var got = try mlp.forward(std.testing.allocator, x, w_gate, w_up, w_down);
    defer got.deinit();

    try std.testing.expectEqual(@as(usize, 2), got.rows);
    try std.testing.expectEqual(@as(usize, 2), got.cols);
    try std.testing.expectApproxEqAbs(@as(f32, 1.4621171572600098), got.at(0, 0), tol);
    try std.testing.expectApproxEqAbs(@as(f32, 2.1931757358900147), got.at(0, 1), tol);
    try std.testing.expectApproxEqAbs(@as(f32, 1.4621171572600098), got.at(1, 0), tol);
    try std.testing.expectApproxEqAbs(@as(f32, 0), got.at(1, 1), tol);
}

test "mlp forward with a negative gate produces a negative output" {
    // gate = -1, up = 1, w_down = 1
    // silu(-1) = -1 * 1 / (1 + exp(1)) = -1 / 3.718281828459045
    var x = try make(std.testing.allocator, 1, 1, &[_]f32{1});
    defer x.deinit();
    var w_gate = try make(std.testing.allocator, 1, 1, &[_]f32{-1});
    defer w_gate.deinit();
    var w_up = try make(std.testing.allocator, 1, 1, &[_]f32{1});
    defer w_up.deinit();
    var w_down = try make(std.testing.allocator, 1, 1, &[_]f32{1});
    defer w_down.deinit();

    var got = try mlp.forward(std.testing.allocator, x, w_gate, w_up, w_down);
    defer got.deinit();

    try std.testing.expect(got.at(0, 0) < 0);
    try std.testing.expectApproxEqAbs(@as(f32, -0.26894142136999512), got.at(0, 0), tol);
}

test "mlp forward on a zero row of x gives a zero row of output" {
    // x = [[0 0]  w_gate = w_down = identity  w_up = ones
    //       [1 1]]
    var x = try make(std.testing.allocator, 2, 2, &[_]f32{ 0, 0, 1, 1 });
    defer x.deinit();
    var identity = try make(std.testing.allocator, 2, 2, &[_]f32{ 1, 0, 0, 1 });
    defer identity.deinit();
    var ones = try make(std.testing.allocator, 2, 2, &[_]f32{ 1, 1, 1, 1 });
    defer ones.deinit();

    var got = try mlp.forward(std.testing.allocator, x, identity, ones, identity);
    defer got.deinit();

    try std.testing.expectEqual(@as(f32, 0), got.at(0, 0));
    try std.testing.expectEqual(@as(f32, 0), got.at(0, 1));
    for (got.data) |v| try std.testing.expect(std.math.isFinite(v));
    // The non-zero row still has to come out live, so a collapsed pass is caught.
    try std.testing.expect(got.at(1, 0) > 0);
}

test "mlp forward rejects every mismatched weight" {
    var x = try make(std.testing.allocator, 2, 2, &[_]f32{ 1, 2, 3, 4 });
    defer x.deinit();
    var square = try make(std.testing.allocator, 2, 2, &[_]f32{ 1, 0, 0, 1 });
    defer square.deinit();
    var three_rows = try make(std.testing.allocator, 3, 2, &[_]f32{ 1, 0, 0, 1, 1, 1 });
    defer three_rows.deinit();

    try std.testing.expectError(
        error.DimensionMismatch,
        mlp.forward(std.testing.allocator, x, three_rows, square, square),
    );
    try std.testing.expectError(
        error.DimensionMismatch,
        mlp.forward(std.testing.allocator, x, square, three_rows, square),
    );
    // One row per hidden column, so w_down needs two rows for a hidden width of two.
    try std.testing.expectError(
        error.DimensionMismatch,
        mlp.forward(std.testing.allocator, x, square, square, three_rows),
    );

    var narrow = try make(std.testing.allocator, 2, 1, &[_]f32{ 1, 0 });
    defer narrow.deinit();
    // w_gate and w_up have to agree on the hidden width before the elementwise step.
    try std.testing.expectError(
        error.DimensionMismatch,
        mlp.forward(std.testing.allocator, x, narrow, square, narrow),
    );
}
