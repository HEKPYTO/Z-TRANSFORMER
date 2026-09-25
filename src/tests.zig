const std = @import("std");
const lib = @import("ztransformer");
const tensor = @import("tensor.zig");

test "version is non-empty" {
    try std.testing.expect(lib.version.len > 0);
}

test "name returns the project name" {
    try std.testing.expectEqualStrings("Z-TRANSFORMER", lib.name());
}

test "tensor init allocates rows times cols zeroed" {
    var t = try tensor.Tensor.init(std.testing.allocator, 3, 4);
    defer t.deinit();

    try std.testing.expectEqual(@as(usize, 12), t.data.len);
    try std.testing.expectEqual(@as(usize, 3), t.rows);
    try std.testing.expectEqual(@as(usize, 4), t.cols);
    for (t.data) |v| try std.testing.expectEqual(@as(f32, 0), v);
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

test "tensor matmul rejects a dimension mismatch" {
    var a = try tensor.Tensor.init(std.testing.allocator, 2, 3);
    defer a.deinit();
    var b = try tensor.Tensor.init(std.testing.allocator, 2, 2);
    defer b.deinit();

    try std.testing.expectError(error.DimensionMismatch, tensor.matmul(a, b));
}

const norm = @import("norm.zig");
const rope = @import("rope.zig");
const mlp = @import("mlp.zig");
const attention = @import("attention.zig");
const loss = @import("loss.zig");

test "every numerics module is wired" {
    _ = norm;
    _ = rope;
    _ = mlp;
    _ = attention;
    _ = loss;
    try std.testing.expect(true);
}

comptime {
    _ = @import("norm_test.zig");
    _ = @import("rope_test.zig");
    _ = @import("mlp_test.zig");
    _ = @import("attention_test.zig");
    _ = @import("loss_test.zig");
}
