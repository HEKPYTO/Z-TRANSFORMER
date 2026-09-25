const std = @import("std");
const loss = @import("loss.zig");
const Tensor = @import("tensor.zig").Tensor;

test "cross entropy of uniform logits is exactly ln of the vocabulary" {
    // 3 rows of 4 zeros. Every row is uniform, so softmax is 1/4 on each entry
    // whatever the target is, and the loss of a row is
    //   logsumexp([0 0 0 0]) - 0 = ln(4) = 1.3862943611198906
    // T rows of the same value average back to that same value.
    var logits = try Tensor.init(std.testing.allocator, 3, 4);
    defer logits.deinit();
    const targets = [_]u32{ 0, 1, 3 };

    const got = try loss.forward(logits, &targets);

    try std.testing.expectEqual(1.3862943611198906, got);
}

test "cross entropy is logsumexp minus the target logit" {
    //   [0 1 2]   [ 3 ]     row 0, target 2
    //   [1 0 0] @ [ 1 ]  =  row 1, target 0
    //             [ 1 ]
    // row 0, max = 2, shifted [-2 -1 0]
    //   sum = e^-2 + e^-1 + 1 = 0.1353352832 + 0.3678794412 + 1 = 1.5032147244
    //   logsumexp = 2 + ln(1.5032147244) = 2.4076059644
    //   loss_0 = 2.4076059644 - logits[0][2] = 2.4076059644 - 2 = 0.4076059644
    // row 1, max = 1, shifted [0 -1 -1]
    //   sum = 1 + 2*e^-1 = 1 + 0.7357588823 = 1.7357588823
    //   logsumexp = 1 + ln(1.7357588823) = 1.5514447139
    //   loss_1 = 1.5514447139 - logits[1][0] = 1.5514447139 - 1 = 0.5514447139
    // mean = (0.4076059644 + 0.5514447139) / 2 = 0.9590506783 / 2 = 0.4795253392
    var logits = try Tensor.init(std.testing.allocator, 2, 3);
    defer logits.deinit();
    const rows = [_][3]f32{ .{ 0, 1, 2 }, .{ 1, 0, 0 } };
    for (rows, 0..) |r, i| for (r, 0..) |v, j| logits.set(i, j, v);
    const targets = [_]u32{ 2, 0 };

    const got = try loss.forward(logits, &targets);

    try std.testing.expectApproxEqAbs(@as(f64, 0.4795253391882158), got, 1e-9);
}

test "cross entropy averages over rows instead of summing them" {
    // 4 rows of 2 zeros. A row costs ln(2) = 0.6931471805599453 whatever its
    // target, so the mean over 4 rows is 0.6931471805599453 and the sum would
    // be 4 * 0.6931471805599453 = 2.772588722239781. The two differ by 4x, so
    // this pins the division by the row count.
    var logits = try Tensor.init(std.testing.allocator, 4, 2);
    defer logits.deinit();
    const targets = [_]u32{ 0, 1, 0, 1 };

    const got = try loss.forward(logits, &targets);

    try std.testing.expectEqual(0.6931471805599453, got);
}

test "cross entropy rejects a target count that disagrees with the row count" {
    var logits = try Tensor.init(std.testing.allocator, 3, 2);
    defer logits.deinit();
    const too_few = [_]u32{ 0, 1 };
    const too_many = [_]u32{ 0, 1, 0, 1 };

    try std.testing.expectError(error.TargetCountMismatch, loss.forward(logits, &too_few));
    try std.testing.expectError(error.TargetCountMismatch, loss.forward(logits, &too_many));
}

test "cross entropy rejects a target index past the vocabulary" {
    // 3 rows of 5 columns, so a target of 5 is one past the end. The call has
    // to come back as an error value; a panic or a wrapped index would take the
    // whole test process down instead, and expectError would not be reached.
    var logits = try Tensor.init(std.testing.allocator, 3, 5);
    defer logits.deinit();
    const past_end = [_]u32{ 0, 4, 5 };
    // Truncating to a smaller width would fold this onto a valid column 0.
    const all_ones = [_]u32{ std.math.maxInt(u32), std.math.maxInt(u32), std.math.maxInt(u32) };

    try std.testing.expectError(error.TargetOutOfRange, loss.forward(logits, &past_end));
    try std.testing.expectError(error.TargetOutOfRange, loss.forward(logits, &all_ones));
}

test "cross entropy rejects an empty batch instead of dividing by zero" {
    var logits = try Tensor.init(std.testing.allocator, 0, 4);
    defer logits.deinit();
    const no_targets = [_]u32{};

    try std.testing.expectError(error.EmptyBatch, loss.forward(logits, &no_targets));
}

test "raising the target logit lowers the loss" {
    var logits = try Tensor.init(std.testing.allocator, 1, 3);
    defer logits.deinit();
    const targets = [_]u32{0};

    const before = try loss.forward(logits, &targets);
    try std.testing.expectEqual(1.0986122886681098, before); // ln(3), all logits zero
    logits.set(0, 0, 1);
    const after = try loss.forward(logits, &targets);

    try std.testing.expect(after < before);
}

test "a logit of 1e30 stays finite" {
    // A softmax vector would underflow the two small entries to exactly zero
    // probability and report an infinite loss for them. The logsumexp form
    // keeps the row max outside the exponent, so nothing overflows.
    var logits = try Tensor.init(std.testing.allocator, 2, 3);
    defer logits.deinit();
    for (0..2) |r| logits.set(r, 0, 1e30);
    const on_the_big = [_]u32{ 0, 0 };
    const off_the_big = [_]u32{ 1, 1 };

    const near_zero = try loss.forward(logits, &on_the_big);
    try std.testing.expect(!std.math.isNan(near_zero));
    try std.testing.expect(!std.math.isInf(near_zero));
    // 1e30 + ln(1) - 1e30, which cancels exactly.
    try std.testing.expectEqual(0.0, near_zero);

    const huge = try loss.forward(logits, &off_the_big);
    try std.testing.expect(!std.math.isNan(huge));
    try std.testing.expect(!std.math.isInf(huge));
    // 1e30 in f32 is 1000000015047466219876688855040 exactly, 1.5e-8 relative
    // off the decimal literal, so 1e-6 pins the magnitude without overfitting
    // the f32 rounding.
    try std.testing.expectApproxEqRel(@as(f64, 1e30), huge, 1e-6);
}
