const std = @import("std");
const optim = @import("optim.zig");
const Tensor = @import("tensor.zig").Tensor;

test "adamw step 1 update is lr times g over g plus eps" {
    // p starts at 0, so the parameter after the step is exactly the update.
    // t is 1 and both moments are zero, so the corrections 1 - 0.9^1 = 0.1 and
    // 1 - 0.95^1 = 0.05 cancel the first update to 1:
    //   m = 0.1 * 1 = 0.1      mhat = 0.1 / 0.1 = 1
    //   v = 0.05 * 1^2 = 0.05  vhat = 0.05 / 0.05 = 1
    //   update = 0.1 * 1 / (sqrt(1) + 1e-8) = 0.1 / 1.00000001
    //           = 0.09999999900000001
    // f32 has a 24 bit mantissa and an ulp of 7.45e-9 at 0.1, so the store
    // lands on one of the two neighbouring f32 values; 1e-7 covers either and
    // is still four orders of magnitude tighter than the 0.8 an implementation
    // that drops eps would miss by.
    var p = try Tensor.init(std.testing.allocator, 1, 1);
    defer p.deinit();
    var g = try Tensor.init(std.testing.allocator, 1, 1);
    defer g.deinit();
    g.set(0, 0, 1);

    var opt = try optim.AdamW.init(std.testing.allocator, p);
    defer opt.deinit();
    try opt.step(&p, g, 0.1, 0.0);

    try std.testing.expectEqual(@as(u64, 1), opt.t);
    try std.testing.expect(p.at(0, 0) < 0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.099999999), -p.at(0, 0), 1e-7);
}

test "adamw a non-zero weight decay adds to the gradient rather than replacing it" {
    // THE CONTRAST the previous version of this test claimed to be and was not:
    // it passed `0.0` for `weight_decay` exactly as the test above it did, with
    // the same `p`, `g` and `lr`, so the two were the same test written twice and
    // neither could fail on anything the other missed. A zero decay over a
    // parameter that starts at zero is `lr * 0 * 0 = 0` -- nothing, by
    // construction, whatever the implementation does with the term.
    //
    // So `p` starts at 1, which is the only reason there is anything to decay,
    // and the same step runs at both values:
    //   wd = 0.0:  p = 1 - 0.1 * 1 / 1.00000001        = 0.90000000
    //   wd = 0.1:  p = 1 - 0.1 * 1 / 1.00000001 - 0.1 * 0.1 * 1
    //                                        = 0.89000000
    // The two differ by the decay term and nothing else, so an implementation
    // that dropped `weight_decay` whenever a gradient was present would land on
    // the first figure here and fail. The zero-gradient decay test below covers
    // the other side of the pair, where the gradient term is 0 and only the
    // decay is left; between them the two terms are never both live anywhere
    // else in this file.
    var p0 = try Tensor.init(std.testing.allocator, 1, 1);
    defer p0.deinit();
    var p1 = try Tensor.init(std.testing.allocator, 1, 1);
    defer p1.deinit();
    var g = try Tensor.init(std.testing.allocator, 1, 1);
    defer g.deinit();
    p0.set(0, 0, 1);
    p1.set(0, 0, 1);
    g.set(0, 0, 1);

    var no_decay = try optim.AdamW.init(std.testing.allocator, p0);
    defer no_decay.deinit();
    var with_decay = try optim.AdamW.init(std.testing.allocator, p1);
    defer with_decay.deinit();

    try no_decay.step(&p0, g, 0.1, 0.0);
    try with_decay.step(&p1, g, 0.1, 0.1);

    try std.testing.expectApproxEqAbs(@as(f32, 0.9), p0.at(0, 0), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.89), p1.at(0, 0), 1e-6);
}

test "adamw eps damps a gradient small enough to reach it" {
    // g = 1e-4 and lr = 1 put the update at
    //   1e-4 / (1e-4 + 1e-8) = 1 / 1.0001 = 0.99990001
    // The first test's g = 1 and lr = 0.1 cannot see eps at all: the whole eps
    // effect there is 1e-9, and the f32 ulp at 0.1 is 7.45e-9, so the store
    // rounds 0.099999999 and a 0.1 with no eps to the same f32. Here the gap
    // to an eps-free 1.0 is 1e-4, a hundred times the tolerance, so dropping
    // eps fails here.
    var p = try Tensor.init(std.testing.allocator, 1, 1);
    defer p.deinit();
    var g = try Tensor.init(std.testing.allocator, 1, 1);
    defer g.deinit();
    g.set(0, 0, 1e-4);

    var opt = try optim.AdamW.init(std.testing.allocator, p);
    defer opt.deinit();
    try opt.step(&p, g, 1.0, 0.0);

    try std.testing.expectApproxEqAbs(@as(f32, 0.99990001), -p.at(0, 0), 1e-6);
}

test "adamw bias correction separates the update from the uncorrected one" {
    // Step 1 from zero moments moves p by
    //   0.1 * 1 / (sqrt(1) + 1e-8) = 0.099999999
    // because the corrections 1 - 0.9^1 = 0.1 and 1 - 0.95^1 = 0.05 divide out
    // the 0.1 and 0.05 the recurrence just wrote.
    // Re-running the same gradient on the moments step 1 left behind, with the
    // step counter rewound so the corrections are 0.1 and 0.05 again, gives
    //   m = 0.9 * 0.1 + 0.1 = 0.19          mhat = 0.19 / 0.1 = 1.9
    //   v = 0.95 * 0.05 + 0.05 = 0.0975     vhat = 0.0975 / 0.05 = 1.95
    //   sqrt(vhat) = 1.39642400
    //   update = 0.1 * 1.9 / 1.39642400 = 0.13606182
    // The same state with the correction dropped answers
    //   0.1 * 0.19 / 0.31224990 = 0.06084870
    // which is 0.075 away, so the tight assertion below is what fails if the
    // correction goes missing. Rewinding the counter and the parameter has to
    // bring the first answer back.
    var p = try Tensor.init(std.testing.allocator, 1, 1);
    defer p.deinit();
    var g = try Tensor.init(std.testing.allocator, 1, 1);
    defer g.deinit();
    g.set(0, 0, 1);

    var opt = try optim.AdamW.init(std.testing.allocator, p);
    defer opt.deinit();
    try opt.step(&p, g, 0.1, 0.0);
    const corrected = p.at(0, 0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.099999999), -corrected, 1e-7);

    const m0 = opt.m.data[0]; // 0.1
    const v0 = opt.v.data[0]; // 0.05
    opt.t = 0;
    p.set(0, 0, 0);
    try opt.step(&p, g, 0.1, 0.0);

    try std.testing.expectApproxEqAbs(@as(f32, 0.13606182), -p.at(0, 0), 1e-7);

    opt.m.data[0] = m0;
    opt.v.data[0] = v0;
    opt.t = 1;
    p.set(0, 0, 0);
    try opt.step(&p, g, 0.1, 0.0);
    try std.testing.expectApproxEqAbs(corrected, p.at(0, 0), 1e-7);
}

test "adamw carries its moments into the second step" {
    // The gradient changes from 1 to 0.25 between steps, which is the only way
    // to see the carried moments: with a constant g the recurrence gives
    // m_t = g(1 - 0.9^t) exactly, so mhat = g at every t and a freshly zeroed
    // moment produces the same answer.
    // Step 1, t = 1: m = 0.1, v = 0.05, corrections 0.1 and 0.05
    //   mhat = 1, vhat = 1, update = 0.1 / 1.00000001 = 0.099999999
    //   p1 = 0 - 0.099999999 = -0.099999999
    // Step 2, t = 2:
    //   m = 0.9 * 0.1 + 0.1 * 0.25 = 0.09 + 0.025 = 0.115
    //   v = 0.95 * 0.05 + 0.05 * 0.0625 = 0.0475 + 0.003125 = 0.050625
    //   corrections 1 - 0.81 = 0.19 and 1 - 0.9025 = 0.0975
    //   mhat = 0.115 / 0.19 = 0.60526316
    //   vhat = 0.050625 / 0.0975 = 0.51923077
    //   sqrt(vhat) = 0.72057669
    //   update = 0.1 * 0.60526316 / (0.72057669 + 1e-8) = 0.08399705
    //   p2 = -0.099999999 - 0.08399705 = -0.18399705
    // m and v live in f32 buffers, so the stores are 0.115000003 and
    // 0.050625007, which move p2 by under 1e-8. Re-zeroing the moments instead
    // would give 0.1 * 0.25 / (0.25 + 1e-8) = 0.099999996 and p2 = -0.199999995,
    // 0.016 away, so the assertion below is the one that fails on a reset.
    var p = try Tensor.init(std.testing.allocator, 1, 1);
    defer p.deinit();
    var g = try Tensor.init(std.testing.allocator, 1, 1);
    defer g.deinit();

    var opt = try optim.AdamW.init(std.testing.allocator, p);
    defer opt.deinit();

    g.set(0, 0, 1);
    try opt.step(&p, g, 0.1, 0.0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.099999999), -p.at(0, 0), 1e-7);

    g.set(0, 0, 0.25);
    try opt.step(&p, g, 0.1, 0.0);
    try std.testing.expectEqual(@as(u64, 2), opt.t);
    try std.testing.expectApproxEqAbs(@as(f32, 0.18399705), -p.at(0, 0), 1e-6);
}

test "adamw weight decay moves a parameter whose gradient is zero" {
    // g = 0 gives mhat = 0 and vhat = 0, so the gradient term is
    // 0 / (0 + 1e-8) = 0 and only the decoupled term is left:
    //   p = 1 - 0.1 * 0 - 0.1 * 0.1 * 1 = 1 - 0.01 = 0.99
    // A coupled update folds the decay into the gradient, giving 0.1 * 0 = 0
    // and p = 1, so this one assertion is the whole decoupled question.
    var p = try Tensor.init(std.testing.allocator, 1, 1);
    defer p.deinit();
    var g = try Tensor.init(std.testing.allocator, 1, 1);
    defer g.deinit();
    p.set(0, 0, 1);

    var opt = try optim.AdamW.init(std.testing.allocator, p);
    defer opt.deinit();
    try opt.step(&p, g, 0.1, 0.1);

    try std.testing.expectApproxEqAbs(@as(f32, 0.99), p.at(0, 0), 1e-6);
}

test "adamw step rejects a gradient that does not match the parameter" {
    // The moments are shaped like the parameter at init, so a gradient of a
    // different width is a caller bug. It has to come back as an error value
    // rather than walk off the end of the moment buffers.
    var p = try Tensor.init(std.testing.allocator, 1, 1);
    defer p.deinit();
    var g = try Tensor.init(std.testing.allocator, 1, 2);
    defer g.deinit();

    var opt = try optim.AdamW.init(std.testing.allocator, p);
    defer opt.deinit();

    try std.testing.expectError(error.LengthMismatch, opt.step(&p, g, 0.1, 0.0));
}

test "adamw init frees the first moment when the second allocation fails" {
    // The two moment buffers are separate allocations, so the second one
    // failing has to release the first. std.testing.allocator under this test
    // is what reports the leak, and the run fails if it finds one.
    var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 1 });
    var like = try Tensor.init(std.testing.allocator, 1, 1);
    defer like.deinit();

    try std.testing.expectError(error.OutOfMemory, optim.AdamW.init(failing.allocator(), like));
}

test "cosine lr hits the warmup, midpoint, and end values" {
    // base_lr 0.1, total 10, warmup 2.
    //   step 0 < 2: 0.1 * 1 / 2 = 0.05
    //   step 2:     progress = 0 / 8 = 0, cos(0) = 1, 0.1 * 0.5 * 2 = 0.1
    //   step 3:     progress = 1 / 8, angle pi/8, cos(pi/8) = 0.92387953
    //               0.1 * 0.5 * 1.92387953 = 0.09619398
    //   step 6:     progress = 4 / 8 = 0.5, cos(pi/2) = 0, 0.1 * 0.5 = 0.05
    //   step 10:    progress = 8 / 8 = 1, cos(pi) = -1, 0.1 * 0.5 * 0 = 0
    try std.testing.expectEqual(@as(f32, 0.05), optim.cosineLR(0, 10, 2, 0.1));
    try std.testing.expectEqual(@as(f32, 0.1), optim.cosineLR(2, 10, 2, 0.1));
    try std.testing.expectApproxEqAbs(@as(f32, 0.09619398), optim.cosineLR(3, 10, 2, 0.1), 1e-7);
    try std.testing.expectEqual(@as(f32, 0.05), optim.cosineLR(6, 10, 2, 0.1));
    try std.testing.expectEqual(@as(f32, 0.0), optim.cosineLR(10, 10, 2, 0.1));
}

test "cosine lr never goes negative across a whole run" {
    // Every step of a 20 step run with 5 warmup steps, at three base rates.
    // The decay half runs from step 5 to step 20, so progress sweeps 0 to 1
    // and cos(pi * progress) sweeps 1 to -1 without leaving [-1, 1].
    for ([_]f32{ 0.001, 0.01, 1.0 }) |base| {
        var step: u64 = 0;
        while (step <= 20) : (step += 1) {
            const lr = optim.cosineLR(step, 20, 5, base);
            try std.testing.expect(lr >= 0);
            try std.testing.expect(std.math.isFinite(lr));
        }
    }
    // A step past the end is a caller bug, but the answer still has to be a
    // rate and not a negative one. progress would be 13 / 15 = 0.866, which is
    // inside the run, so a second run is what puts it past: step 21 of a 10
    // step run with 2 warmup gives progress 19 / 8 = 2.375, and cos of that is
    // below -1, so the clamp is what holds the answer at 0.
    try std.testing.expectEqual(@as(f32, 0.0), optim.cosineLR(21, 10, 2, 0.1));
}

test "cosine lr degrades gracefully when warmup is degenerate" {
    // warmup 5 with total 3 never reaches the decay half, so every step is
    // 0.1 * (step + 1) / 5: 0.02, 0.04, 0.06, 0.08. The warmup divisor is
    // 5, not 0, so nothing divides by zero.
    try std.testing.expectEqual(@as(f32, 0.02), optim.cosineLR(0, 3, 5, 0.1));
    try std.testing.expectEqual(@as(f32, 0.08), optim.cosineLR(3, 3, 5, 0.1));
    // warmup 0 with total 3 goes straight to the decay half, progress
    // 0 / 3 = 0, cos(0) = 1, so the first step is the full 0.1.
    try std.testing.expectEqual(@as(f32, 0.1), optim.cosineLR(0, 3, 0, 0.1));
    // total == warmup leaves an empty decay half. The 0 / (total - warmup) that
    // would be 0 / 0 is guarded to 0 / 1, so step 3 sees progress 0 and 0.1.
    try std.testing.expectEqual(@as(f32, 0.1), optim.cosineLR(3, 3, 3, 0.1));
    // total 0 with no warmup has no decay half at all, and must still answer.
    // The value, not its finiteness: the schedule collapses to `base_lr` on this
    // path, so `isFinite` would hold for any answer at all.
    try std.testing.expectEqual(@as(f32, 0.1), optim.cosineLR(0, 0, 0, 0.1));
}

test "clip by norm scales a gradient that is over the limit" {
    // [3 4] has norm sqrt(9 + 16) = 5. max_norm 1 gives a scale of 1 / 5
    // = 0.2, so 3 becomes 0.6 and 4 becomes 0.8, and the reported norm is the
    // 5 measured before the scale was applied.
    var g = try Tensor.init(std.testing.allocator, 1, 2);
    defer g.deinit();
    g.set(0, 0, 3);
    g.set(0, 1, 4);
    const grads = [_]Tensor{g};

    const norm = optim.clipByNorm(&grads, 1.0);

    try std.testing.expectEqual(@as(f32, 5.0), norm);
    try std.testing.expectApproxEqAbs(@as(f32, 0.6), g.at(0, 0), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.8), g.at(0, 1), 1e-6);
}

test "clip by norm leaves an under-limit gradient untouched" {
    // [0.1 -0.7] has norm sqrt(0.01 + 0.49) = sqrt(0.5) = 0.70710678, under
    // the limit of 100, so no scaling runs. Compared as raw bit patterns,
    // because a value comparison would also pass a round trip through f64.
    var g = try Tensor.init(std.testing.allocator, 1, 2);
    defer g.deinit();
    g.set(0, 0, 0.1);
    g.set(0, 1, -0.7);
    const before = [_]u32{ @as(u32, @bitCast(g.at(0, 0))), @as(u32, @bitCast(g.at(0, 1))) };
    const grads = [_]Tensor{g};

    const norm = optim.clipByNorm(&grads, 100.0);

    try std.testing.expectApproxEqAbs(@as(f32, 0.70710678), norm, 1e-6);
    try std.testing.expectEqual(before[0], @as(u32, @bitCast(g.at(0, 0))));
    try std.testing.expectEqual(before[1], @as(u32, @bitCast(g.at(0, 1))));
}

test "clip by norm scales every tensor by the global factor" {
    // [3 0] and [4] have a combined norm of 5 against a limit of 1, so both
    // take the same 1 / 5 = 0.2 and land on [0.6 0] and [0.8]. Clipping each
    // tensor on its own would use 1 / 3 and 1 / 4 instead and give
    // [1 0] and [1], a factor of 1.5 to 1.67 too little shrink.
    var a = try Tensor.init(std.testing.allocator, 1, 2);
    defer a.deinit();
    a.set(0, 0, 3);
    a.set(0, 1, 0);
    var b = try Tensor.init(std.testing.allocator, 1, 1);
    defer b.deinit();
    b.set(0, 0, 4);
    const grads = [_]Tensor{ a, b };

    const norm = optim.clipByNorm(&grads, 1.0);

    try std.testing.expectEqual(@as(f32, 5.0), norm);
    try std.testing.expectApproxEqAbs(@as(f32, 0.6), a.at(0, 0), 1e-6);
    try std.testing.expectEqual(@as(f32, 0.0), a.at(0, 1));
    try std.testing.expectApproxEqAbs(@as(f32, 0.8), b.at(0, 0), 1e-6);
}

test "clip by norm on a zero gradient neither divides by zero nor makes NaN" {
    // Every element is 0, so the norm is sqrt(0) = 0 and the scale would be
    // 1 / 0. The call has to answer 0 and leave the tensor at 0, not at NaN.
    var g = try Tensor.init(std.testing.allocator, 2, 2);
    defer g.deinit();
    const grads = [_]Tensor{g};

    const norm = optim.clipByNorm(&grads, 1.0);

    try std.testing.expectEqual(@as(f32, 0.0), norm);
    for (g.data) |v| {
        try std.testing.expectEqual(@as(f32, 0.0), v);
        try std.testing.expect(!std.math.isNan(v));
    }
}

test "the same parameter and gradient sequence is bit identical across runs" {
    // Two runs of one seed have to agree to the byte, which is what the committed
    // loss curve claims, so the update order has to be fixed. Three steps over a 2x2 parameter with two
    // different gradients, run twice, compared as raw bit patterns.
    const grads_seq = [_][4]f32{
        .{ 0.5, -1.25, 2.0, 0.75 },
        .{ -0.5, 0.25, 0.125, -2.0 },
        .{ 1.0, 1.0, 1.0, 1.0 },
    };
    const rates = [_]f32{ 0.05, 0.02, 0.01 };

    var first: [4]f32 = undefined;
    var run: usize = 0;
    while (run < 2) : (run += 1) {
        var p = try Tensor.init(std.testing.allocator, 2, 2);
        defer p.deinit();
        var g = try Tensor.init(std.testing.allocator, 2, 2);
        defer g.deinit();
        var opt = try optim.AdamW.init(std.testing.allocator, p);
        defer opt.deinit();

        for (grads_seq, 0..) |vals, step| {
            for (vals, 0..) |v, i| g.set(i / 2, i % 2, v);
            try opt.step(&p, g, rates[step], 0.01);
        }
        try std.testing.expectEqual(@as(u64, 3), opt.t);

        if (run == 0) {
            first = p.data[0..4].*;
        } else {
            for (first, p.data) |a, b| {
                try std.testing.expectEqual(@as(u32, @bitCast(a)), @as(u32, @bitCast(b)));
            }
        }
    }
}
