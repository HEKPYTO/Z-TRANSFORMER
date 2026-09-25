//! AdamW with decoupled weight decay, cosine schedule with warmup, global gradient clipping.
//!
//! This module consumes gradients it is handed. It computes none, and it has no
//! backward pass, so the caller owns the backward functions and the buffers they
//! write into. Every loop here runs in a fixed order over a flat buffer, because
//! Phase 2 wants two runs of one seed to agree bit for bit.

const std = @import("std");
const Tensor = @import("tensor.zig").Tensor;

const beta1: f64 = 0.9;
const beta2: f64 = 0.95;
const eps: f64 = 1e-8;

pub const AdamW = struct {
    /// First and second moment estimates, shaped like the parameter and owned
    /// by the caller so one optimizer state can outlive the tensors it was
    /// built from.
    m: Tensor,
    v: Tensor,
    /// Completed steps. Bias correction divides by 1 - beta^t, so the count
    /// has to be the number of steps already taken and is incremented before
    /// the update. Counting from 0 would divide by zero on the first call.
    t: u64,

    pub fn init(allocator: std.mem.Allocator, like: Tensor) !AdamW {
        var m = try Tensor.init(allocator, like.rows, like.cols);
        errdefer m.deinit();
        return .{ .m = m, .v = try Tensor.init(allocator, like.rows, like.cols), .t = 0 };
    }

    pub fn deinit(self: *AdamW) void {
        self.m.deinit();
        self.v.deinit();
    }

    /// One parameter update, in place.
    pub fn step(self: *AdamW, p: *Tensor, g: Tensor, lr: f32, weight_decay: f32) !void {
        if (g.data.len != p.data.len or
            self.m.data.len != p.data.len or
            self.v.data.len != p.data.len) return error.LengthMismatch;

        self.t += 1;
        const rate: f64 = lr;
        const decay: f64 = weight_decay;
        const bc1 = 1.0 - std.math.pow(f64, beta1, @as(f64, @floatFromInt(self.t)));
        const bc2 = 1.0 - std.math.pow(f64, beta2, @as(f64, @floatFromInt(self.t)));

        // The moments are f32 buffers but the arithmetic is f64, so the stored
        // parameter is the correctly rounded f32 of the exact update. Run at
        // f32, each m and v store rounds by up to 6e-8 relative, the ratio and
        // the parameter store round twice more, and the parameter keeps all of
        // it. A weight of 100 at lr 0.1 moves 10 per step, so 6e-8 on the
        // update is 6e-7 of parameter error that no later step takes back out.
        for (0..p.data.len) |i| {
            const gi: f64 = g.data[i];
            const mi = beta1 * @as(f64, self.m.data[i]) + (1.0 - beta1) * gi;
            const vi = beta2 * @as(f64, self.v.data[i]) + (1.0 - beta2) * gi * gi;
            self.m.data[i] = @floatCast(mi);
            self.v.data[i] = @floatCast(vi);

            const pi: f64 = p.data[i];
            const mhat = mi / bc1;
            const vhat = vi / bc2;
            // Decoupled: the decay scales the parameter, not the gradient, so a
            // parameter with no gradient still shrinks.
            const next = pi - rate * (mhat / (@sqrt(vhat) + eps)) - rate * decay * pi;
            p.data[i] = @floatCast(next);
        }
    }
};

/// Learning rate for `step`, 0-based, over a `total` step run with `warmup`
/// linear steps before the cosine half.
pub fn cosineLR(step: u64, total: u64, warmup: u64, base_lr: f32) f32 {
    const base: f64 = base_lr;
    if (step < warmup) {
        return @floatCast(base * (@as(f64, @floatFromInt(step)) + 1.0) /
            @as(f64, @floatFromInt(warmup)));
    }
    // A run whose warmup reaches its total has an empty decay half, and
    // (total - warmup) is then 0. Stepping over that one keeps progress at 0
    // instead of dividing by zero.
    const decay_len = if (total > warmup) total - warmup else 1;
    // Past the last step, progress would exceed 1 and cos would fall below -1,
    // handing back a negative rate. Clamping ends the decay at 0.
    const progress = @min(
        @as(f64, @floatFromInt(step - warmup)) / @as(f64, @floatFromInt(decay_len)),
        1.0,
    );
    return @floatCast(base * 0.5 * (1.0 + @cos(std.math.pi * progress)));
}

/// Scale `grads` in place by `max_norm` over their combined L2 norm when that
/// norm is over the limit, and report the norm measured before scaling.
pub fn clipByNorm(grads: []const Tensor, max_norm: f32) f32 {
    // One f64 accumulator across every element of every tensor. A GPT-mini has
    // on the order of 1e7 gradient elements, and an f32 running sum of that
    // many squares drifts by up to n * eps = 1e7 * 6e-8 = 0.6 relative, which
    // is a clip factor up to 60 percent off. f64 holds the same sum to about
    // 1e-9.
    var sum: f64 = 0;
    for (grads) |g| {
        for (g.data) |v| {
            const x: f64 = v;
            sum += x * x;
        }
    }
    const norm = @sqrt(sum);
    // Written as a negated comparison so a norm of 0, which is not over any
    // limit and has no scale to divide by, returns before the division.
    if (!(norm > @as(f64, max_norm))) return @floatCast(norm);

    const scale = @as(f64, max_norm) / norm;
    for (grads) |g| {
        for (g.data) |*v| v.* = @floatCast(@as(f64, v.*) * scale);
    }
    return @floatCast(norm);
}
