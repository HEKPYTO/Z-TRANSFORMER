//! RMSNorm with eps 1e-5 and the weight multiply fused into the same pass.
const std = @import("std");
const Tensor = @import("tensor.zig").Tensor;

/// The additive epsilon inside the RMS.
///
/// Public because it is not only the forward pass's constant: the gradient in
/// `autograd.normBackward` differentiates the same expression and therefore needs
/// the same number, and the parity exporter writes it into the reference config.
/// Three private copies of `1e-5` is two too many — change one and the gradient
/// silently differentiates a different function than the forward pass evaluates,
/// which no test that compares a gradient to a finite difference of the forward
/// pass would catch.
pub const eps: f64 = 1e-5;

/// Llama-3 RMSNorm over the last axis. `x` is [T, d], `weight` is a length d
/// vector laid out as [1, d] or [d, 1], and the result is [T, d].
pub fn forward(allocator: std.mem.Allocator, x: Tensor, weight: Tensor) !Tensor {
    const d = x.cols;
    // A Tensor is dense, so a length d weight is d contiguous elements in both
    // accepted layouts. Any other length is a shape error, not a panic.
    if (weight.data.len != d) return error.DimensionMismatch;
    const w = weight.data[0..d];

    var out = try Tensor.init(allocator, x.rows, d);
    for (0..x.rows) |r| {
        const x_row = x.rowConst(r);
        var sum_sq: f64 = 0;
        for (x_row) |v| sum_sq += @as(f64, v) * @as(f64, v);
        // f64 accumulator, narrowed once per row. The justification is the 512-row
        // measurement written down in `norm_test.zig`, which is the widest row this
        // model actually builds: f64 lands 5.7e-08 from exact, f32 lands 1.667e-06,
        // and the 1e-6 tolerance sits between them.
        //
        // An earlier version of this comment justified the same choice with a
        // 4096-wide row -- on the grounds that 4096 f32 squares "drift the scale by
        // more than 1e-6". That is true of f32 in general, and `norm_test.zig` says
        // in as many words that 4096 "is a shape nothing here ever produces, and a
        // tolerance picked for it says nothing about the rows that run". The comment
        // was arguing for a decision with a measurement the enforcing test disowned.
        const rms: f32 = @floatCast(@sqrt(sum_sq / @as(f64, @floatFromInt(d)) + eps));

        const y_row = out.row(r);
        for (x_row, 0..) |v, i| y_row[i] = v / rms * w[i];
    }
    return out;
}
