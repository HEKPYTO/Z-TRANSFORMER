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
        // f64 accumulator, narrowed once per row: 4096 f32 squares lose the low
        // bits of the row and drift the scale by more than 1e-6, which the
        // numerics tests hold this op to.
        const rms: f32 = @floatCast(@sqrt(sum_sq / @as(f64, @floatFromInt(d)) + eps));

        const y_row = out.row(r);
        for (x_row, 0..) |v, i| y_row[i] = v / rms * w[i];
    }
    return out;
}
