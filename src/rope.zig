//! Rotary position embedding, theta 500000, Llama-3 pairwise layout.

const std = @import("std");
const Tensor = @import("tensor.zig").Tensor;

/// Rotates `[T, d]` so row `r` carries absolute position `pos + r`. Returns
/// `error.OddHeadDim` when `d` is odd, never a silent truncate.
///
/// Llama-3 splits the head dim in half: element `i` pairs with element `i + d/2`,
/// never with its neighbour `i + 1` the way the original paper interleaves.
///
/// Frequency, angle and the two multiply-adds are all `f64`, because `t * freq`
/// grows large enough that `cos` and `sin` of an `f32` angle lose the low bits.
/// The one narrowing back to `f32` is the store into the output tensor.
pub fn forward(allocator: std.mem.Allocator, x: Tensor, pos: usize, theta: f64) !Tensor {
    const d = x.cols;
    if (d % 2 != 0) return error.OddHeadDim;
    const half = d / 2;

    var out = try Tensor.init(allocator, x.rows, d);

    for (0..x.rows) |r| {
        const src = x.rowConst(r);
        const dst = out.row(r);
        const t: f64 = @floatFromInt(pos + r);
        for (0..half) |i| {
            // 2i/d in f64, not usize: integer division folds every exponent to
            // zero, which would collapse all frequencies onto 1.
            const exponent = @as(f64, @floatFromInt(2 * i)) / @as(f64, @floatFromInt(d));
            const angle = t / std.math.pow(f64, theta, exponent);
            const c = @cos(angle);
            const s = @sin(angle);
            const lo: f64 = @floatCast(src[i]);
            const hi: f64 = @floatCast(src[i + half]);
            dst[i] = @floatCast(lo * c - hi * s);
            dst[i + half] = @floatCast(hi * c + lo * s);
        }
    }
    return out;
}
