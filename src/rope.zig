//! Rotary position embedding, theta 500000, Llama-3 pairwise layout.

const std = @import("std");
const Tensor = @import("tensor.zig").Tensor;

/// Rotates `[T, n_heads * head_dim]` so row `r` carries absolute position
/// `pos + r`. Returns `error.OddHeadDim` when `head_dim` is zero or odd, and
/// `error.DimensionMismatch` when the row is not a whole number of head blocks.
/// Neither is a silent truncate.
///
/// The row is `n_heads` heads laid out end to end, each `head_dim` wide, and
/// each one rotates on its own. Llama-3 splits a head's own dim in half:
/// element `i` pairs with element `i + head_dim/2` of the *same* head, never
/// with its neighbour `i + 1` the way the original paper interleaves. Taking
/// `x.cols` as the head dim instead pairs element `i` of one head with element
/// `i` of another, and divides every frequency exponent by `n_heads`.
///
/// Frequency, angle and the two multiply-adds are all `f64`, because `t * freq`
/// grows large enough that `cos` and `sin` of an `f32` angle lose the low bits.
/// The one narrowing back to `f32` is the store into the output tensor.
pub fn forward(
    allocator: std.mem.Allocator,
    x: Tensor,
    pos: usize,
    theta: f64,
    head_dim: usize,
) !Tensor {
    if (head_dim == 0 or head_dim % 2 != 0) return error.OddHeadDim;
    if (x.cols % head_dim != 0) return error.DimensionMismatch;
    const half = head_dim / 2;

    var out = try Tensor.init(allocator, x.rows, x.cols);

    for (0..x.rows) |r| {
        const src = x.rowConst(r);
        const dst = out.row(r);
        const t: f64 = @floatFromInt(pos + r);
        for (0..x.cols / head_dim) |h| {
            const base = h * head_dim;
            for (0..half) |i| {
                // 2i/head_dim in f64, not usize: integer division folds every
                // exponent to zero, which would collapse all frequencies onto 1.
                const exponent = @as(f64, @floatFromInt(2 * i)) / @as(f64, @floatFromInt(head_dim));
                const angle = t / std.math.pow(f64, theta, exponent);
                const c = @cos(angle);
                const s = @sin(angle);
                const lo: f64 = @floatCast(src[base + i]);
                const hi: f64 = @floatCast(src[base + i + half]);
                dst[base + i] = @floatCast(lo * c - hi * s);
                dst[base + i + half] = @floatCast(hi * c + lo * s);
            }
        }
    }
    return out;
}
