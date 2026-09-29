//! SwiGLU feed-forward: silu(gate) times up, then the down projection.
const std = @import("std");
const model = @import("model.zig");
const tensor = @import("tensor.zig");
const Tensor = tensor.Tensor;

/// One gated activation per hidden column:
///     gate = x @ w_gate
///     up   = x @ w_up
///     out  = (silu(gate) * up) @ w_down
///
/// `x` is [T, d], `w_gate` and `w_up` are [d, h], `w_down` is [h, d], and `out`
/// is [T, d]. Every reduction runs in the fixed order `matmul` already uses, so
/// a run is bit-identical to the last one.
pub fn forward(
    allocator: std.mem.Allocator,
    x: Tensor,
    w_gate: Tensor,
    w_up: Tensor,
    w_down: Tensor,
) !Tensor {
    return forwardWith(allocator, x, w_gate, w_up, w_down, null, 0);
}

/// `forward`, with the three tensors the hand-written backward reads handed to
/// `sink` as they are produced: `mlp_gate`, `mlp_up` and `mlp_hidden`.
///
/// They live here rather than in the caller because the caller cannot see them.
/// `autograd` needs all three to walk SwiGLU out again, and rebuilding them from
/// `x` is two of the three most expensive matmuls in a block, so it is the
/// difference between collecting the forward's own work and computing it twice.
/// `layer` is the sink's layer index and is unused when `sink` is null.
pub fn forwardWith(
    allocator: std.mem.Allocator,
    x: Tensor,
    w_gate: Tensor,
    w_up: Tensor,
    w_down: Tensor,
    sink: ?*model.Sink,
    layer: usize,
) !Tensor {
    // The elementwise step needs one up column per gate column, and the caller
    // promised an output as wide as its input, so both are checked before the
    // first allocation. The per-axis row checks belong to `matmul`.
    if (w_gate.cols != w_up.cols or w_down.cols != x.cols) return error.DimensionMismatch;

    var gate = try tensor.matmul(x, w_gate);
    defer gate.deinit();
    if (sink) |s| s.put(s, .mlp_gate, layer, gate);
    var up = try tensor.matmul(x, w_up);
    defer up.deinit();
    if (sink) |s| s.put(s, .mlp_up, layer, up);

    var a = try Tensor.init(allocator, gate.rows, gate.cols);
    defer a.deinit();
    for (a.data, gate.data, up.data) |*dst, g, u| dst.* = silu(g) * u;
    if (sink) |s| s.put(s, .mlp_hidden, layer, a);

    return tensor.matmul(a, w_down);
}

/// silu(z) = z * sigmoid(z).
///
/// The textbook form z / (1 + exp(-z)) leaves f32 range at z = -89, where
/// exp(-z) becomes +inf and the negative tail collapses to -0.0. Negating the
/// exponent on that side keeps exp in range and preserves the true small
/// negative result, down to the subnormals.
pub fn silu(z: f32) f32 {
    if (z >= 0) return z / (1 + @exp(-z));
    const e = @exp(z);
    return z * e / (1 + e);
}
