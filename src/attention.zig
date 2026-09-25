//! Grouped-query causal attention with a single-pass softmax.

const std = @import("std");
const tensor = @import("tensor.zig");

const Tensor = tensor.Tensor;

pub const Config = struct {
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
};

/// Causal grouped-query attention over q, k and v that already carry their
/// position embedding. q is [T, n_heads * head_dim], k and v are
/// [T, n_kv_heads * head_dim], and the result is [T, n_heads * head_dim] with
/// head h in columns h * head_dim .., so every head slice is contiguous.
///
/// Query head h reads kv head h / (n_heads / n_kv_heads) and only keys at or
/// before its own position.
pub fn forward(allocator: std.mem.Allocator, q: Tensor, k: Tensor, v: Tensor, cfg: Config) !Tensor {
    // A zero kv head count divides by zero, a zero head dim makes the scale
    // infinite, and a ragged split leaves a query head with no kv head.
    if (cfg.n_kv_heads == 0 or cfg.head_dim == 0 or cfg.n_heads % cfg.n_kv_heads != 0) {
        return error.InvalidHeadConfig;
    }
    if (k.rows != v.rows or k.rows != q.rows or
        k.cols != cfg.n_kv_heads * cfg.head_dim or
        v.cols != k.cols or q.cols != cfg.n_heads * cfg.head_dim)
    {
        return error.DimensionMismatch;
    }

    const dim = cfg.head_dim;
    const group = cfg.n_heads / cfg.n_kv_heads;
    // Scores stay f64 and narrow to f32 only when stored: an fp32 sum over a
    // long context drops bits the softmax denominator and the weighted sum of v
    // are built from, and f64 makes that error smaller than the f32 rounding
    // that the caller sees anyway.
    const scale = 1.0 / @sqrt(@as(f64, @floatFromInt(dim)));

    var out = try Tensor.init(allocator, q.rows, cfg.n_heads * dim);
    errdefer out.deinit();

    var scores = try allocator.alloc(f64, q.rows);
    defer allocator.free(scores);

    for (0..cfg.n_heads) |h| {
        const kv = h / group;
        for (0..q.rows) |t| {
            const q_head = q.rowConst(t)[h * dim ..][0..dim];
            var row_max: f64 = -std.math.inf(f64);

            for (0..t + 1) |s| {
                const k_head = k.rowConst(s)[kv * dim ..][0..dim];
                var dot: f64 = 0;
                for (q_head, k_head) |a, b| dot += @as(f64, a) * @as(f64, b);
                const score = dot * scale;
                scores[s] = score;
                row_max = @max(row_max, score);
            }

            // The row max cancels against the denominator, so subtracting it
            // leaves the softmax unchanged while keeping every exp in range.
            var denom: f64 = 0;
            for (0..t + 1) |s| {
                scores[s] = @exp(scores[s] - row_max);
                denom += scores[s];
            }
            for (0..t + 1) |s| scores[s] /= denom;

            const out_row = out.row(t);
            for (0..dim) |j| {
                var acc: f64 = 0;
                for (0..t + 1) |s| {
                    const v_head = v.rowConst(s)[kv * dim ..][0..dim];
                    acc += scores[s] * @as(f64, v_head[j]);
                }
                out_row[h * dim + j] = @floatCast(acc);
            }
        }
    }
    return out;
}
