//! Grouped-query causal attention with a single-pass softmax.

const std = @import("std");
const model = @import("model.zig");
const tensor = @import("tensor.zig");

// Public because the tools in `src/cuda` cannot reach `tensor.zig`: a module
// rooted there is handed this one whole, and this module's own relative import
// already owns that file. A caller holding a value from `forward` needs to name
// its type.
pub const Tensor = tensor.Tensor;

pub const Config = struct {
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
};

/// The head geometry `model.defaultConfig` implies. Exposed rather than left to
/// each caller to assemble, because the tools in `src/cuda` cannot: a module
/// rooted in `src/cuda` is handed this one whole, and this module's own relative
/// imports already own `model.zig` and `tensor.zig`, so passing either of those
/// as its own module is an ownership conflict the compiler rejects.
pub fn defaultConfig() Config {
    const m = model.defaultConfig();
    return .{ .n_heads = m.n_heads, .n_kv_heads = m.n_kv_heads, .head_dim = m.head_dim };
}

/// Causal grouped-query attention over q, k and v that already carry their
/// position embedding. q is [T, n_heads * head_dim], k and v are
/// [T, n_kv_heads * head_dim], and the result is [T, n_heads * head_dim] with
/// head h in columns h * head_dim .., so every head slice is contiguous.
///
/// Query head h reads kv head h / (n_heads / n_kv_heads) and only keys at or
/// before its own position.
pub fn forward(allocator: std.mem.Allocator, q: Tensor, k: Tensor, v: Tensor, cfg: Config) !Tensor {
    return forwardWith(allocator, q, k, v, cfg, null, 0);
}

/// `forward`, with the softmax matrix handed to `sink`. Note the narrowing: the
/// sink receives `f32`, because the sink is a `Tensor`, while the context this
/// function returns is accumulated from the same probabilities still held in
/// `f64`. A caller that wanted the `f64` values cannot get them from here, and
/// the two are not the same number.
///
/// It lives here rather than in the caller because the caller cannot see it:
/// each probability is computed into a `q.rows` f64 scratch row and reduced into
/// the context one line later, so without this the matrix exists only as that
/// row. `layer` is the sink's layer index and is unused when `sink` is null.
///
/// The one intermediate whose size is quadratic in the context, so it is
/// allocated only when there is a sink to hand it to, and `forward` above is
/// this with a null sink. Note who that does and does not spare: the autograd
/// pass, the training run and the bench all reach attention through
/// `model.forwardWith` with a *live* sink, so they do materialise it. What the
/// null path buys is a caller that wants no intermediate at all, and the cost
/// the sink does pay is T * n_heads * T f32 -- 1 MiB at the shipped T=256 and
/// four heads, freed before this returns, so one layer's worth of peak and not
/// the model's. Measured, the step time did not move.
pub fn forwardWith(
    allocator: std.mem.Allocator,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cfg: Config,
    sink: ?*model.Sink,
    layer: usize,
) !Tensor {
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

    // [T * n_heads, T]: row h * T + t is the softmax over the causal prefix of
    // query t in head h. `Tensor.init` zeroes it, and the keys a causal query
    // does not attend to are exactly zero, so only the prefix is written below
    // and the upper triangle needs no pass of its own.
    var probs: ?Tensor = null;
    defer if (probs) |*p| p.deinit();
    if (sink != null) probs = try Tensor.init(allocator, q.rows * cfg.n_heads, q.rows);

    // Deliberately not zeroed, and the reason is worth writing down because the
    // safety is invisible from any one call site: this buffer is reused for every
    // position, and only `0..t + 1` is written for position `t`. Nothing has ever
    // read past `t + 1` because three separate bounds agree on it — the score dot,
    // the exp loop and the `v` sum. A fourth loop that walked the full row would
    // read last position's values and nothing would say so. Zeroing it per
    // position would cost a memset in the innermost loop of the forward pass, so
    // the invariant is documented instead of enforced.
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

            if (probs) |*p| {
                const row = p.row(h * q.rows + t);
                for (0..t + 1) |s| row[s] = @floatCast(scores[s]);
            }

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
    if (sink) |s| s.put(s, .attn_probs, layer, probs.?);
    return out;
}
