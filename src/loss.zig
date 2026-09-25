//! Cross-entropy of predicted logits against a token id array.

const std = @import("std");
const Tensor = @import("tensor.zig").Tensor;

/// Mean cross-entropy of `logits`, one row per position, against `targets`.
///
/// loss_t = logsumexp(logits[t]) - logits[t][targets[t]], averaged over rows.
pub fn forward(logits: Tensor, targets: []const u32) !f64 {
    const t_count = logits.rows;
    if (t_count == 0) return error.EmptyBatch;
    if (targets.len != t_count) return error.TargetCountMismatch;
    const v_count = logits.cols;

    // The logits are f32 because the tensor is f32, and the upcast to f64 is
    // deliberate: the f32 rounding already happened upstream, so reducing and
    // averaging at f64 costs nothing and keeps the reported loss from losing
    // another 24 bits on the way out. Nothing here narrows back to f32.
    var total: f64 = 0;
    for (targets, 0..) |target, i| {
        const row = logits.rowConst(i);
        // targets come from corpus data, so this is a trust boundary. Widening
        // u32 to usize is lossless on every Zig target, which is what keeps a
        // corrupt id from wrapping down into the valid range before the check.
        const t: usize = target;
        if (t >= v_count) return error.TargetOutOfRange;

        // logsumexp rather than a softmax vector: the row max is pulled out
        // from under the exponent, so exp() never overflows and no V-element
        // probability buffer is needed. Forming softmax and taking -ln of one
        // entry instead reports an infinite loss as soon as an entry underflows
        // to zero probability, which a single large logit is enough to cause.
        // v_count is at least 1 here, because the bounds check above rejects
        // every target when the vocabulary is empty.
        var max = @as(f64, row[0]);
        for (row[1..]) |z| max = @max(max, @as(f64, z));
        var sum: f64 = 0;
        for (row) |z| sum += @exp(@as(f64, z) - max);
        total += max + std.math.log(f64, std.math.e, sum) - @as(f64, row[t]);
    }

    // Fixed row order, one f64 accumulator, no reassociation: the same logits
    // and targets give the same bits every run, which Phase 2 training needs.
    return total / @as(f64, @floatFromInt(t_count));
}
