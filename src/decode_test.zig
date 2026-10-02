const std = @import("std");
const decode = @import("decode.zig");
const model = @import("model.zig");
const kv_cache = @import("kv_cache.zig");
const tensor = @import("tensor.zig");
const device = @import("cuda/device.zig");

/// 1 layer, d_model 128, vocab 16, ctx 32: a decode step is the whole block and
/// nothing else. Four heads over two kv heads of 32 is the shipped model's GQA
/// ratio and width.
///
/// The width is `zt_attn_dim_ok`'s floor rather than a choice, and it is not
/// about this file being tidy. `device.Attn.init` calls that predicate before it
/// allocates anything, so a holder built at `tiny`'s old head_dim of 4 is refused
/// and `generate` under `model.cuda_attn` answers `error.ConfigRefused` to every
/// call below -- including the three that are checking something else entirely.
/// The CPU path builds no holder, so a narrower width would still work there; it is
/// the GPU build that cannot be asked to decode a width its kernel refuses.
const tiny = model.Config{
    .n_layers = 1,
    .n_heads = 4,
    .n_kv_heads = 2,
    .head_dim = 32,
    .n_ctx = 32,
    .vocab_size = 16,
    .ffn_mult = 1,
};

/// A deterministic value in [-1, 1) for fill position `i`, hashed from the index
/// rather than walked from an accumulator. Bounded is the requirement: this sweep
/// writes about five thousand values, and `x * 1.7 + 0.13` in f32 is past infinity
/// long before that, and two infinities compared against each other would fail the
/// gate below for a reason that has nothing to do with attention. Hashed, so the
/// value at an index does not depend on how many were drawn before it.
fn fill(i: usize) f32 {
    var x: u32 = @truncate(i *% 2654435761);
    x ^= x >> 15;
    x *%= 2246822519;
    x ^= x >> 13;
    // The top 24 bits, which convert to f32 exactly.
    return @as(f32, @floatFromInt(x >> 8)) / 8388608.0 - 1.0;
}

/// Argmax with the same tie rule `decode.zig` uses, written out rather than imported:
/// `argmax` there is private on purpose, and a second implementation of the rule in
/// the test would be the thing under test if this one were the rule.
fn argmax(logits: []const f32) usize {
    var best: usize = 0;
    for (logits, 0..) |l, i| {
        if (l > logits[best]) best = i;
    }
    return best;
}

test "a cached decode step agrees with a full forward pass over the prompt" {
    // THE TEST THIS FILE EXISTS FOR. `model.forwardWith` rotates every row at
    // position 0, reads the whole sequence at once and attends each position to its
    // own causal prefix. `decode.generate` rotates one token at its absolute `pos`
    // and reads the same keys back out of a cache, one row per step. If either the
    // rotation or the key count is wrong the step still returns a plausible [1, vocab]
    // tensor, and the only thing in the tree that can see the difference is a token
    // this prompt's forward pass would not have produced.
    var p = try model.initParams(std.testing.allocator, tiny, 7);
    defer p.deinit();
    const prompt = [_]u32{ 5, 0, 15, 2 };

    var full = try model.forwardWith(std.testing.allocator, p, tiny, &prompt, null);
    defer full.deinit();
    const want = argmax(full.rowConst(prompt.len - 1));

    const got = try decode.generate(std.testing.allocator, p, tiny, &prompt, 1);
    defer std.testing.allocator.free(got);

    try std.testing.expectEqual(@as(u32, @intCast(want)), got[0]);
}

test "the second generated token matches a full forward over the grown prompt" {
    // The test above runs one step, at `prompt.len - 1`, and cannot see a `pos` that
    // fails to advance: nothing past that position is ever fed. This one generates
    // two, so `step` runs again one position further on with the previously sampled
    // token in hand, which is the case a stale `pos` gets wrong -- the rotated key
    // lands in the wrong cache row and the second token disagrees with the forward
    // pass over prompt ++ got[0].
    var p = try model.initParams(std.testing.allocator, tiny, 7);
    defer p.deinit();
    const prompt = [_]u32{ 5, 0, 15, 2 };

    const got = try decode.generate(std.testing.allocator, p, tiny, &prompt, 2);
    defer std.testing.allocator.free(got);
    try std.testing.expectEqual(@as(usize, 2), got.len);

    var grown: [5]u32 = undefined;
    @memcpy(grown[0..prompt.len], &prompt);
    grown[prompt.len] = got[0];
    var full = try model.forwardWith(std.testing.allocator, p, tiny, &grown, null);
    defer full.deinit();
    const want = argmax(full.rowConst(grown.len - 1));

    try std.testing.expectEqual(@as(u32, @intCast(want)), got[1]);
}

test "generate refuses what it cannot do instead of guessing" {
    var p = try model.initParams(std.testing.allocator, tiny, 7);
    defer p.deinit();

    // An empty prompt has no last position to read the first generated token from,
    // and this repository has no BOS id to invent a first token from.
    try std.testing.expectError(
        error.EmptyPrompt,
        decode.generate(std.testing.allocator, p, tiny, &.{}, 1),
    );
    // The cache would refuse the same run one step later with the same error, so
    // this only pins that it arrives before the prefill rather than during it.
    try std.testing.expectError(
        error.SequenceTooLong,
        decode.generate(std.testing.allocator, p, tiny, &.{ 1, 2, 3 }, tiny.n_ctx),
    );
    // An id `tok_embed` cannot be indexed with. It arrives as u32 and the check is
    // made after widening, so a corrupt id cannot wrap down into the valid range.
    try std.testing.expectError(
        error.TokenOutOfRange,
        decode.generate(std.testing.allocator, p, tiny, &.{ 0, tiny.vocab_size }, 1),
    );

    // And the good call still works afterwards, so none of the three refusals left
    // the parameters or the allocator in a state the next one trips over -- which is
    // what a leak or a half-built cache would show up as.
    const ok = try decode.generate(std.testing.allocator, p, tiny, &.{1}, 1);
    defer std.testing.allocator.free(ok);
    try std.testing.expectEqual(@as(usize, 1), ok.len);
    try std.testing.expect(ok[0] < tiny.vocab_size);
}

// THE GATE ON THE DECODE SEAM, and it is a separate gate from `src/train.zig`'s
// because the two arms hand the kernel three different numbers: the training arm
// passes `q_offset` 0 and `t` the window, and this one passes the token's absolute
// position and `t` of 1 over a cache the CPU loop never sees as a matrix.
//
// WHERE IT RUNS, and the CPU run is not asked to report a pass it did not earn.
// Every build in the graph that links no CUDA object takes the else arm and returns
// `error.SkipZigTest`, which the runner prints as `... SKIP` on its own line and
// does NOT count as passed -- so `zig build test` reports 217 of 219 with two
// skipped, and neither skip is a silent pass.
//
// `zig build cuda-attn-check` is the one step that links `src/cuda/attn_kernels.cu`,
// and it runs THIS test as well as the training gate: `build.zig` roots a second test
// artifact at this file and makes the step depend on both, rather than rooting the
// step at `src/tests.zig` and re-running all 219 against a CUDA-linked binary. Both
// halves fail while `src/model.zig:cuda_attn` is false, so the step cannot pass
// without having measured something.
//
// An earlier version of this comment said the step was rooted at `src/train.zig` and
// therefore did not carry this test, and named a hand-run command instead. That was
// true when written and is why the step now carries it; a gate whose only invocation
// is a paragraph is a gate nobody runs, which this repository has now had to learn
// twice. The three tests above are the other half of the coverage: under `cuda_attn`
// true they grade `generate` against `model.forwardWith` at every position, which is
// what catches a wrong `q_offset` or `n_keys` derivation end to end. The pair
// together is the seam; neither half is the whole of it.
test "attnStep refuses a shape it cannot compute instead of answering zeros" {
    const gpa = std.testing.allocator;
    const dim = tiny.head_dim;
    const q_cols = tiny.n_heads * dim;
    const kv_cols = tiny.n_kv_heads * dim;

    var q = try tensor.Tensor.init(gpa, 1, q_cols);
    defer q.deinit();
    for (q.data, 0..) |*v, i| v.* = fill(i);
    var l = try kv_cache.Layer.init(gpa, tiny.n_ctx, kv_cols);
    defer l.deinit();

    // An empty prefix. Every loop in `attnStep` is bounded by `n_keys`, so zero keys
    // skips all of them: `denom` stays 0, no division runs, and `out` reads back the
    // zeros `Tensor.init` gave it. A plausible zero context, no error -- which is the
    // defect. The target column here is filled and non-zero, so nothing else could be
    // mistaken for this.
    try std.testing.expectError(
        error.DimensionMismatch,
        decode.attnStep(gpa, q, &l, tiny),
    );

    // The two refusals `cudaAttnStep` makes, mirrored. Both arms must refuse the same
    // shapes, or the CPU twin answers something the GPU arm declined to and the gate
    // between them compares a refusal with a number.
    var bad_heads = tiny;
    bad_heads.n_kv_heads = 0;
    try std.testing.expectError(error.InvalidHeadConfig, decode.attnStep(gpa, q, &l, bad_heads));

    var odd_heads = tiny;
    odd_heads.n_heads = 3; // 3 over 2 kv heads is not a group
    try std.testing.expectError(error.InvalidHeadConfig, decode.attnStep(gpa, q, &l, odd_heads));

    var narrow_q = try tensor.Tensor.init(gpa, 1, q_cols - 1);
    defer narrow_q.deinit();
    try std.testing.expectError(
        error.DimensionMismatch,
        decode.attnStep(gpa, narrow_q, &l, tiny),
    );

    // And one filled key is enough to answer for real, so the refusals above are not
    // the function declining everything. `q`'s first kv-head worth of columns is the
    // key row and the next is the value row; the two are independent slices, so the
    // same memory cannot stand for both.
    //
    // The output is checked for being NON-ZERO, not merely finite. An all-zero `out`
    // is what the empty-prefix case above would have produced had it not been
    // refused, and an `isFinite` assertion is satisfied by it -- so a finiteness
    // check alone would let the original defect back in through this arm.
    try l.append(q.data[0..kv_cols], q.data[kv_cols .. 2 * kv_cols]);
    var out = try decode.attnStep(gpa, q, &l, tiny);
    defer out.deinit();
    try std.testing.expectEqual(@as(usize, 1), out.rows);
    try std.testing.expectEqual(q_cols, out.cols);
    var any_nonzero = false;
    for (out.data) |v| {
        try std.testing.expect(std.math.isFinite(v));
        if (v != 0) any_nonzero = true;
    }
    try std.testing.expect(any_nonzero);
}

test "the CUDA decode step agrees with the CPU one over a filling cache" {
    if (comptime model.cuda_attn) {
        const gpa = std.testing.allocator;
        const dim = tiny.head_dim;
        const q_cols = tiny.n_heads * dim;
        const kv_cols = tiny.n_kv_heads * dim;

        // The holder `generate` builds: sized at `n_ctx`, so `k` and `v` hold every
        // row the sweep will write while the launch is told one query row.
        var holder = try device.Attn.init(tiny.n_ctx, tiny.n_heads, tiny.n_kv_heads, dim);
        defer holder.deinit();

        var q = try tensor.Tensor.init(gpa, 1, q_cols);
        defer q.deinit();
        var l = try kv_cache.Layer.init(gpa, tiny.n_ctx, kv_cols);
        defer l.deinit();

        // EVERY position, and not one position. `cudaAttnStep` derives `q_offset`
        // from the filled prefix rather than taking a position, so a `q_offset`
        // left at 0 -- or an `n_keys` that counted the whole holder rather than
        // what has been appended -- agrees with the CPU at position 0 and disagrees
        // from position 1 on. One position would grade the arithmetic and miss the
        // wiring, and the wiring is the half of this that is new.
        var pos: usize = 0;
        while (pos < tiny.n_ctx) : (pos += 1) {
            // Distinct index ranges per buffer AND per row, and the per-row part is the
            // one that matters. Every key row and every value row identical would make the
            // answer independent of `n_keys` and of `q_offset` alike -- the softmax
            // over a prefix of identical rows is the mean of identical rows, which is
            // the row -- so a gate on that data passes with the offset set to 0. It
            // did, once, and the only reason it showed up is the negative control
            // that `src/cuda/run-attn.sh` runs and this file used to not.
            const row_base = q_cols + pos * 2 * kv_cols;
            for (q.data, pos * q_cols..) |*e, i| e.* = fill(row_base + i);
            var k_row: [kv_cols]f32 = undefined;
            var v_row: [kv_cols]f32 = undefined;
            for (0..kv_cols) |i| {
                k_row[i] = fill(row_base + i);
                v_row[i] = fill(row_base + kv_cols + i);
            }
            // Rotated in the real world by `step` before this; the kernel does not
            // care, and a rotation here would be `rope.forward`'s arithmetic copied
            // into a test for no gain.
            try l.append(&k_row, &v_row);

            // THE CPU SIDE, first, so the reference exists before anything reaches
            // the device. `decode.attnStep` IS the loop `step` runs when `cuda_attn`
            // is false -- the implementation the CUDA arm is a claim about, not a
            // copy of it, for `src/train.zig`'s reason: a comparison against a
            // re-typed loop only proves the copy is stable.
            var want = try decode.attnStep(gpa, q, &l, tiny);
            defer want.deinit();

            var got = try decode.cudaAttnStep(gpa, &holder, q, &l, tiny);
            defer got.deinit();
            try std.testing.expectEqual(want.data.len, got.data.len);

            // RELATIVE, and the reason is this repository's own rather than a
            // preference: `sh src/cuda/run-attn.sh`'s `ATTN_TOL` is an ABSOLUTE
            // bound calibrated on that script's own uniform +/-0.5 inputs, so it
            // says nothing about the magnitudes a decode step actually produces.
            // `max|a-b| <= tol * max|b|` means the same thing at any scale, which is
            // the only property a forward comparison needs.
            var worst: f32 = 0;
            var scale: f32 = 0;
            for (want.data, got.data) |w, h| {
                worst = @max(worst, @abs(w - h));
                scale = @max(scale, @abs(w));
            }
            // The reference has to carry a magnitude at all. `scale == 0` would turn
            // the gate into an equality test on a pair of zero tensors, which
            // passes, and a comparison that can pass on nothing is the defect this
            // is written to stop repeating.
            if (!(scale > 0)) return error.EmptyReference;
            if (!(worst <= cuda_rel_tol * scale)) {
                std.debug.print(
                    "\nCUDA decode step at pos {d} differs from the CPU one by {e}" ++
                        " against a reference of at most {e},\nwhich is {e} times the" ++
                        " relative gate of {e}.\n",
                    .{ pos, worst, scale, worst / scale, cuda_rel_tol },
                );
                return error.AttentionMismatch;
            }
        }
    } else {
        return error.SkipZigTest;
    }
}

/// The relative gate above, and it is `src/train.zig`'s `attn_rel_tol` restated
/// rather than imported: that one is private to a file this test does not own, and a
/// gate that could not be reached would have to invent one.
///
/// 1e-4, for the same reason and with the same arithmetic. The defects this seam can
/// carry -- a `q_offset` left at 0, an `n_keys` past the filled prefix, a collapsed
/// GQA group, a missing `1/sqrt(head_dim)` -- each move the answer by a fraction of
/// itself rather than by a part in a million of it, and the sweep above is what
/// turns the first two from a pass at position 0 into a failure.
const cuda_rel_tol: f32 = 1e-4;
