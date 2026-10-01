//! Tests for the finite-difference check itself. `autograd_test.zig` grades
//! gradients with this module; this file tests the parts of it that are data
//! rather than a verdict, so a defect in the reporting cannot hide behind a
//! verdict that happens to be right.
//!
//! Nothing here reads the process's streams. `report` is handed a writer and
//! writes to it, so what it writes is the `Report` it was given, and the
//! `Report` is what these tests assert on.
const std = @import("std");
const model = @import("model.zig");
const autograd = @import("autograd.zig");
const gradcheck = @import("gradcheck.zig");

// The shapes and the parameter draw come from the autograd suite, which owns
// them: one config for one layer, one config for two, and the draw that leaves
// every norm weight at one so the model is not a constant logit row. A second
// copy of either would be a second thing to keep right.
const fixtures = @import("autograd_test.zig");

const tiny = fixtures.tiny;
const two_layers = model.Config{
    .n_layers = 2,
    .n_heads = 2,
    .n_kv_heads = 2,
    .head_dim = 4,
    .n_ctx = 32,
    .vocab_size = 16,
    .ffn_mult = 1,
};

/// Grouped-query attention at `n_kv_heads = 1`: every query head shares one kv
/// head, so `dk` and `dv` each sum four terms where an ungrouped head sums one.
/// That is the accumulation the budget has to survive, and it is the fixture
/// that measures the budget rather than inheriting a verdict: `mismatch == null`
/// only says the check did not fail, while `headroom` says how much room was
/// left. The wider shape is `autograd_test.zig`'s, reused rather than reinvented,
/// so the two suites grade the same model.
const gqa = model.Config{
    .n_layers = 1,
    .n_heads = 4,
    .n_kv_heads = 1,
    .head_dim = 4,
    .n_ctx = 32,
    .vocab_size = 32,
    .ffn_mult = 1,
};

const tok: []const u32 = &.{ 3, 1, 4, 0 };
const tgt: []const u32 = &.{ 1, 4, 0, 2 };

fn liveParams(allocator: std.mem.Allocator, cfg: model.Config) !model.Params {
    return fixtures.liveParams(allocator, cfg);
}

fn grads(allocator: std.mem.Allocator, cfg: model.Config, p: model.Params) !autograd.Grads {
    var g = try autograd.zeroGrads(allocator, p);
    errdefer g.deinit();
    var logits = try model.forward(allocator, p, cfg, tok);
    defer logits.deinit();
    var dl = try autograd.dLossDLogits(allocator, logits, tgt);
    defer dl.deinit();
    try autograd.backward(allocator, p, &g, cfg, tok, dl);
    return g;
}

test "gradcheck: compare returns the whole sweep as data" {
    var p = try liveParams(std.testing.allocator, tiny);
    defer p.deinit();
    var g = try grads(std.testing.allocator, tiny, p);
    defer g.deinit();

    const r = try gradcheck.compare(std.testing.allocator, tiny, p, tok, tgt, &g);
    defer r.deinit();

    // One row per parameter tensor, in the order the failure line would name
    // them: the embedding, every field of every layer, then the final norm.
    try std.testing.expectEqual(2 + 9 * tiny.n_layers, r.groups.len);
    try std.testing.expectEqualStrings("tok_embed", r.groups[0].field);
    try std.testing.expectEqual(@as(?usize, null), r.groups[0].layer);
    try std.testing.expectEqualStrings("final_norm", r.groups[r.groups.len - 1].field);
    const first_layer = [_][]const u8{
        "attn_norm", "wq", "wk", "wv", "wo", "mlp_norm", "w_gate", "w_up", "w_down",
    };
    for (first_layer, 0..) |field, i| {
        try std.testing.expectEqualStrings(field, r.groups[1 + i].field);
        try std.testing.expectEqual(@as(?usize, 0), r.groups[1 + i].layer);
    }

    // The budget is a real number and a discriminating one: above nothing, and
    // a small fraction of the largest gradient in the model, or it would be a
    // budget that cannot fail. `floor < gmax` alone is far too loose to do that
    // job: the measured floor is 1.1e-4 against a gmax of 0.88, so the bound
    // permits a budget nearly four thousand times looser than the real one, and
    // the corruption test below perturbs by 1.0, which such a budget would still
    // catch. A tenth of a percent is the bound `src/README.md` already states in
    // prose, so this is the assertion that keeps that sentence true. It is the
    // complement to the headroom assertion below: this one forbids a budget so
    // loose it cannot fail, that one forbids one so tight a correct gradient
    // trips over it.
    try std.testing.expect(r.floor > 0);
    var gmax: f64 = 0;
    for (g.tok_embed.data) |v| gmax = @max(gmax, @abs(@as(f64, @floatCast(v))));
    // Only the tight bound is asserted. `floor < gmax` was here too and was
    // implied by the next line for every `gmax > 0`, so it could never be the
    // assertion that fails and only read like coverage.
    try std.testing.expect(r.floor < gmax * 0.001);

    // A correct gradient is accepted, and the per tensor worst is a real
    // measurement with an index that exists.
    try std.testing.expectEqual(@as(?gradcheck.Mismatch, null), r.mismatch);
    for (r.groups) |group| {
        try std.testing.expect(group.worst >= 0);
        try std.testing.expect(group.worst_at < group.param.data.len);
    }
}

test "gradcheck: compare finds the first element that does not fit" {
    // The first one, not the only one: a sweep with one wrong element still
    // reports it, and reports it as the row and index it is.
    var p = try liveParams(std.testing.allocator, two_layers);
    defer p.deinit();
    var g = try grads(std.testing.allocator, two_layers, p);
    defer g.deinit();

    // layers[0].wq, element 9: a weight row that feeds a real head, so its
    // gradient is not one of the zeros a zeroed model produces.
    g.layers[0].wq.data[9] = 1.0;

    const r = try gradcheck.compare(std.testing.allocator, two_layers, p, tok, tgt, &g);
    defer r.deinit();

    const m = r.mismatch orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(@as(?usize, 0), m.layer);
    try std.testing.expectEqualStrings("wq", m.field);
    try std.testing.expectEqual(@as(usize, 9), m.index);
    // The budget is per element, so this element's is at or under the largest
    // the Report carries, and the gap is over its own rather than merely near it.
    try std.testing.expect(m.budget <= r.floor);
    try std.testing.expect(m.diff > m.budget);
}

test "gradcheck: the budget bounds a correct gradient on grouped-query attention" {
    // The load-bearing assertion of the whole module, and the one the old
    // `sqrt(vocab / T)` budget fails. It failed on a correct gradient and
    // nothing else: the grouped path sums four dk terms per kv head, the finite
    // difference carries that as noise, and `tok_embed[30]` read 5.6% over budget
    // at vocab 256 while at vocab 16 the same formula was a thousand times looser
    // than the discrepancy it was describing. A check that reports a correct
    // gradient as wrong is worse than no check, because a maintainer learns to
    // ignore the only instrument there is.
    var p = try liveParams(std.testing.allocator, gqa);
    defer p.deinit();
    var g = try grads(std.testing.allocator, gqa, p);
    defer g.deinit();

    const r = try gradcheck.compare(std.testing.allocator, gqa, p, tok, tgt, &g);
    defer r.deinit();

    // Two bounds, not one: `mismatch == null` is also true of a budget that has
    // been multiplied by a hundred, and the corruption test in
    // `autograd_test.zig` is the only thing that would notice. `headroom` is the
    // margin itself, so a floor on it says the formula is a bound with room to
    // spare rather than a constant that happened to clear this fixture.
    //
    // The headroom is asserted as a BAND around a measured value, and the band
    // is the fix. The assertion used to be a one-sided floor, `headroom > 1.5`,
    // and a floor cannot catch a budget that has been loosened: weakening
    // multiplies the budget, which DIVIDES `diff / budget`, which RAISES the
    // headroom. The sweep at k = 12 measures 2.6184, so a 1.5x weakening reports
    // 3.93 and sails under a 1.5 floor, and raising the floor to 3.93 to catch
    // it would have made every correct gradient a failure. The gate was pointed
    // the wrong way: it could only ever catch a budget that had been tightened.
    //
    // So both sides are asserted, and the reference is a measurement. This
    // fixture is PINNED — `liveParams` draws from the literal 0xbeef — so the
    // sweep is the same sweep every run: 2.6184 in ReleaseFast, 2.6442 in Debug.
    // The tolerance is 0.35, about 13%, which is an order of magnitude above the
    // 1% spread between the two configurations and an order of magnitude below
    // the 51% a 1.5x weakening moves it by.
    //
    // What that buys, as arithmetic. A budget multiplied by f reports 2.6184 * f,
    // so the band catches any change beyond roughly 0.87x to 1.13x in either
    // direction. The 1.5x weakening that 1.5 could not see reports 3.93 and is
    // caught. A budget inflated ten times reports 26 and is caught. The broken
    // `sqrt(vocab / T)` budget measures 0.88 at this vocab and 0.62 at
    // `vocab 16`, and both are caught, as is any budget tightened toward 1.0,
    // which is the failure the old floor was the only thing watching for.
    try std.testing.expectApproxEqAbs(2.62, r.headroom, 0.35);

    // And the budget is still small enough to catch a wrong gradient. Corrupting
    // `wk` by twice the budget has to be reported, or the headroom above is only
    // slack in one direction.
    const before = g.layers[0].wk.data[5];
    g.layers[0].wk.data[5] = before + @as(f32, @floatCast(2.0 * r.floor));
    const bad = try gradcheck.compare(std.testing.allocator, gqa, p, tok, tgt, &g);
    defer bad.deinit();
    const m = bad.mismatch orelse return error.TestUnexpectedResult;
    try std.testing.expectEqualStrings("wk", m.field);
    try std.testing.expect(m.diff > m.budget);
}

test "gradcheck: line renders one mismatch" {
    // A formatter is worth a test that pins the whole line: a reader who has
    // only the line has to be able to re-derive the failure from it.
    const in_layer = gradcheck.Mismatch{
        .layer = 0,
        .field = "wq",
        .index = 9,
        .analytic = 0.008003430,
        .numeric = 0.008057021,
        .diff = 0.000053591,
        .budget = 0.000096393,
    };
    const text = try gradcheck.line(std.testing.allocator, in_layer);
    defer std.testing.allocator.free(text);
    try std.testing.expectEqualStrings(
        "gradient mismatch at layers[0].wq[9]: analytic 0.008003430  numeric 0.008057021  diff 0.000053591  budget 0.000096393",
        text,
    );

    // The two tensors outside the layer stack name themselves without the
    // `layers[N]` prefix, because there is no layer to index.
    const in_stack = gradcheck.Mismatch{
        .layer = null,
        .field = "tok_embed",
        .index = 37,
        .analytic = -0.5,
        .numeric = 0.5,
        .diff = 1.0,
        .budget = 0.25,
    };
    const other = try gradcheck.line(std.testing.allocator, in_stack);
    defer std.testing.allocator.free(other);
    try std.testing.expectEqualStrings(
        "gradient mismatch at tok_embed[37]: analytic -0.500000000  numeric 0.500000000  diff 1.000000000  budget 0.250000000",
        other,
    );
}

test "gradcheck: report writes the table it is handed" {
    // `report` is a printer, so what matters is the bytes, and the bytes are
    // only reachable if the caller chooses the destination. The sweep is the
    // shape that is hardest for it: a model whose norms were zeroed, so the
    // gradients it reads are all exactly zero and a relative figure has no
    // scale of its own.
    var p = try liveParams(std.testing.allocator, tiny);
    defer p.deinit();
    for (p.layers) |*l| {
        l.attn_norm.fill(0);
        l.mlp_norm.fill(0);
    }
    p.final_norm.fill(0);

    var g = try grads(std.testing.allocator, tiny, p);
    defer g.deinit();
    const r = try gradcheck.compare(std.testing.allocator, tiny, p, tok, tgt, &g);
    defer r.deinit();
    try std.testing.expectEqual(@as(?gradcheck.Mismatch, null), r.mismatch);
    for (r.groups) |group| {
        if (std.mem.eql(u8, group.field, "wq")) try std.testing.expectEqual(0.0, group.worst);
    }

    var buf: [4096]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    try gradcheck.report(&w, r);

    // One row per group, in the order the groups are in, and no more. Calling
    // the printer would have proved none of this; only the bytes do.
    var rows = std.mem.splitScalar(u8, std.mem.trim(u8, w.buffered(), "\n"), '\n');
    for (r.groups, 0..) |group, i| {
        const row = rows.next() orelse return error.TestUnexpectedResult;
        // The label, which is what makes the row name a parameter rather than a
        // position, and the numbers beside it: the worst figure read for that
        // parameter, the index it was read at, the step, and the floor.
        try std.testing.expect(std.mem.indexOf(u8, row, group.field) != null);
        try std.testing.expect(renders(row, group.worst, "{e:.3}"));
        try std.testing.expect(std.mem.indexOf(u8, row, "h 1e-3") != null);
        try std.testing.expect(renders(row, r.floor, "{e:.3}"));
        // The headroom, in the column the table gives it: a passing sweep that
        // says nothing about how much room it had left cannot be told apart from
        // one whose budget is too loose to catch anything.
        try std.testing.expect(renders(row, r.headroom, "{e:.2}"));
        try std.testing.expect(std.mem.indexOf(u8, row, "headroom") != null);
        // The index, read from the column it belongs to rather than from the row
        // at large, since every other column carries digits too.
        const at = std.mem.indexOf(u8, row, "at ") orelse return error.TestUnexpectedResult;
        try std.testing.expect(renders(row[at + 3 ..], @as(f64, @floatFromInt(group.worst_at)), "{d}"));
        // A layer row is indexed, so two rows sharing a field are still two
        // parameters, and the first one says which.
        if (i == 1) try std.testing.expect(std.mem.startsWith(u8, row, "gradcheck layers[0].attn_norm"));
    }
    try std.testing.expect(rows.next() == null);
}

/// Whether `row` carries `value` rendered the way the table renders it.
fn renders(row: []const u8, value: anytype, comptime fmt: []const u8) bool {
    var num: [64]u8 = undefined;
    const text = std.fmt.bufPrint(&num, fmt, .{value}) catch return false;
    return std.mem.indexOf(u8, row, text) != null;
}

test "gradcheck: checkAll passes on a correct gradient and restores the parameters" {
    // The wrapper runs backward and then grades with it, and it hands back the
    // caller's parameters. Both are worth asserting at the seam, because both
    // are promises the wrapper makes and neither is visible in what it returns.
    //
    // Its failure branch is not here: `checkAll` computes the gradient itself,
    // so no caller can hand it a wrong one, and a genuinely wrong gradient is
    // the thing this module exists to detect rather than a thing a test can
    // stage. The branch is the two functions below it, which are covered.
    var p = try liveParams(std.testing.allocator, tiny);
    defer p.deinit();
    const before = try std.testing.allocator.dupe(f32, p.tok_embed.data);
    defer std.testing.allocator.free(before);

    try gradcheck.checkAll(std.testing.allocator, tiny, p, tok, tgt);

    try std.testing.expectEqualSlices(f32, before, p.tok_embed.data);
}
