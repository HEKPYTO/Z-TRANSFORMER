const std = @import("std");
const data = @import("data.zig");

/// The first input of a batch is the first token of its window, so over a
/// strictly ascending stream it names the window the batch came from. Lets a
/// test pin literals per window without depending on the shuffle order.
fn windowOf(batch: data.Batch, stride: usize) usize {
    return batch.inputs[0] / stride;
}

test "split puts the last five percent of a 1000 token stream in val" {
    var tokens: [1000]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i);

    const corpus = try data.split(&tokens, 0.05);

    try std.testing.expectEqual(@as(usize, 950), corpus.train.len);
    try std.testing.expectEqual(@as(usize, 50), corpus.val.len);
    try std.testing.expectEqual(@as(u32, 0), corpus.train[0]);
    try std.testing.expectEqual(@as(u32, 949), corpus.train[949]);
    try std.testing.expectEqual(@as(u32, 950), corpus.val[0]);
    try std.testing.expectEqual(@as(u32, 999), corpus.val[49]);
    // No overlap and no gap: val begins on the very address train stops at.
    try std.testing.expectEqual(corpus.train.ptr + corpus.train.len, corpus.val.ptr);
}

test "split at zero and at one empties the other half" {
    var tokens: [10]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i);

    const all_train = try data.split(&tokens, 0.0);
    try std.testing.expectEqual(@as(usize, 10), all_train.train.len);
    try std.testing.expectEqual(@as(usize, 0), all_train.val.len);

    const all_val = try data.split(&tokens, 1.0);
    try std.testing.expectEqual(@as(usize, 0), all_val.train.len);
    try std.testing.expectEqual(@as(usize, 10), all_val.val.len);
    try std.testing.expectEqual(@as(u32, 0), all_val.val[0]);
    try std.testing.expectEqual(@as(u32, 9), all_val.val[9]);
}

test "split of an empty stream is two empty halves" {
    const corpus = try data.split(&[_]u32{}, 0.05);
    try std.testing.expectEqual(@as(usize, 0), corpus.train.len);
    try std.testing.expectEqual(@as(usize, 0), corpus.val.len);
}

test "split rejects a fraction outside zero to one" {
    // A ratio above 1 or below 0 would otherwise compute a cut outside the
    // slice, and a NaN passes every `>=` test it meets.
    const tokens = [_]u32{ 1, 2, 3, 4 };
    try std.testing.expectError(error.BadValFraction, data.split(&tokens, -0.1));
    try std.testing.expectError(error.BadValFraction, data.split(&tokens, 1.1));
    try std.testing.expectError(error.BadValFraction, data.split(&tokens, std.math.nan(f64)));
}

test "targets are the inputs shifted one token left" {
    // Five tokens at ctx 2 is one window of three: inputs 1,2 and targets 2,3.
    // The target at index i is the input at index i+1 of the same window, so a
    // batch that forgets the shift trains the model to predict its own input.
    const stream = [_]u32{ 1, 2, 3, 4, 5 };
    var b = try data.Batcher.init(std.testing.allocator, &stream, 2, 0);
    defer b.deinit();

    const got = try b.next();
    try std.testing.expect(got != null);
    var batch = got.?;
    defer batch.deinit();

    try std.testing.expectEqualSlices(u32, &.{ 1, 2 }, batch.inputs);
    try std.testing.expectEqualSlices(u32, &.{ 2, 3 }, batch.targets);
    try std.testing.expect((try b.next()) == null);
}

test "each window is a consecutive slice of the stream, shifted once" {
    // Eight tokens at ctx 2 gives a stride of 3 and two whole windows. Window 0
    // reads tokens 0,1,2 and window 1 reads 3,4,5, so the shift crosses inside
    // each window and never off the end of the stream.
    const stream = [_]u32{ 0, 1, 2, 3, 4, 5, 6, 7 };
    var b = try data.Batcher.init(std.testing.allocator, &stream, 2, 5);
    defer b.deinit();

    const first = (try b.next()).?;
    var fb = first;
    defer fb.deinit();
    const second = (try b.next()).?;
    var sb = second;
    defer sb.deinit();

    var inputs = [_][]const u32{ fb.inputs, sb.inputs };
    var targets = [_][]const u32{ fb.targets, sb.targets };
    std.mem.sort([]const u32, &inputs, {}, struct {
        fn lt(_: void, a: []const u32, c: []const u32) bool {
            return a[0] < c[0];
        }
    }.lt);
    std.mem.sort([]const u32, &targets, {}, struct {
        fn lt(_: void, a: []const u32, c: []const u32) bool {
            return a[0] < c[0];
        }
    }.lt);

    try std.testing.expectEqual(@as(usize, 0), windowOf(fb, 3));
    try std.testing.expectEqual(@as(usize, 1), windowOf(sb, 3));
    try std.testing.expectEqualSlices(u32, &.{ 0, 1 }, inputs[0]);
    try std.testing.expectEqualSlices(u32, &.{ 1, 2 }, targets[0]);
    try std.testing.expectEqualSlices(u32, &.{ 3, 4 }, inputs[1]);
    try std.testing.expectEqualSlices(u32, &.{ 4, 5 }, targets[1]);
    try std.testing.expect((try b.next()) == null);
}

test "every batch holds exactly ctx inputs and ctx targets" {
    const ctx = 5;
    const stride = ctx + 1;
    var stream: [4 * stride]u32 = undefined;
    for (&stream, 0..) |*t, i| t.* = @intCast(i);

    var b = try data.Batcher.init(std.testing.allocator, &stream, ctx, 31337);
    defer b.deinit();

    var count: usize = 0;
    while (try b.next()) |got| {
        var batch = got;
        defer batch.deinit();
        try std.testing.expectEqual(@as(usize, ctx), batch.inputs.len);
        try std.testing.expectEqual(@as(usize, ctx), batch.targets.len);
        count += 1;
    }
    try std.testing.expectEqual(@as(usize, 4), count);
}

test "draining every batch reproduces the stream, shifted once per window" {
    // 16 distinct tokens at ctx 3 gives a stride of 4 and four whole windows
    // with nothing left over. Inside one window the first token is seen once,
    // as an input, the last token once, as a target, and the two interior
    // tokens twice, once as an input and once as the next input's target. The
    // per window count pattern is therefore 1,2,2,1 and the drain yields
    // 4 * (1+2+2+1) = 24 = 2 * windows * ctx values. Nothing is invented and
    // nothing is skipped: the only tokens the shift does not carry twice are
    // the two window edges, one per end, and the target side reaches exactly
    // one token past the input side, which is the last token of the last
    // window.
    const token_count = 16;
    const ctx = 3;
    var stream: [token_count]u32 = undefined;
    for (&stream, 0..) |*t, i| t.* = @intCast(100 + i);

    var counts = [_]u16{0} ** token_count;
    var total: usize = 0;
    var b = try data.Batcher.init(std.testing.allocator, &stream, ctx, 4242);
    defer b.deinit();
    while (try b.next()) |got| {
        var batch = got;
        defer batch.deinit();
        for (batch.inputs) |t| {
            counts[@intCast(t - 100)] += 1;
            total += 1;
        }
        for (batch.targets) |t| {
            counts[@intCast(t - 100)] += 1;
            total += 1;
        }
    }

    try std.testing.expectEqual(@as(usize, 24), total);
    for (counts, 0..) |c, i| {
        const expected: u16 = if (i % 4 == 0 or i % 4 == 3) 1 else 2;
        try std.testing.expectEqual(expected, c);
    }
}

test "the shuffle reorders whole batches and never the tokens inside one" {
    // Seed 1337 over eight windows. The drain must visit each window exactly
    // once, in an order that is not the stream order, and every batch must
    // still run in ascending stream order with its target one step ahead of its
    // input. A shuffle applied inside a batch turns a strictly ascending run
    // into a descending pair, which is what the two loops below catch.
    const ctx = 2;
    const stride = ctx + 1;
    const windows = 8;
    var stream: [windows * stride]u32 = undefined;
    for (&stream, 0..) |*t, i| t.* = @intCast(i);

    var b = try data.Batcher.init(std.testing.allocator, &stream, ctx, 1337);
    defer b.deinit();

    var seen = [_]bool{false} ** windows;
    var drained: usize = 0;
    var in_stream_order = true;
    while (try b.next()) |got| {
        var batch = got;
        defer batch.deinit();

        for (batch.inputs, 0..) |t, i| {
            if (i > 0) try std.testing.expect(batch.inputs[i - 1] < t);
            try std.testing.expectEqual(batch.inputs[i] + 1, batch.targets[i]);
        }

        const w = windowOf(batch, stride);
        try std.testing.expect(w < windows);
        try std.testing.expect(!seen[w]);
        seen[w] = true;
        if (w != drained) in_stream_order = false;
        drained += 1;
    }

    try std.testing.expectEqual(@as(usize, windows), drained);
    for (seen) |s| try std.testing.expect(s);
    try std.testing.expect(!in_stream_order);
}

test "one seed gives one batch sequence and reset returns to it" {
    const ctx = 2;
    const stride = ctx + 1;
    const windows = 12;
    var stream: [windows * stride]u32 = undefined;
    for (&stream, 0..) |*t, i| t.* = @intCast(i);

    var a = try data.Batcher.init(std.testing.allocator, &stream, ctx, 1);
    defer a.deinit();
    var b = try data.Batcher.init(std.testing.allocator, &stream, ctx, 1);
    defer b.deinit();
    var c = try data.Batcher.init(std.testing.allocator, &stream, ctx, 2);
    defer c.deinit();
    // Never advanced, so it is the reference the post-reset drain is compared
    // against once `b` has been used up.
    var d = try data.Batcher.init(std.testing.allocator, &stream, ctx, 1);
    defer d.deinit();

    var differs_from_other_seed = false;
    for (0..windows) |_| {
        const ga = (try a.next()).?;
        var ba = ga;
        defer ba.deinit();
        const gb = (try b.next()).?;
        var bb = gb;
        defer bb.deinit();
        const gc = (try c.next()).?;
        var bc = gc;
        defer bc.deinit();

        try std.testing.expectEqualSlices(u32, ba.inputs, bb.inputs);
        try std.testing.expectEqualSlices(u32, ba.targets, bb.targets);
        if (!std.mem.eql(u32, ba.inputs, bc.inputs)) differs_from_other_seed = true;
    }
    try std.testing.expect((try a.next()) == null);
    try std.testing.expect(differs_from_other_seed);

    // Twelve windows under a different permutation would have to collide
    // exactly, so the seed is load bearing rather than decorative.
    a.reset();
    for (0..windows) |_| {
        const ga = (try a.next()).?;
        var ba = ga;
        defer ba.deinit();
        const gd = (try d.next()).?;
        var bd = gd;
        defer bd.deinit();
        try std.testing.expectEqualSlices(u32, ba.inputs, bd.inputs);
        try std.testing.expectEqualSlices(u32, ba.targets, bd.targets);
    }
    try std.testing.expect((try a.next()) == null);
    try std.testing.expect((try d.next()) == null);
}

test "next returns null at the end and keeps returning null" {
    // Four tokens at ctx 3 is exactly one window, so the second call is already
    // past the end and every later call has to stay there.
    const stream = [_]u32{ 10, 20, 30, 40 };
    var b = try data.Batcher.init(std.testing.allocator, &stream, 3, 11);
    defer b.deinit();

    const got = try b.next();
    try std.testing.expect(got != null);
    var batch = got.?;
    defer batch.deinit();
    try std.testing.expectEqualSlices(u32, &.{ 10, 20, 30 }, batch.inputs);
    try std.testing.expectEqualSlices(u32, &.{ 20, 30, 40 }, batch.targets);

    try std.testing.expect((try b.next()) == null);
    try std.testing.expect((try b.next()) == null);
    try std.testing.expect((try b.next()) == null);
}

test "a stream shorter than one window yields no batches" {
    // ctx 4 needs five tokens, and every length from nothing up to four is one
    // short. Zero batches and no read past the end, not a crash.
    const ctx = 4;
    for (0..ctx) |len| {
        var stream: [ctx]u32 = undefined;
        for (&stream, 0..) |*t, i| t.* = @intCast(i);

        var b = try data.Batcher.init(std.testing.allocator, stream[0..len], ctx, 3);
        defer b.deinit();
        try std.testing.expect((try b.next()) == null);
        try std.testing.expect((try b.next()) == null);
    }
}

test "a stream of exactly ctx plus one tokens yields exactly one batch" {
    const ctx = 3;
    var stream: [ctx + 1]u32 = undefined;
    for (&stream, 0..) |*t, i| t.* = @intCast(100 + i);

    var b = try data.Batcher.init(std.testing.allocator, &stream, ctx, 7);
    defer b.deinit();

    const got = try b.next();
    try std.testing.expect(got != null);
    var batch = got.?;
    defer batch.deinit();
    try std.testing.expectEqualSlices(u32, &.{ 100, 101, 102 }, batch.inputs);
    try std.testing.expectEqualSlices(u32, &.{ 101, 102, 103 }, batch.targets);
    try std.testing.expect((try b.next()) == null);
}

test "a trailing partial window is dropped, never padded" {
    // Ten tokens at ctx 2 has a stride of 3, so three whole windows fit at
    // indices 0, 3 and 6 and the final token 9 is one short of a fourth
    // window. Padding it would put a token in the stream that the corpus never
    // held, so the drain stops at three batches.
    const stream = [_]u32{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    var b = try data.Batcher.init(std.testing.allocator, &stream, 2, 21);
    defer b.deinit();

    var firsts = [_]u32{ 0, 0, 0 };
    var targets_first = [_]u32{ 0, 0, 0 };
    var count: usize = 0;
    while (try b.next()) |got| {
        var batch = got;
        defer batch.deinit();
        try std.testing.expect(count < firsts.len);
        firsts[count] = batch.inputs[0];
        targets_first[count] = batch.targets[0];
        count += 1;
    }
    try std.testing.expectEqual(@as(usize, 3), count);

    var want_firsts = [_]u32{ 0, 3, 6 };
    var want_targets = [_]u32{ 1, 4, 7 };
    std.mem.sort(u32, &firsts, {}, std.sort.asc(u32));
    std.mem.sort(u32, &targets_first, {}, std.sort.asc(u32));
    try std.testing.expectEqualSlices(u32, &want_firsts, &firsts);
    try std.testing.expectEqualSlices(u32, &want_targets, &targets_first);
    // The dropped tail is 7, 8 and 9, the last two of them left short of a
    // window and the ninth never reached as a target either.
    try std.testing.expect((try b.next()) == null);
}

test "a zero context is an error, not a stream of empty batches" {
    const stream = [_]u32{ 1, 2, 3, 4 };
    try std.testing.expectError(error.BadContext, data.Batcher.init(std.testing.allocator, &stream, 0, 1));
}

fn initDrainRelease(allocator: std.mem.Allocator, stream: []const u32) !void {
    var b = try data.Batcher.init(allocator, stream, 2, 4);
    defer b.deinit();
    while (try b.next()) |got| {
        var batch = got;
        batch.deinit();
    }
}

test "an allocation failure in init or next leaks nothing" {
    // `next` holds a half built batch on its error path, so the errdefer that
    // frees the first slice is the only thing standing between an out of memory
    // and a leak. This walks every allocation the walk performs, fails each one
    // in turn, and requires the error to surface rather than be swallowed.
    const stream = [_]u32{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 };
    try std.testing.checkAllAllocationFailures(std.testing.allocator, initDrainRelease, .{&stream});
}
