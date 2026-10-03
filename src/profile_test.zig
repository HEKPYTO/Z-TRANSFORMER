const std = @import("std");
const profile = @import("profile.zig");

const testing = std.testing;

test "an op that never ran has no share and no calls" {
    var t: profile.Totals = .{};
    t.add(.forward, 1000);
    t.add(.backward, 3000);
    t.span_ns = 5000;

    const s = t.rows();
    try testing.expectEqual(@as(i128, 0), s.ns[@intFromEnum(profile.Op.eval)]);
    try testing.expectEqual(@as(i128, 0), s.calls[@intFromEnum(profile.Op.eval)]);
    try testing.expectEqual(@as(f64, 0.0), s.share[@intFromEnum(profile.Op.eval)]);
    // The two that ran carry the rest, and their shares are of the SPAN.
    try testing.expectEqual(@as(f64, 0.2), s.share[@intFromEnum(profile.Op.forward)]);
    try testing.expectEqual(@as(f64, 0.6), s.share[@intFromEnum(profile.Op.backward)]);
}

test "the shares fall short of one by the un-attributed remainder" {
    // The whole reason the denominator is the span and not the bucket sum. A
    // profiler dividing by its own buckets would print 1.000000 here and an
    // equality check would call that complete.
    var t: profile.Totals = .{};
    t.add(.forward, 1000);
    t.add(.backward, 3000);
    t.span_ns = 5000; // 4000 accounted for, 1000 not

    const s = t.rows();
    try testing.expect(s.shareSum() < 1.0);
    try testing.expectApproxEqAbs(@as(f64, 0.8), s.shareSum(), 1e-12);
}

test "the span is the denominator, and the bucket sum is reported beside it" {
    var t: profile.Totals = .{};
    t.add(.forward, 2500);
    t.span_ns = 5000;

    try testing.expectEqual(@as(i128, 2500), t.bucketSumNs());
    try testing.expectEqual(@as(i128, 5000), t.denominatorNs());
}

test "with no span the bucket sum is the denominator, so a hand-built table still divides" {
    // A `Totals` assembled without `finish()` has no span. Falling back to the
    // bucket sum keeps `rows()` from dividing by zero, and the shares then sum to
    // one -- which is the tautology, and is why the span exists.
    var t: profile.Totals = .{};
    t.add(.forward, 1000);
    t.add(.backward, 3000);

    try testing.expectEqual(@as(i128, 0), t.span_ns);
    try testing.expectEqual(@as(i128, 4000), t.denominatorNs());
    try testing.expectApproxEqAbs(@as(f64, 1.0), t.rows().shareSum(), 1e-12);
}

test "add accumulates per op and counts every call" {
    var t: profile.Totals = .{};
    t.add(.adam, 10);
    t.add(.adam, 20);
    t.add(.adam, 30);
    const i = @intFromEnum(profile.Op.adam);

    try testing.expectEqual(@as(i128, 60), t.ns[i]);
    try testing.expectEqual(@as(i128, 3), t.calls[i]);
}

test "every op kind is reported, so a new one cannot be added and silently dropped" {
    // The table and the array are both sized from `Op.count`. If a kind were
    // added to the enum and the table iterated something else, this is where it
    // would show: the row count is the enum's.
    var t: profile.Totals = .{};
    for (std.enums.values(profile.Op)) |op| t.add(op, 1);
    t.span_ns = profile.Op.count;

    const s = t.rows();
    try testing.expectEqual(profile.Op.count, s.ns.len);
    for (s.calls) |c| try testing.expectEqual(@as(i128, 1), c);
}
