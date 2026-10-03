//! Per-op timing for one training step.
//!
//! The point is attribution. `src/cuda/README.md` publishes that attention is a
//! few percent of a step and `src/scale.zig` projects it from arithmetic; both
//! are estimates of where the time goes and neither measures it, so a reader can
//! agree with every published number and still not know which op to make faster.
//! This is that measurement.
//!
//! **Off unless asked for, and that is a correctness requirement, not a
//! performance one.** `outputs/loss.csv` is one host's bytes and
//! `zig build determinism` compares two runs for byte equality, so a timer read
//! on every op of every step is an arithmetic path the committed claim depends
//! on. The profiler is compiled in but inert unless `ZTRANSFORMER_PROFILE=1`, and
//! every entry point checks one global that is null on the default path.
//!
//! The clock is `Io.Clock.awake`, which is `CLOCK_MONOTONIC` on Linux and
//! `CLOCK_UPTIME_RAW` on macOS. Not `Io.Clock.real`: wall clock can step
//! backwards under NTP, and a negative per-op duration in a share table is a
//! worse defect than a coarse one.
//!
//! **The denominator is the wall-clock span, not the sum of the buckets.** That
//! is the whole difference between a share table and a tautology. Dividing each
//! op by the sum of all ops makes the shares sum to 100% by construction, so the
//! table would look complete whether the probes covered the step or not, and a
//! check that "the shares sum to 100%" could never fail. Dividing by the time
//! between the first and last reading leaves the un-attributed remainder
//! visible, so the shares sum to LESS than 100% by exactly the amount the
//! profiler failed to account for -- and that remainder is a number a reader can
//! look at.

const std = @import("std");
const Io = std.Io;

/// Every op kind a step is attributed to, named for what the code calls it, so a
/// reader can grep a probe to the line it times.
///
/// `loop` is FIRST and is placed at the TOP of the step loop, not after the
/// last op, because Zig runs every `defer` at its scope's closing brace -- which
/// is after the body's last probe and before the next one. Without this row that
/// time would land in `fetch`, and `fetch` would really be "the previous
/// step's frees, the row append, and the batch fetch", which is not what its name
/// says. `Cache.deinit` alone frees every block's eleven tensors, so the
/// mislabelling was not a rounding error: it renamed the largest non-backward
/// cost.
pub const Op = enum {
    loop,
    fetch,
    forward,
    loss,
    dlogits,
    backward,
    clip,
    adam,
    zero,
    eval,

    pub const count = @typeInfo(Op).@"enum".fields.len;
};

pub const env_var = "ZTRANSFORMER_PROFILE";

/// Accumulated nanoseconds and call counts per op kind, plus the span they were
/// measured over.
pub const Totals = struct {
    ns: [Op.count]i128 = @splat(0),
    calls: [Op.count]i128 = @splat(0),
    /// Wall-clock nanoseconds from the first reading to the last. The denominator
    /// for every share, and deliberately NOT the sum of `ns`.
    span_ns: i128 = 0,

    pub fn add(self: *Totals, op: Op, d_ns: i128) void {
        const i = @intFromEnum(op);
        self.ns[i] += d_ns;
        self.calls[i] += 1;
    }

    /// The sum of the buckets. Reported beside the span, because the gap between
    /// them IS the un-attributed time and hiding it is what makes a profiler
    /// look complete when it is not.
    pub fn bucketSumNs(self: Totals) i128 {
        var sum: i128 = 0;
        for (self.ns) |n| sum += n;
        return sum;
    }

    /// The denominator: the span when one was measured, the bucket sum
    /// otherwise, so a `Totals` built by hand still divides by something.
    pub fn denominatorNs(self: Totals) i128 {
        return if (self.span_ns > 0) self.span_ns else self.bucketSumNs();
    }

    /// Share table in declaration order, so two runs print the same table in the
    /// same order and a diff between them means something.
    pub fn rows(self: Totals) Shares {
        const denom = self.denominatorNs();
        var s: Shares = undefined;
        for (std.enums.values(Op), 0..) |_, i| {
            const ns = self.ns[i];
            s.ns[i] = ns;
            s.calls[i] = self.calls[i];
            s.share[i] = if (denom > 0) @as(f64, @floatFromInt(ns)) / @as(f64, @floatFromInt(denom)) else 0.0;
        }
        return s;
    }

    pub const Shares = struct {
        ns: [Op.count]i128,
        calls: [Op.count]i128,
        share: [Op.count]f64,
        /// Every share added up. Below 1.0 by the un-attributed remainder, which
        /// is the number that says whether the probes covered the step.
        pub fn shareSum(self: Shares) f64 {
            var sum: f64 = 0.0;
            for (self.share) |v| sum += v;
            return sum;
        }
    };

    /// The table, to stderr. A profiled run's numbers are not the curve and do
    /// not belong on stdout beside it, where a reader piping the curve would
    /// meet them.
    pub fn writeTable(self: Totals) void {
        const s = self.rows();
        // Unsigned, because `{d}` on a signed integer prints a leading `+` and
        // a table of durations has no negative value to report. It also makes
        // the rows parseable by the step that checks them.
        var total_calls: i128 = 0;
        for (s.calls) |c| total_calls += c;
        std.debug.print("op            ns_total   calls     share\n", .{});
        for (std.enums.values(Op), 0..) |op, i| {
            std.debug.print("{s:<12} {d:>9} {d:>8} {d:>8.3}%\n", .{
                @tagName(op),
                @as(u64, @intCast(@max(0, s.ns[i]))),
                @as(u64, @intCast(@max(0, s.calls[i]))),
                s.share[i] * 100.0,
            });
        }
        std.debug.print("{s:<12} {d:>9} {d:>8} {d:>8.3}%\n", .{
            "SUM", @as(u64, @intCast(@max(0, self.bucketSumNs()))), @as(u64, @intCast(total_calls)), s.shareSum() * 100.0,
        });
        std.debug.print("{s:<12} {d:>9} {s:>8} {d:>8.3}%\n", .{
            "SPAN", @as(u64, @intCast(@max(0, self.denominatorNs()))), "", 100.0,
        });
    }
};

/// One running measurement. A caller holds it and calls `stop` around each op.
pub const Profiler = struct {
    io: Io,
    totals: Totals = .{},
    last: Io.Timestamp,
    began: Io.Timestamp,
    /// False until the first probe. While it is false, `stop` MOVES the baseline
    /// instead of banking the gap, so the span runs from the first op of the
    /// first step rather than from `start`.
    ///
    /// This matters because `main` calls `enable` before `train.run`, and
    /// `train.run` does its one-time work before the first step: `initParams`,
    /// `zeroGrads`, one `AdamW.init` per parameter tensor, and two `Batcher.init`s. Banking
    /// that into the first probe would put the cost of building the model under
    /// a row named `loop`, and the reported "share of a step" would be a
    /// share of a setup plus a step. Discarding it is the honest choice: the
    /// table is about steps, and setup has no row.
    started: bool = false,

    pub fn start(io: Io) Profiler {
        const now = Io.Clock.awake.now(io);
        return .{ .io = io, .last = now, .began = now };
    }

    /// Close the gap since the last probe and attribute it to `op`. The reading
    /// that closes one probe opens the next, so no time is counted twice.
    pub fn stop(self: *Profiler, op: Op) void {
        const now = Io.Clock.awake.now(self.io);
        const d = self.last.durationTo(now);
        self.last = now;
        if (!self.started) {
            self.started = true;
            self.began = now;
            return;
        }
        self.totals.add(op, @intCast(@max(0, d.nanoseconds)));
    }

    /// Record the span and hand back the totals. The last gap, from the final
    /// probe to here, belongs to no op and shows up as the un-attributed share.
    pub fn finish(self: *Profiler) Totals {
        const now = Io.Clock.awake.now(self.io);
        self.last = now;
        self.totals.span_ns = @intCast(@max(0, self.began.durationTo(now).nanoseconds));
        return self.totals;
    }
};

/// Turn profiling on when the environment asks for it. Returns the profiler the
/// caller should hold, or null when it is off -- which is the default and the
/// only path the committed curve ever takes.
pub fn enable(arena: std.mem.Allocator, io: Io, environ: *std.process.Environ.Map) !?*Profiler {
    const raw = environ.get(env_var) orelse return null;
    // `=1` exactly, so a typo reads as off rather than as on.
    if (!std.mem.eql(u8, raw, "1")) return null;
    const p = try arena.create(Profiler);
    p.* = Profiler.start(io);
    return p;
}

/// The profiler `train.run` reads. Set by `main` before the run when
/// `ZTRANSFORMER_PROFILE=1`, and null otherwise.
///
/// A module-level global rather than a parameter on `train.run`, because that
/// function is called from a dozen tests and an argument would touch all of them
/// to plumb something that is null in every one.
pub var active: ?*Profiler = null;
