//! What one CPU attention call costs at each context length, printed beside the
//! PCIe floor a GPU implementation would face moving the very same tensors.
//!
//! Two numbers side by side, and a verdict that is nothing but a comparison
//! between them. This is the instrument that decided a fused attention kernel
//! was worth writing, so it is built to be argued with rather than believed: the
//! arithmetic is in three small pure functions below with tests on them, and the
//! table prints how many calls each row was measured over, because at the long
//! lengths one call takes seconds and a budget alone would buy a single sample.
//! Three is the floor and `min_calls` is where it is enforced.
//!
//! The left column is measured on whatever host ran it. The right column is
//! arithmetic: bytes that must cross the bus divided by a bandwidth this project
//! has measured elsewhere and names. The reading is one-directional and that is
//! the point. A kernel that has to read q, k and v across the bus and write the
//! result back cannot finish faster than those bytes divided by the bus,
//! whatever arithmetic it does once they arrive -- so `floor_us` is a **lower
//! bound** on kernel time, the ratio is the **largest** speedup available, and a
//! kernel that took as long as the bus would already be at that ratio.

const std = @import("std");
const Io = std.Io;
const attention = @import("attention.zig");
const tensor = @import("tensor.zig");

const Tensor = tensor.Tensor;

/// The shipped window first, then the lengths above it. The spacing is
/// deliberate: the two claims that came out of this sweep are one about the
/// shipped shape and one about long context, and a sweep with a gap where the
/// crossing might be would not support either.
const sweep = [_]usize{ 256, 512, 1024, 2048, 4096 };

/// Host-to-device bandwidth this project has measured, in bytes per second, from
/// the `ship` row of `src/cuda/README.md`: 256 x 128 x 4 bytes in and the same
/// out is 262144 bytes, and the table's `gpu_e2e` minus its `gpu` puts that round
/// trip at 43.14 us, so 262144 / 43.14e-6 = 6.1e9.
///
/// It belongs to that card, that driver and that machine. It is written here
/// rather than taken on faith so the floor column can be recomputed by hand, and
/// it is the weakest number in this file: a reader who distrusts it changes this
/// one constant and reruns.
pub const pcie_bytes_per_sec: f64 = 6.1e9;

/// How long to keep calling `forward` before believing the clock. One call at the
/// longest length takes seconds, so no budget short of minutes buys a second
/// sample there; the table says how many calls each row actually got rather than
/// implying a precision it does not have.
const budget_ns: u64 = 500 * std.time.ns_per_ms;
const max_calls: usize = 4096;

/// The least number of calls a row may be measured over, whatever the budget
/// says. At the longest length one call is seconds, so a budget alone buys one
/// sample, and a single sample on a shared host is not a figure this project
/// quotes anywhere else.
const min_calls: usize = 3;

/// Bytes a GPU implementation must move for one call: q in, k in, v in, result
/// out. The weights are not counted, because they are shared across every call
/// and would be resident either way.
pub fn pcieBytes(T: usize, n_heads: usize, n_kv_heads: usize, head_dim: usize) u64 {
    const t: u64 = @intCast(T);
    const nh: u64 = @intCast(n_heads);
    const nk: u64 = @intCast(n_kv_heads);
    const d: u64 = @intCast(head_dim);
    return (2 * t * nh * d + 2 * t * nk * d) * 4;
}

/// Microseconds those bytes take on the bus, which is the floor.
pub fn floorUs(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / pcie_bytes_per_sec * 1e6;
}

/// Which side of the floor the CPU landed on, as the exact string the table
/// prints. Kept as one function so a test can pin the choice rather than the
/// prose describing it.
pub fn verdict(cpu_us: f64, floor_us: f64) []const u8 {
    return if (cpu_us < floor_us)
        "cpu under the floor: no kernel of this shape can win"
    else
        "floor is below the cpu: the kernel's own cost decides";
}

pub fn print(init: std.process.Init, gpa: std.mem.Allocator, w: *std.Io.Writer) !void {
    const cfg = attention.defaultConfig();
    const acfg = attention.Config{
        .n_heads = cfg.n_heads,
        .n_kv_heads = cfg.n_kv_heads,
        .head_dim = cfg.head_dim,
    };

    try w.writeAll(
        \\ztransformer attention bench
        \\
        \\One call of attention.forward at each context length, on this host, with
        \\the shipped head geometry. The timer includes the two allocations the call
        \\makes. Note which two: this is the evaluation path, where forward is given
        \\no sink. The training step passes a live sink and pays a third and far
        \\larger allocation to materialise the whole score matrix, so a training
        \\step costs more than the number in this table and this table is not a
        \\model of one.
        \\
        \\floor_us is q + k + v in and the result out, over PCIe, at 6.1 GB/s. That
        \\is a lower bound and not a prediction, so the ratio beside it is the
        \\largest speedup available rather than a floor on the win.
        \\
        \\cpu_us is the MINIMUM per-call time over `calls` calls rather than the mean,
        \\and the ratio beside it is built from that minimum. Contention and frequency
        \\scaling only make a call slower, so the fastest call observed is the closest
        \\estimate of the unloaded cost on a shared host.
        \\
        \\
    );
    try w.print(
        \\
        \\     ctx       calls  cpu_us_min   pcie_MiB    floor_us   ratio  verdict
        \\  --------  ---------  ------------  ---------  ----------  --------  -----------------------------
        \\
    , .{});

    for (sweep) |t| {
        var q = try Tensor.init(gpa, t, cfg.n_heads * cfg.head_dim);
        defer q.deinit();
        var k = try Tensor.init(gpa, t, cfg.n_kv_heads * cfg.head_dim);
        defer k.deinit();
        var v = try Tensor.init(gpa, t, cfg.n_kv_heads * cfg.head_dim);
        defer v.deinit();
        fill(q.data);
        fill(k.data);
        fill(v.data);

        // One untimed call, so the first timed one is not paying for whatever the
        // allocator and the page faults do once.
        {
            var warm = try attention.forward(gpa, q, k, v, acfg);
            warm.deinit();
        }

        // Timed call by call, and the MINIMUM is what the table reports. That is
        // a deliberate choice and it is the opposite of what a mean would tell a
        // reader: contention and frequency scaling only ever make a call slower,
        // so on a shared host the fastest call observed is the closest estimate of
        // what the code costs unloaded and the mean is mostly a measurement of
        // whoever else was running. Two sweeps of this table on the same 32-core
        // host put the shipped row at 8344 us and 10311 us -- a 24% difference
        // that is entirely host load and neither number being wrong.
        //
        // Reading the clock once per call is what makes a per-call minimum
        // possible at all. At the shortest length a call is milliseconds, so a
        // read costing tens of nanoseconds is noise rather than overhead.
        var calls: usize = 0;
        var best_ns: u64 = std.math.maxInt(u64);
        while (calls < max_calls) {
            const t0 = Io.Clock.awake.now(init.io);
            var out = try attention.forward(gpa, q, k, v, acfg);
            out.deinit();
            const dt: u64 = @intCast(t0.untilNow(init.io, .awake).toNanoseconds());
            if (dt < best_ns) best_ns = dt;
            calls += 1;
            if (calls >= min_calls and dt * @as(u64, calls) >= budget_ns) break;
        }
        const cpu_us = @as(f64, @floatFromInt(best_ns)) / 1000.0;

        const bytes = pcieBytes(t, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim);
        const mib = @as(f64, @floatFromInt(bytes)) / (1024.0 * 1024.0);
        const fl = floorUs(bytes);

        try w.print("{d:>10}  {d:>9}  {d:>12.2}  {d:>9.3}  {d:>10.2}  {d:>7.0}x  {s}\n", .{
            t,
            calls,
            cpu_us,
            mib,
            fl,
            cpu_us / fl,
            verdict(cpu_us, fl),
        });
    }
    try w.writeAll("\n");
}

/// Deterministic, and the same every run, so two hosts can be compared. The
/// values themselves do not matter: nothing in `forward` branches on them, so
/// there is no interesting case to hit here. Initialising is still not optional,
/// because an uninitialised read is undefined behaviour rather than a slow number.
fn fill(data: []f32) void {
    var x: u32 = 0xbeef;
    for (data) |*d| {
        x = x *% 1664525 +% 1013904223;
        d.* = @as(f32, @floatFromInt(x >> 8)) / 16777216.0 - 0.5;
    }
}
