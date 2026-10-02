//! The CPU half of the fused attention benchmark: it generates the inputs, runs
//! `attention.forward` to get the reference answer and the CPU timing, and writes
//! both out for `src/cuda/attn.cu` to compare against. That program compares and
//! does not generate, so a kernel result can never be graded against tensors it
//! also produced.
//!
//! Files written into the scratch directory, one set per shape:
//!   <tag>.q.bin  <tag>.k.bin  <tag>.v.bin   inputs
//!   <tag>.y.bin                                the CPU answer, in f32
//!   manifest.tsv            one row per shape, which attn.cu reads
//!
//! manifest.tsv is: tag, T, n_heads, n_kv_heads, head_dim, iters, cpu_us, and
//! every row carries its OWN geometry rather than a shared one. The first five
//! rows are the shipped configuration, because the question this answers starts
//! with whether a kernel beats the CPU at the shape the project actually runs.
//! The next two are Llama-3's -- 32 heads over 8 kv heads at head_dim 128 --
//! because a kernel only ever run at head_dim 32 has not been shown to run at the
//! width the goal names, and at that width the tile has to be narrower than the
//! head or the shared-memory ask exceeds what a block will be granted.
//!
//! The last four are the WIDTHS, and they exist because the kernel used to refuse
//! them. `zt_attn_dim_ok` demanded a power of two, so head_dims of 96, 192 and 80
//! were turned away for a property no loop reads -- and those are Phi-3's head,
//! DeepSeek-V2 and V3's MLA head, and a family of 80-wide ones. A width guard that
//! rejects three shipping models is a defect that only shows up when someone tries
//! to run one, so the widths are in the manifest and graded like any other row.
//! `phi3-T512` is there because one context length cannot show that a new width
//! behaves: `ctx256` and `ctx512` at head_dim 32 differ, and so do `llama3-T256`
//! and `llama3-T512`.
//!
//! `gemma2-T256` is also the shared-memory case. At head_dim 256 the tile the
//! benchmark asks for wants more per-block memory than an sm_86 will opt into, so
//! the launcher narrows the tile rather than ending the process, and this row is
//! what decides whether a run at a narrower tile is allowed to count.
//!
//! Nothing here is a gate. The parity decision is `attn.cu`'s, and it is the only
//! thing in the pair that decides anything.

const std = @import("std");
const Io = std.Io;
// A named module, not a relative path: a module rooted at src/cuda/ may not
// import outside its own directory, so run-attn.sh passes src/autograd.zig in
// whole. Only one module: autograd.zig owns model, tensor, norm, rope, mlp and
// attention through its own relative imports, and it re-exports `attention`, so
// both the forward reference and the backward one come through this single
// dependency. Passing src/attention.zig alongside it would compile every shared
// symbol twice. norm_twin.zig does the same for src/norm.zig.
const autograd = @import("autograd");
const attention = autograd.attention;
const model = autograd.model;

const Tensor = attention.Tensor;

/// The same lengths `zig build attn-bench` sweeps, so the two tables can be read
/// against each other without either being re-derived. `iters` is the GPU repeat
/// count `attn.cu` times; it falls with T because the kernel is quadratic.
///
/// The rows past the first five are not the shipped geometry and they are the
/// reason this table has a geometry column at all. The goal names Llama-3, whose
/// attention is 32 heads over 8 kv heads at head_dim 128, and at that width a tile
/// of 128 asks a block for 130 KiB of shared memory -- past what an sm_86 block
/// will opt into. A kernel only ever run at head_dim 32 has not been shown to run
/// at the width the project is about, so these rows run it there. The last four
/// add the widths the power-of-two guard used to refuse, and the CPU reference is
/// `attention.forward` unchanged, because it is written against whatever geometry
/// it is handed and holds no opinion about powers of two either.
const shapes = [_]struct {
    tag: []const u8,
    T: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    iters: usize,
}{
    .{ .tag = "ctx256", .T = 256, .n_heads = 4, .n_kv_heads = 2, .head_dim = 32, .iters = 2000 },
    .{ .tag = "ctx512", .T = 512, .n_heads = 4, .n_kv_heads = 2, .head_dim = 32, .iters = 1000 },
    .{ .tag = "ctx1024", .T = 1024, .n_heads = 4, .n_kv_heads = 2, .head_dim = 32, .iters = 300 },
    .{ .tag = "ctx2048", .T = 2048, .n_heads = 4, .n_kv_heads = 2, .head_dim = 32, .iters = 80 },
    .{ .tag = "ctx4096", .T = 4096, .n_heads = 4, .n_kv_heads = 2, .head_dim = 32, .iters = 20 },
    .{ .tag = "llama3-T256", .T = 256, .n_heads = 32, .n_kv_heads = 8, .head_dim = 128, .iters = 50 },
    .{ .tag = "llama3-T512", .T = 512, .n_heads = 32, .n_kv_heads = 8, .head_dim = 128, .iters = 10 },
    // Phi-3: 32 heads over 8 kv heads at head_dim 96. Two context lengths, because
    // one cannot show that a width behaves rather than happening to.
    .{ .tag = "phi3-T256", .T = 256, .n_heads = 32, .n_kv_heads = 8, .head_dim = 96, .iters = 50 },
    .{ .tag = "phi3-T512", .T = 512, .n_heads = 32, .n_kv_heads = 8, .head_dim = 96, .iters = 10 },
    // DeepSeek's MLA head width, with no GQA on it: 16 heads over 16 kv heads, so
    // the group is 1 and the two collapses GQA and the correct kernel part company
    // for a different reason than at every other row.
    .{ .tag = "mla-T256", .T = 256, .n_heads = 16, .n_kv_heads = 16, .head_dim = 192, .iters = 50 },
    // Gemma-2's head width, and the widest this kernel accepts. The tile that the
    // rest of the table asks for does not fit a block here.
    .{ .tag = "gemma2-T256", .T = 256, .n_heads = 8, .n_kv_heads = 4, .head_dim = 256, .iters = 50 },
};

/// How long to keep calling `forward` before believing the clock, and the cap.
const budget_ns: u64 = 500 * std.time.ns_per_ms;
const max_calls: usize = 4096;
/// The least number of calls a row may be measured over. One call at T = 4096
/// takes seconds, so a budget alone buys a single sample on a shared host.
const min_calls: usize = 3;

pub fn main(init: std.process.Init) !void {
    const scratch = init.arena.allocator();
    const args = try init.minimal.args.toSlice(scratch);
    if (args.len != 2) {
        std.debug.print("usage: {s} <scratch-dir>\n", .{args[0]});
        return error.Usage;
    }
    const dir = args[1];

    // The tensors go on the real allocator, not the arena. `forward` allocates
    // its own output and this frees it between calls, so an arena would grow by
    // one output per iteration and the long shapes would measure the allocator
    // rather than the op.
    const gpa = init.gpa;

    // The first five rows are the shipped geometry and every row after them is
    // not, so the table is checked against the config rather than assumed to be it.
    // Sliced to five on purpose: the point of the check is that the SHIPPED rows
    // are the shipped geometry, and widening the slice would quietly make it a
    // claim about the other six.
    const shipped = attention.defaultConfig();
    for (shapes[0..5]) |s| {
        if (s.n_heads != shipped.n_heads or s.n_kv_heads != shipped.n_kv_heads or
            s.head_dim != shipped.head_dim) return error.SweepIsNotTheShippedGeometry;
    }

    var manifest: std.ArrayList(u8) = .empty;
    defer manifest.deinit(gpa);

    for (shapes) |s| {
        const cfg = attention.Config{
            .n_heads = s.n_heads,
            .n_kv_heads = s.n_kv_heads,
            .head_dim = s.head_dim,
        };
        var q = try Tensor.init(gpa, s.T, s.n_heads * s.head_dim);
        defer q.deinit();
        var k = try Tensor.init(gpa, s.T, cfg.n_kv_heads * cfg.head_dim);
        defer k.deinit();
        var v = try Tensor.init(gpa, s.T, cfg.n_kv_heads * cfg.head_dim);
        defer v.deinit();
        fill(q.data, 0xbeef);
        fill(k.data, 0x1234);
        fill(v.data, 0x5678);

        // One untimed call, so the timed calls are not paying for the first
        // touch of the pages the allocator has just handed back.
        {
            var warm = try attention.forward(gpa, q, k, v, cfg);
            warm.deinit();
        }

        // Timed call by call and reported as a MINIMUM, for the same reason
        // `zig build attn-bench` reports a minimum: contention and frequency
        // scaling only make a call slower, so the fastest call observed is the
        // closest estimate of the unloaded cost. A mean here would make the
        // kernel look better than it is whenever the host is busy, which is
        // exactly the direction this table must not be wrong in.
        var calls: usize = 0;
        var best_ns: u64 = std.math.maxInt(u64);
        while (calls < max_calls) {
            const t0 = Io.Clock.awake.now(init.io);
            var out = try attention.forward(gpa, q, k, v, cfg);
            out.deinit();
            const dt: u64 = @intCast(t0.untilNow(init.io, .awake).toNanoseconds());
            if (dt < best_ns) best_ns = dt;
            calls += 1;
            if (calls >= min_calls and dt * @as(u64, calls) >= budget_ns) break;
        }
        const cpu_us = @as(f64, @floatFromInt(best_ns)) / 1000.0;

        // The reference answer is a separate call, kept rather than reused from
        // the timing loop, so the bytes attn.cu grades are not bytes something
        // measured on the way past.
        var ref = try attention.forward(gpa, q, k, v, cfg);
        defer ref.deinit();

        try writeFloats(init.io, try std.fmt.allocPrint(scratch, "{s}/{s}.q.bin", .{ dir, s.tag }), q.data);
        try writeFloats(init.io, try std.fmt.allocPrint(scratch, "{s}/{s}.k.bin", .{ dir, s.tag }), k.data);
        try writeFloats(init.io, try std.fmt.allocPrint(scratch, "{s}/{s}.v.bin", .{ dir, s.tag }), v.data);
        try writeFloats(init.io, try std.fmt.allocPrint(scratch, "{s}/{s}.y.bin", .{ dir, s.tag }), ref.data);

        // The backward's input and its three reference answers. `dout` is filled
        // from its own seed rather than derived from `ref`, because
        // `attentionBackward` never sees the forward output -- it rebuilds the
        // softmax from q and k -- so any `dout` exercises the identical code path,
        // and a distinct seed is what stops a kernel that swaps `dout` for
        // something else from passing.
        var dout = try Tensor.init(gpa, s.T, s.n_heads * s.head_dim);
        defer dout.deinit();
        fill(dout.data, 0x9abc);

        // `attentionBackward` takes a `model.Config` and reads three fields of it.
        // The rest are given values that are never read rather than left
        // undefined, so a future field that IS read fails loudly here instead of
        // grading against whatever happened to be on the stack.
        const mcfg = model.Config{
            .n_layers = 1,
            .n_heads = s.n_heads,
            .n_kv_heads = s.n_kv_heads,
            .head_dim = s.head_dim,
            .n_ctx = s.T,
            .vocab_size = 0,
            .ffn_mult = 0,
        };
        // The CPU backward is timed here, and this is the ONLY place in the
        // repository where its cost is measured. It matters because the backward
        // is what decides whether moving attention to the GPU is worth anything
        // at the shape this project actually trains at: the forward is 2.6% of a
        // step on the 32-core host, so the backward's multiple of it is the whole
        // question. Every other figure for it in this repository was an estimate
        // ("about four times the work"), and an estimate cannot decide a
        // go/no-go.
        //
        // Reported as a MINIMUM, for the reason the forward's timing above gives:
        // on a shared host contention and frequency scaling only make a call
        // slower, so the fastest call observed is the closest estimate of the
        // unloaded cost. A mean would flatter the kernel whenever the host is
        // busy, which is the direction this number must not be wrong in -- it
        // would understate how much there is to win.
        var bwd_us: f64 = 0;
        {
            var bcalls: usize = 0;
            var bbest_ns: u64 = std.math.maxInt(u64);
            while (bcalls < 3) : (bcalls += 1) {
                var warm_b = try autograd.attentionBackward(gpa, q, k, v, dout, mcfg);
                warm_b.deinit();
            }
            bcalls = 0;
            while (bcalls < 3) : (bcalls += 1) {
                const tb0 = Io.Clock.awake.now(init.io);
                var g = try autograd.attentionBackward(gpa, q, k, v, dout, mcfg);
                g.deinit();
                const dtb: u64 = @intCast(tb0.untilNow(init.io, .awake).toNanoseconds());
                if (dtb < bbest_ns) bbest_ns = dtb;
            }
            bwd_us = @as(f64, @floatFromInt(bbest_ns)) / 1000.0;
        }
        var bwd = try autograd.attentionBackward(gpa, q, k, v, dout, mcfg);
        defer bwd.deinit();

        try writeFloats(init.io, try std.fmt.allocPrint(scratch, "{s}/{s}.dout.bin", .{ dir, s.tag }), dout.data);
        try writeFloats(init.io, try std.fmt.allocPrint(scratch, "{s}/{s}.dq.bin", .{ dir, s.tag }), bwd.dq.data);
        try writeFloats(init.io, try std.fmt.allocPrint(scratch, "{s}/{s}.dk.bin", .{ dir, s.tag }), bwd.dk.data);
        try writeFloats(init.io, try std.fmt.allocPrint(scratch, "{s}/{s}.dv.bin", .{ dir, s.tag }), bwd.dv.data);

        try manifest.appendSlice(gpa, try std.fmt.allocPrint(scratch, "{s} {d} {d} {d} {d} {d} {d:.6}\n", .{
            s.tag, s.T, s.n_heads, s.n_kv_heads, s.head_dim, s.iters, cpu_us,
        }));
        std.debug.print("attn: {s} T={d} {d}/{d} heads dim {d} cpu {d:.2} us over {d} calls" ++ "   cpu_backward {d:.2} us\n", .{
            s.tag, s.T, s.n_heads, s.n_kv_heads, s.head_dim, cpu_us, calls, bwd_us,
        });
    }

    try writeText(init.io, try std.fmt.allocPrint(scratch, "{s}/manifest.tsv", .{dir}), manifest.items);
    std.debug.print("attn: wrote inputs, the reference output and manifest.tsv\n", .{});
}

/// Deterministic, and different per tensor so a kernel that transposes q against
/// k cannot pass by accident. Nothing in `forward` branches on the values, so
/// only the range matters: a NaN here would change the timing by making the
/// compare fail rather than by making the arithmetic interesting.
fn fill(data: []f32, seed: u32) void {
    var x: u32 = seed;
    for (data) |*d| {
        x = x *% 1664525 +% 1013904223;
        d.* = @as(f32, @floatFromInt(x >> 8)) / 16777216.0 - 0.5;
    }
}

fn writeFloats(io: Io, path: []const u8, data: []const f32) !void {
    const file = try Io.Dir.cwd().createFile(io, path, .{});
    defer file.close(io);
    var buffer: [4096]u8 = undefined;
    var out: Io.File.Writer = .init(file, io, &buffer);
    try out.interface.writeAll(std.mem.sliceAsBytes(data));
    try out.interface.flush();
}

fn writeText(io: Io, path: []const u8, bytes: []const u8) !void {
    const file = try Io.Dir.cwd().createFile(io, path, .{});
    defer file.close(io);
    var buffer: [4096]u8 = undefined;
    var out: Io.File.Writer = .init(file, io, &buffer);
    try out.interface.writeAll(bytes);
    try out.interface.flush();
}
