//! The mutation table, and the only thing that ever writes a mutant.
//!
//! `sh tools/mutation/run.sh` drives this. It never touches the tree it was
//! started from: it is handed the path of a throwaway git worktree and edits
//! that, so a crash mid-run cannot leave the repository with a mutant in it.
//!
//!   mutate list                      every name, one per line
//!   mutate show   <name>             the file it edits and the exact change
//!   mutate apply  <name> <root>      apply it, or fail loudly
//!
//! `apply` refuses unless the text it is looking for occurs exactly `expect`
//! times. That check is the whole reason this is a program and not a `sed`
//! line: a pattern that has stopped matching, because the source moved, would
//! otherwise write an unmutated file and the suite would pass, and the harness
//! would report a live mutant as a survivor. A false survivor is worse than no
//! harness, because it is a number a reader would believe.

const std = @import("std");
const Io = std.Io;

/// One deliberate defect, and everything needed to apply it and to talk about
/// the result.
const Mutation = struct {
    /// The selector `run.sh` matches on.
    name: []const u8,
    /// Relative to the repository root.
    file: []const u8,
    /// Exact source text, which is what makes the count check possible.
    from: []const u8,
    to: []const u8,
    /// How many times `from` must occur. One unless a mutation is a pattern
    /// that genuinely repeats, such as the three causal loops.
    expect: usize = 1,
    /// The load-bearing claim this mutation attacks, printed with the result.
    what: []const u8,
    // There is no `class` field here any more. One mutation set it, nothing read
    // it, and `run.sh` classifies survivors from its own table of the mutation
    // name, with tools/README.md stating the same taxonomy for a reader. A field
    // nothing reads is a third copy that drifts from both.
};

const mutations = [_]Mutation{
    .{
        // Written without an allocation on purpose. The first version of this
        // mutation allocated an f64 scratch row, and the suite "caught" it via
        // the leak detector because one extra allocation shifts which allocation
        // that test injects a failure into. That is a catch of the mutation's
        // bookkeeping, not of its arithmetic, and a mutation harness that
        // reports it as caught is measuring the wrong thing.
        //
        // This version moves the j loop inside k and accumulates one output
        // element in an f64 local. The k reduction still runs in the same fixed
        // order for every output element, so the run-to-run bit-identity the
        // module claims is preserved and the nesting is unobservable; the only
        // observable difference is the accumulator's width.
        .name = "matmul-f64-acc",
        .file = "src/tensor.zig",
        .what = "matmul accumulates its k reduction in f32; the mutant accumulates in f64",
        // Re-anchored on the eight-lane block that replaced the scalar loop, and it
        // spans the WHOLE while-block: an earlier attempt stopped at the k-loop's
        // closing brace, which left the original store line behind it and produced a
        // mutant that did not parse. The guard caught that as "zig fmt rejected the
        // mutant" rather than running it, which is the guard doing its job.
        //
        // The mutant keeps the lane structure and the k order and changes only the
        // accumulator width. That is the property: a wider reduction is more
        // accurate, so a suite that does not notice it is not testing the arithmetic.
        .from =
        \\        while (j + lanes <= b.cols) : (j += lanes) {
        \\            var acc: [lanes]f32 = @splat(0);
        \\            for (0..a.cols) |k| {
        \\                const scale = a_row[k];
        \\                const b_row = b.rowConst(k);
        \\                inline for (0..lanes) |u| acc[u] += scale * b_row[j + u];
        \\            }
        \\            inline for (0..lanes) |u| out_row[j + u] = acc[u];
        \\        }
        ,
        .to =
        \\        while (j + lanes <= b.cols) : (j += lanes) {
        \\            var acc: [lanes]f64 = @splat(0);
        \\            for (0..a.cols) |k| {
        \\                const scale = a_row[k];
        \\                const b_row = b.rowConst(k);
        \\                inline for (0..lanes) |u| acc[u] += @as(f64, scale) * @as(f64, b_row[j + u]);
        \\            }
        \\            inline for (0..lanes) |u| out_row[j + u] = @floatCast(acc[u]);
        \\        }
        ,
    },
    .{
        .name = "norm-reassociate",
        .file = "src/norm.zig",
        .what = "the fused scale is v / rms * w[i] reassociated to v * w[i] / rms",
        .from = "        for (x_row, 0..) |v, i| y_row[i] = v / rms * w[i];",
        .to = "        for (x_row, 0..) |v, i| y_row[i] = v * w[i] / rms;",
    },
    .{
        .name = "norm-f32-acc",
        .file = "src/norm.zig",
        .what = "the RMS sum-of-squares reduction keeps f32 precision and widens after",
        // Two lines on each side, deliberately. The original pattern spanned the
        // accumulator AND the justification comment AND the `rms` line, so when that
        // comment was rewritten -- the 4096-wide row it argued from was disowned by
        // `norm_test.zig` -- the pattern stopped matching code that never changed and
        // the mutant measured nothing. Squaring in f32 and widening keeps `sum_sq`
        // an f64, so the `rms` line below is untouched and the anchor stays the two
        // lines that carry the arithmetic. The defect under test is the same one: a
        // reduction that loses low bits before it accumulates them.
        .from =
        \\        var sum_sq: f64 = 0;
        \\        for (x_row) |v| sum_sq += @as(f64, v) * @as(f64, v);
        ,
        .to =
        \\        var sum_sq: f64 = 0;
        \\        for (x_row) |v| sum_sq += @as(f64, @as(f32, v * v));
        ,
    },
    .{
        .name = "norm-eps-zero",
        .file = "src/norm.zig",
        .what = "the RMSNorm epsilon goes to zero, in the one place both passes read it",
        .from = "pub const eps: f64 = 1e-5;",
        .to = "pub const eps: f64 = 0.0;",
    },
    .{
        .name = "clip-ge",
        .file = "src/optim.zig",
        .what = "clipByNorm clips when the norm equals the limit, not only when it exceeds it",
        .from = "    if (!(norm > @as(f64, max_norm))) return @floatCast(norm);",
        .to = "    if (!(norm >= @as(f64, max_norm))) return @floatCast(norm);",
    },
    .{
        .name = "optim-eps-zero",
        .file = "src/optim.zig",
        .what = "the AdamW denominator epsilon goes to zero",
        .from = "const eps: f64 = 1e-8;",
        .to = "const eps: f64 = 0.0;",
    },
    .{
        .name = "attn-no-mask",
        .file = "src/attention.zig",
        .what = "the causal mask is dropped: every query position reads the whole context",
        .from = "            for (0..t + 1) |s| {",
        .to = "            for (0..q.rows) |s| {",
        .expect = 3,
    },
    .{
        // `kv = h / group` with a group of one is `kv = h`, so this is the same
        // defect written on the line that defines the group. Written here
        // because the other spelling leaves `group` unused and the mutant does
        // not compile, which measures nothing.
        .name = "attn-no-gqa",
        .file = "src/attention.zig",
        .what = "every query head reads kv head h, so grouped-query sharing is gone",
        .from = "    const group = cfg.n_heads / cfg.n_kv_heads;",
        .to = "    const group: usize = 1;",
    },
    .{
        .name = "attn-dim-group-write",
        .file = "src/autograd.zig",
        .what = "the dq lane-group split writes a scaled value for group 2 and above, " ++
            "so only the second and later groups of the shipped head_dim are wrong. " ++
            "Every other fixture in the tree runs head_dim 12 or smaller, which is " ++
            "one group, so this is caught by exactly one test and by nothing else",
        .from =
        \\                for (0..lanes) |u| g.dq.set(t, h * dim + j + u, @floatCast(acc[u]));
        ,
        .to =
        \\                for (0..lanes) |u| g.dq.set(t, h * dim + j + u, @floatCast(if (g_lane >= 2) acc[u] * 1.5 else acc[u]));
        ,
    },
    .{
        .name = "attn-unroll-lane-write",
        .file = "src/autograd.zig",
        .what = "one unrolled lane in the gathered-row score dot writes lane zero's " ++
            "result, so seven of the eight prefixes in a group take the same dot product",
        .from =
        \\                        probs[s + u] = dot[u] * scale;
        \\                        row_max = @max(row_max, probs[s + u]);
        ,
        .to =
        \\                        probs[s + u] = dot[0] * scale;
        \\                        row_max = @max(row_max, probs[s + u]);
        ,
    },
    .{
        .name = "attn-no-rowmax",
        .file = "src/attention.zig",
        .what = "the softmax row max is dropped: every exp is taken on the raw score",
        .from =
        \\                const score = dot * scale;
        \\                scores[s] = score;
        \\                row_max = @max(row_max, score);
        ,
        .to =
        \\                const score = dot * scale;
        \\                scores[s] = score;
        \\                row_max = score;
        ,
    },
    .{
        .name = "rope-interleaved",
        .file = "src/rope.zig",
        .what = "RoPE pairs adjacent elements (i, 2i+1) instead of the halves (i, i+head_dim/2)",
        .from =
        \\                const lo: f64 = @floatCast(src[base + i]);
        \\                const hi: f64 = @floatCast(src[base + i + half]);
        \\                dst[base + i] = @floatCast(lo * c - hi * s);
        \\                dst[base + i + half] = @floatCast(hi * c + lo * s);
        ,
        .to =
        \\                const lo: f64 = @floatCast(src[base + 2 * i]);
        \\                const hi: f64 = @floatCast(src[base + 2 * i + 1]);
        \\                dst[base + 2 * i] = @floatCast(lo * c - hi * s);
        \\                dst[base + 2 * i + 1] = @floatCast(hi * c + lo * s);
        ,
    },
    .{
        .name = "gradcheck-eps-zero",
        .file = "src/gradcheck.zig",
        .what = "the finite-difference budget collapses to zero, so every gradient passes",
        .from = "const f32_epsilon: f64 = 1.0 / 16777216.0;",
        .to = "const f32_epsilon: f64 = 0.0;",
    },
    .{
        .name = "gradcheck-step",
        .file = "src/gradcheck.zig",
        .what = "the central-difference step goes from 1e-3 to 1e-1",
        .from = "const step: f32 = 1e-3;",
        .to = "const step: f32 = 1e-1;",
    },
    .{
        .name = "train-no-gradclear",
        .file = "src/train.zig",
        .what = "gradients are not cleared between steps, so every step adds to its own history",
        .from = "            for (flat_grads) |*g| g.fill(0);",
        .to = "            for (flat_grads) |_| {}",
    },
    .{
        .name = "mlp-swap-gate",
        .file = "src/mlp.zig",
        .what = "the SwiGLU gate and up projections are swapped: silu(up) * gate",
        .from = "    for (a.data, gate.data, up.data) |*dst, g, u| dst.* = silu(g) * u;",
        .to = "    for (a.data, gate.data, up.data) |*dst, g, u| dst.* = silu(u) * g;",
    },
    .{
        .name = "loss-sum",
        .file = "src/loss.zig",
        .what = "cross-entropy sums over positions instead of averaging them",
        .from = "    return total / @as(f64, @floatFromInt(t_count));",
        .to = "    return total;",
    },
    .{
        .name = "loss-no-rowmax",
        .file = "src/loss.zig",
        .what = "the softmax denominator subtracts a max seeded with 0 instead of the row maximum",
        // Re-anchored: this loop used to read `row[1..]` and gained the finiteness
        // guard inside it, so the two-line pattern no longer matched. The seed is the
        // thing under test and it is unchanged -- a max seeded with 0 is correct for an
        // all-positive row and wrong for a row whose logits are negative, which is
        // exactly the case `exp(z - max)` overflows on.
        .from =
        \\        var max = @as(f64, row[0]);
        \\        for (row) |zr| {
        \\            const z: f64 = zr;
        \\            if (!std.math.isFinite(z)) return error.NonFiniteLogits;
        \\            max = @max(max, z);
        \\        }
        ,
        .to =
        \\        var max: f64 = 0;
        \\        for (row) |zr| {
        \\            const z: f64 = zr;
        \\            if (!std.math.isFinite(z)) return error.NonFiniteLogits;
        \\            max = @max(max, z);
        \\        }
        ,
    },
};

fn find(allocator: std.mem.Allocator, name: []const u8) ?*const Mutation {
    for (&mutations) |*m| {
        if (std.mem.eql(u8, m.name, name)) return m;
    }
    _ = allocator;
    return null;
}

fn countOf(haystack: []const u8, needle: []const u8) usize {
    if (needle.len == 0) return 0;
    var n: usize = 0;
    var i: usize = 0;
    while (std.mem.indexOfPos(u8, haystack, i, needle)) |pos| {
        n += 1;
        i = pos + needle.len;
    }
    return n;
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.arena.allocator();
    const args = try init.minimal.args.toSlice(gpa);
    if (args.len < 2) return usage();

    var buffer: [4096]u8 = undefined;
    var out: Io.File.Writer = .init(.stdout(), init.io, &buffer);
    const w = &out.interface;
    defer w.flush() catch {};

    if (std.mem.eql(u8, args[1], "list")) {
        for (&mutations) |*m| try w.print("{s}\n", .{m.name});
        return;
    }
    const showing = std.mem.eql(u8, args[1], "show");
    if (!showing and !std.mem.eql(u8, args[1], "apply")) return usage();
    const want: usize = if (showing) 3 else 4;
    if (args.len != want) return usage();
    const m = find(gpa, args[2]) orelse {
        try w.print("no mutation named '{s}'\n", .{args[2]});
        return error.UnknownMutation;
    };
    if (std.mem.eql(u8, args[1], "show")) {
        try w.print("{s}\n  file  {s}\n  what  {s}\n", .{ m.name, m.file, m.what });
        for (lines(gpa, m.from)) |l| try w.print("  -     {s}\n", .{l});
        for (lines(gpa, m.to)) |l| try w.print("  +     {s}\n", .{l});
        return;
    }
    const path = try std.fs.path.join(gpa, &.{ args[3], m.file });
    const text = try Io.Dir.cwd().readFileAlloc(init.io, path, gpa, .unlimited);
    const n = countOf(text, m.from);
    if (n != m.expect) {
        try w.print(
            "{s}: found {d} occurrence(s) of its pattern in {s}, expected {d}. Refusing to run a suite against an unmutated tree.\n",
            .{ m.name, n, m.file, m.expect },
        );
        return error.PatternDidNotApply;
    }

    var total: usize = 0;
    var i: usize = 0;
    while (std.mem.indexOfPos(u8, text, i, m.from)) |pos| {
        total += (pos - i) + m.to.len;
        i = pos + m.from.len;
    }
    total += text.len - i;
    const out_text = try gpa.alloc(u8, total);
    i = 0;
    var at: usize = 0;
    while (std.mem.indexOfPos(u8, text, i, m.from)) |pos| {
        @memcpy(out_text[at..][0 .. pos - i], text[i..pos]);
        at += (pos - i);
        @memcpy(out_text[at..][0..m.to.len], m.to);
        at += m.to.len;
        i = pos + m.from.len;
    }
    @memcpy(out_text[at..][0 .. text.len - i], text[i..]);

    const file = try Io.Dir.cwd().createFile(init.io, path, .{});
    defer file.close(init.io);
    var wb: [4096]u8 = undefined;
    var fw: Io.File.Writer = .init(file, init.io, &wb);
    try fw.interface.writeAll(out_text);
    try fw.interface.flush();
    try w.print("{s}: {s} ({d} site{s})\n", .{ m.name, m.file, n, if (n == 1) "" else "s" });
}

fn lines(gpa: std.mem.Allocator, text: []const u8) [][]const u8 {
    var out: std.ArrayList([]const u8) = .empty;
    var it = std.mem.splitScalar(u8, text, '\n');
    while (it.next()) |l| {
        if (l.len != 0) out.append(gpa, l) catch unreachable;
    }
    return out.toOwnedSlice(gpa) catch unreachable;
}

fn usage() error{Usage} {
    std.debug.print(
        "usage: mutate list | mutate show <name> | mutate apply <name> <root>\n",
        .{},
    );
    return error.Usage;
}

test "every mutation names a file this repository has" {
    // Not runnable here without a repository root, and `run.sh` proves the
    // same thing harder: it applies each entry and fails on any whose pattern
    // no longer occurs. This test exists so an empty or malformed table is a
    // compile error rather than a run that reports "0 of 0 caught".
    try std.testing.expect(mutations.len > 0);
    for (&mutations) |m| {
        try std.testing.expect(m.name.len > 0);
        try std.testing.expect(m.file.len > 0);
        try std.testing.expect(m.what.len > 0);
        try std.testing.expect(countOf(m.from, m.from) == 1);
    }
}
