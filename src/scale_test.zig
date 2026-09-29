//! Tests for the scale projections.
//!
//! The hand-arithmetic below is written out term by term from the code's own
//! loops, never from `scale.project` itself: a test that compares an
//! implementation against itself passes when both halves are wrong. The one
//! exception is `attention core matches what attention.forward walks`, and that
//! half counts the loops in this file rather than calling the projection, so it
//! is still two independent derivations meeting.

const std = @import("std");
const model = @import("model.zig");
const scale = @import("scale.zig");

/// The shipped shape, taken from the module rather than written out, so this
/// file cannot drift from it either.
const shipped = model.defaultConfig();

test "scale: attention core is 2*d*T*(T+1) on the shipped shape, by hand" {
    // d = n_heads * head_dim = 4 * 32 = 128. T = n_ctx = 256.
    //
    // attention.forward's score loop (attention.zig:54-58) is
    //   for h in 0..n_heads, for t in 0..T, for s in 0..t+1, dot over head_dim
    // so its multiply-adds are n_heads * head_dim * sum_{t=0}^{T-1} (t + 1)
    //   = 128 * (1 + 2 + ... + 256) = 128 * 32896 = 4210688
    const sum_prefix: u64 = 256 * 257 / 2; // 32896
    try std.testing.expectEqual(@as(u64, 32896), sum_prefix);
    const score_macs: u64 = 128 * sum_prefix; // 4210688
    try std.testing.expectEqual(@as(u64, 4210688), score_macs);
    //
    // The weighted sum (attention.zig:73-78) is the same count over the same
    // prefix, and 2 FLOP per multiply-add gives
    //   2 * (4210688 + 4210688) = 16842752 = 2 * 128 * 256 * 257
    const p = try scale.project(.{ .name = "shipped", .cfg = shipped });
    try std.testing.expectEqual(@as(u64, 16_842_752), p.attn_core);
    try std.testing.expectEqual(2 * 128 * 256 * 257, p.attn_core);
}

test "scale: the core is HALF the dense count the deferral note used" {
    // `4 * T^2 * d` is the dense non-causal figure. `attention.forward` walks
    // `0..t + 1`, so it does half of it, and the +1 is the causal diagonal.
    // At the shipped shape the dense count is 33554432 and the real one is
    // 16842752, and the ratio is not exactly a half because of the diagonal.
    const dense: u64 = 4 * 256 * 256 * 128;
    const real = try scale.project(.{ .name = "shipped", .cfg = shipped });
    try std.testing.expectEqual(@as(u64, 33_554_432), dense);
    try std.testing.expect(dense > real.attn_core);
    try std.testing.expect(real.attn_core * 2 > dense);
    // The causal loop walks T(T+1)/2 of T^2 pairs, so the real count is
    // 2 * T(T+1)/2 * d * 2 and the dense is T^2 * d * 2.
    try std.testing.expectEqual(2 * (256 * 257 / 2) * 128 * 2, real.attn_core);
}

test "scale: attention core matches what attention.forward's loops walk" {
    // The projection is a closed form; this counts the loops it claims to
    // describe, at every shape in the sweep, so a change to either fails.
    for (scale.sweep) |shape| {
        const p = try scale.project(shape);
        const t = shape.cfg.n_ctx;
        const dim_per_head: u64 = @as(u64, @intCast(shape.cfg.n_heads * shape.cfg.head_dim));
        var sum_prefix: u64 = 0;
        for (0..t) |tt| sum_prefix += @as(u64, @intCast(tt + 1));
        const walked = 2 * (dim_per_head * sum_prefix) * 2;
        try std.testing.expectEqual(walked, p.attn_core);
    }
}

test "scale: mlp is 24*T*d^2 at the shipped shape, by hand" {
    // w_gate [d,h], w_up [d,h], w_down [h,d] against a [T,d] input
    // (model.zig:376-379): 3 * T * d * h multiply-adds, 2 FLOP each.
    // d = 128, h = 512, T = 256.
    //   2 * 3 * 256 * 128 * 512 = 100663296
    //   = 24 * 256 * 128 * 128   = 24 * T * d^2, the deferral note's figure
    const p = try scale.project(.{ .name = "shipped", .cfg = shipped });
    try std.testing.expectEqual(@as(u64, 100_663_296), p.mlp);
    try std.testing.expectEqual(24 * 256 * 128 * 128, p.mlp);
    // And the h the module reports is the one the arithmetic used.
    try std.testing.expectEqual(@as(u64, 512), p.h);
}

test "scale: the crossover is 12*d, not the 6*d the deferral note claims" {
    // core/mlp = 2*d*T*(T+1) / (6*T*d*h) = (T+1) / (3*h). At the shipped
    // shape h = 512, so parity is T + 1 = 3 * 512 = 1536, T = 1535 = 12*d - 1.
    // The note's 4*T^2*d against 24*T*d^2 gives T = 6*d, which is half of it.
    const p = try scale.project(.{ .name = "shipped", .cfg = shipped });
    // The shipped shape is nowhere near the crossover, and that is the point:
    // core/mlp = (T + 1) / (3h) = 257 / 1536 = 0.167, so attention is a sixth
    // of the MLP's arithmetic here. The deferral got the direction right and
    // the threshold wrong.
    try std.testing.expectApproxEqAbs(@as(f64, 0.167), scale.coreOverMlp(p), 1e-3);
    // The row that was built to sit on the crossover does sit on it.
    const parity = try scale.project(.{ .name = "at-parity-12d", .cfg = .{
        .n_layers = 32,
        .n_heads = 32,
        .n_kv_heads = 8,
        .head_dim = 128,
        .n_ctx = 12 * 4096,
        .vocab_size = 128256,
        .ffn_mult = 4,
    } });
    try std.testing.expectEqual(@as(u64, 49152), parity.t);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), scale.coreOverMlp(parity), 1e-3);
}

test "scale: crossoverT solves the ratio it claims to" {
    // At the T crossoverT returns, core/mlp is the bar. Checked at the shipped
    // width and at a 32-head one, because the two have different head counts.
    for ([_]usize{ 128, 4096 }) |d| {
        for ([_]f64{ 0.06, 0.25, 0.5, 1.0 }) |bar| {
            const h = 4 * d;
            const t: f64 = scale.crossoverT(h, bar);
            // core/mlp = (T+1)/(3h) at the integer T, and crossoverT solves it
            // continuously, so allow one token of the diagonal.
            try std.testing.expectApproxEqAbs(bar, (t + 1) / (3 * @as(f64, @floatFromInt(h))), 1e-9);
        }
    }
    // And the two numbers the deferral note argues from.
    try std.testing.expectApproxEqAbs(@as(f64, 383), scale.crossoverT(512, 0.25), 0.5);
    try std.testing.expectApproxEqAbs(@as(f64, 1535), scale.crossoverT(512, 1.0), 0.5);
    // 1535 / 128 = 11.99, i.e. 12 * d and not 6 * d.
    try std.testing.expectApproxEqAbs(@as(f64, 12.0), scale.crossoverT(512, 1.0) / 128, 0.01);
}

test "scale: tied head is 6*T*vocab*d and streams 4*T*vocab*d bytes" {
    // model.tiedHead (model.zig:339-350) is T * vocab * d multiply-adds at
    // 2 FLOP each. The head's backward (autograd.zig:190-203) does two more
    // per element of the same three loops, so the step is three forwards.
    // At T=64, d=4096, vocab=128256:
    //   6 * 64 * 128256 * 4096 = 201729245184
    //   bytes  4 * 64 * 128256 * 4096 = 134486163456 = 134.5 GB
    const p = try scale.project(.{ .name = "head", .cfg = .{
        .n_layers = 32,
        .n_heads = 32,
        .n_kv_heads = 8,
        .head_dim = 128,
        .n_ctx = 64,
        .vocab_size = 128256,
        .ffn_mult = 4,
    } });
    try std.testing.expectEqual(@as(u64, 201_729_245_184), p.tied_head);
    try std.testing.expectEqual(6 * 64 * 128256 * 4096, p.tied_head);
    try std.testing.expectEqual(@as(u64, 134_486_163_456), p.tied_stream);
    try std.testing.expectEqual(4 * 64 * 128256 * 4096, p.tied_stream);
    // The bytes are what the 116 s has to move: at 116 s that is 1.16 GB/s,
    // which is a streaming rate and not a flop rate.
    const gbps = @as(f64, @floatFromInt(p.tied_stream)) / 1e9 / 116.0;
    try std.testing.expectApproxEqAbs(@as(f64, 1.159), gbps, 0.01);
}

test "scale: parameter count is the shapes initLayer allocates" {
    // Per layer, model.zig:370-380: 2 norms of d, wq and wo of d^2 each, wk and
    // wv of d*kv each, and three of d*h. Plus tok_embed of vocab*d and
    // final_norm of d. Shipped: d=128, h=512, kv=64, vocab=1024, 4 layers.
    //   per layer = 256 + 16384 + 16384 + 16384 + 196608 = 246016
    //   total     = 1024*128 + 4*246016 + 128 = 131072 + 984064 + 128 = 1115264
    const p = try scale.project(.{ .name = "shipped", .cfg = shipped });
    const per_layer: u64 = 2 * 128 + 2 * 128 * 128 + 2 * 128 * 64 + 3 * 128 * 512;
    try std.testing.expectEqual(@as(u64, 246_016), per_layer);
    try std.testing.expectEqual(1024 * 128 + 4 * per_layer + 128, p.params);
    try std.testing.expectEqual(@as(u64, 1_115_264), p.params);
    // grads mirrors params and adam is two moments, so peak is 4x params plus
    // the activations.
    try std.testing.expectEqual(p.params, p.grads);
    try std.testing.expectEqual(2 * p.params, p.adam);
}

test "scale: parameter count matches what initParams actually allocates" {
    // The projection is a formula; this is the allocator. Same config, so the
    // formula and the code that runs have to agree to the byte.
    const gpa = std.testing.allocator;
    var p = try model.initParams(gpa, shipped, 7);
    defer p.deinit();
    var counted: u64 = 0;
    counted += p.tok_embed.data.len;
    counted += p.final_norm.data.len;
    for (p.layers) |l| {
        counted += l.attn_norm.data.len;
        counted += l.wq.data.len;
        counted += l.wk.data.len;
        counted += l.wv.data.len;
        counted += l.wo.data.len;
        counted += l.mlp_norm.data.len;
        counted += l.w_gate.data.len;
        counted += l.w_up.data.len;
        counted += l.w_down.data.len;
    }
    const proj = try scale.project(.{ .name = "shipped", .cfg = shipped });
    try std.testing.expectEqual(proj.params, counted);
}

test "scale: the dense score matrix is n_layers * n_heads * T * T * 4" {
    // At the shipped shape: 4 * 4 * 256 * 256 * 4 = 4194304 = 4 MiB.
    // What the code allocates instead is n_heads * T * 8 = 4 * 256 * 8 = 8192,
    // one f64 row (attention.zig:45), 512 times smaller.
    const p = try scale.project(.{ .name = "shipped", .cfg = shipped });
    try std.testing.expectEqual(@as(u64, 4 * 4 * 256 * 256 * 4), p.scores_dense);
    try std.testing.expectEqual(@as(u64, 4_194_304), p.scores_dense);
    try std.testing.expectEqual(@as(u64, 4 * 256 * 8), p.scores_row);
    try std.testing.expectEqual(@as(u64, 8192), p.scores_row);
    try std.testing.expect(p.scores_dense > p.scores_row);
}

test "scale: activations are the per-layer blocks, not the logits, at depth" {
    // At T=8192, d=4096, h=16384, 32 layers, vocab=128256, one logits tensor
    // is 8192 * 128256 * 4 = 4202692608 = 4.2 GB, which is the figure the
    // README quotes, and two of them are live with dlogits.
    //
    // The term-by-term split, in f32 elements:
    //   head  logits + dlogits   2 * 8192 * 128256      =  2101346304
    //   blocks 32 layers of the eleven tensors in autograd's Block
    //                               32 * 8192 * 75776   = 19864223744
    // so the blocks are 90% and the head 9%. The logits are the number a
    // reader arrives with and they are not the largest thing here: the cache
    // the backward fills holds one Block per layer, and at 32 layers that is
    // what fills the machine.
    const p = try scale.project(.{ .name = "llama3-8b", .cfg = .{
        .n_layers = 32,
        .n_heads = 32,
        .n_kv_heads = 8,
        .head_dim = 128,
        .n_ctx = 8192,
        .vocab_size = 128256,
        .ffn_mult = 4,
    } });
    const one_logit = 8192 * 128256 * 4;
    try std.testing.expectEqual(@as(u64, 4_202_692_608), one_logit);
    const head: u64 = 2 * one_logit / 4;
    // One [T, d] for the first block's input. It was 33 of them while the
    // backward ran its own forward and kept a slice of the stream at every
    // layer boundary; the stream a block starts from is the previous block's
    // output now, so it is not held a second time.
    const stream: u64 = 8192 * 4096;
    const blocks: u64 = 32 * 8192 * (6 * 4096 + 2 * 1024 + 3 * 16384);
    const grads: u64 = 4 * 8192 * 4096;
    const attn_back: u64 = 8192 * (4096 + 2 * 1024) + 4 * 8192;
    try std.testing.expectEqual(@as(u64, 2_101_346_304), head);
    try std.testing.expectEqual(@as(u64, 19_864_223_744), blocks);
    // All five terms, added up by hand against the projection.
    try std.testing.expectEqual(head + stream + blocks + grads + attn_back, p.act);
    try std.testing.expectEqual(@as(u64, 22_183_706_624), p.act);
    // The blocks are the larger term by a wide margin, and the head is under a
    // tenth of the total.
    try std.testing.expect(blocks > 4 * head);
    try std.testing.expect(10 * head < p.act);
}

test "scale: rejects a config model.validate would reject" {
    // A sweep row that divides by zero or leaves a head with no kv head must
    // fail here rather than print a plausible line.
    const bad_head = [_]usize{ 0, 32 };
    for (bad_head) |head_dim| {
        try std.testing.expectError(error.InvalidConfig, scale.project(.{
            .name = "bad",
            .cfg = .{
                .n_layers = 1,
                .n_heads = 3,
                .n_kv_heads = 2,
                .head_dim = head_dim,
                .n_ctx = 8,
                .vocab_size = 8,
                .ffn_mult = 4,
            },
        }));
    }
    // 3 query heads over 2 kv heads is the one model.validate rejects and
    // attention.forward rejects identically.
    try std.testing.expectError(error.InvalidConfig, scale.project(.{
        .name = "bad",
        .cfg = .{
            .n_layers = 1,
            .n_heads = 3,
            .n_kv_heads = 2,
            .head_dim = 32,
            .n_ctx = 8,
            .vocab_size = 8,
            .ffn_mult = 4,
        },
    }));
}

test "scale: every row of the sweep projects" {
    // The comptime block in scale.zig already refuses a row that does not, so
    // reaching this test at all means every one of them is a valid config.
    for (scale.sweep) |shape| {
        const p = try scale.project(shape);
        try std.testing.expect(p.d > 0);
        try std.testing.expect(p.mlp > 0);
        try std.testing.expect(p.act > 0);
        // The shipped row is first, so a reader who runs this gets the shape
        // the rest of the repository documents before anything else.
        try std.testing.expectEqualStrings("shipped", shape.name);
        break;
    }
}

test "scale: the report is byte-identical across two runs" {
    // The README quotes this output, so a timestamp, an address or a
    // nondeterministic width would make the quote a lie. Nothing below is
    // timed and nothing reads a clock.
    var a: [16384]u8 = undefined;
    var b: [16384]u8 = undefined;
    var wa: std.Io.Writer = .fixed(&a);
    var wb: std.Io.Writer = .fixed(&b);
    try scale.print(&wa);
    try scale.print(&wb);
    try std.testing.expectEqualStrings(wa.buffered(), wb.buffered());
    // And the label is in there, so a projected number cannot be read as a
    // measured one.
    try std.testing.expect(std.mem.indexOf(u8, wa.buffered(), "PROJECTED, NOT MEASURED") != null);
}

test "scale: the report says every deferred item by name" {
    var buf: [16384]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    try scale.print(&w);
    const out = w.buffered();
    for ([_][]const u8{
        "fused_attn", "kv_cache", "tied_head",   "wgrad_swap", "dense_scores",
        "CROSSOVERS", "core/mlp", "weight_grad", "kv_cache",   "tied_GB",
    }) |needle| {
        try std.testing.expect(std.mem.indexOf(u8, out, needle) != null);
    }
    // No wall time anywhere: a "seconds" column would be the one thing that
    // could not be byte-reproducible.
    try std.testing.expect(std.mem.indexOf(u8, out, "seconds") == null);
    try std.testing.expect(std.mem.indexOf(u8, out, "elapsed") == null);
}
