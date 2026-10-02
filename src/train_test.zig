//! Tests for the training loop.
//!
//! The oracle for "the loop learns" is a loss that falls, measured on real
//! corpus bytes and on repeated text a tiny model memorises down to zero. The
//! oracle for "the loop is reproducible" is two runs of one seed writing
//! byte-identical CSVs. No expected loss value is a literal here: a loss is a
//! measurement on this host, so the assertions are about the margin a run
//! actually produced.
const std = @import("std");
const autograd = @import("autograd.zig");
const model = @import("model.zig");
const train = @import("train.zig");
const optim = @import("optim.zig");
const Tensor = @import("tensor.zig").Tensor;
const Tokenizer = @import("tokenizer.zig").Tokenizer;

const gpa = std.testing.allocator;
const io = std.testing.io;

const corpus_path = "data/tinyshakespeare.txt";
const corpus_bytes = 1_115_394;

/// Where the CSV tests write. `zig build test` and `zig test` both run with the
/// build root as the working directory, and `zig test` does not create
/// `zig-out`, so the test creates the directory and removes the file it wrote.
const scratch_dir = "zig-out/train_test";

/// 1 layer, 2 query heads over 1 kv head, so d_model is twice `head_dim` and
/// the GQA group is 2. The smallest shape that still holds a rotary pair and a
/// grouped head split. The 4 layer default is slow per test and proves no more
/// about the loop.
fn tiny_model(vocab: usize, n_ctx: usize, head_dim: usize, ffn_mult: usize) model.Config {
    return .{
        .n_layers = 1,
        .n_heads = 2,
        .n_kv_heads = 1,
        .head_dim = head_dim,
        .n_ctx = n_ctx,
        .vocab_size = vocab,
        .ffn_mult = ffn_mult,
    };
}

/// Tokens for `windows` whole windows at `ctx`, so a test states its own batch
/// arithmetic: `n * (ctx + 1)` tokens hold exactly `n` windows.
fn tokensFor(windows: usize, ctx: usize) usize {
    return windows * (ctx + 1);
}

/// A seeded token stream with ids inside `vocab`. Every test except the two
/// corpus ones uses this rather than the tokenizer, because the schedule, epoch
/// and clipping properties are about the loop, and a random stream has nothing
/// in it to memorise, so a pass cannot come from memorisation.
fn syntheticTokens(n: usize, vocab: usize, seed: u64) ![]u32 {
    const out = try gpa.alloc(u32, n);
    errdefer gpa.free(out);
    var prng = std.Random.Xoshiro256.init(seed);
    const rnd = prng.random();
    for (out) |*t| t.* = @intCast(rnd.uintLessThan(usize, vocab));
    return out;
}

fn readCorpusPrefix(len: usize) ![]u8 {
    // The limit is a ceiling, not a length: reading a file exactly at the limit
    // is reported as error.StreamTooLong, so it sits above the byte count.
    const whole = try std.Io.Dir.cwd().readFileAlloc(io, corpus_path, gpa, .limited(corpus_bytes + 1));
    defer gpa.free(whole);
    return gpa.dupe(u8, whole[0..len]);
}

fn readFileAt(path: []const u8) ![]u8 {
    return std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 20));
}

/// A path under a per-process scratch directory, created if it is not there.
/// The caller frees the path and defers `removeFile`.
///
/// The pid is in the directory name because these names are fixed strings. Two
/// copies of this suite in one tree — a Debug and a ReleaseFast binary, or two
/// shells running `zig test` — both wrote `zig-out/train_test/a.csv` and each
/// one's deferred `removeFile` took the other's file out from under it. The
/// failure surfaced as `FileNotFound` on a read that had already succeeded, and
/// it reproduced 5 times in 5 rounds when the same binary was run twice
/// concurrently. Serialising the two modes in the build hid it; the collision
/// was always there for anyone who ran the suite in two shells.
fn scratchPath(name: []const u8) ![]u8 {
    // One allocation, freed by the caller. Formatting the directory and the file
    // name separately would leak the directory string on the success path, which
    // the leak-detecting allocator this suite runs under reports as a test
    // failure rather than as noise.
    const path = try std.fmt.allocPrint(gpa, "{s}/{d}/{s}", .{ scratch_dir, std.c.getpid(), name });
    errdefer gpa.free(path);
    const dir = std.fs.path.dirname(path).?;
    _ = try std.Io.Dir.cwd().createDirPathStatus(io, dir, .default_dir);
    return path;
}

fn removeFile(path: []const u8) void {
    std.Io.Dir.cwd().deleteFile(io, path) catch {};
}

test "loss falls on a slice of the vendored corpus" {
    const text = try readCorpusPrefix(1_500);
    defer gpa.free(text);
    var tk = try Tokenizer.init(gpa);
    defer tk.deinit();
    try tk.train(text, 64);
    const ids = try tk.encode(gpa, text);
    defer gpa.free(ids);

    const ctx = 32;
    const cfg = train.Config{
        .model = tiny_model(tk.vocab.items.len, ctx, 4, 1),
        .epochs = 4,
        .ctx = ctx,
        // A d_model 8 model with a 0.02 initialisation needs a tenth of a percent
        // step to move its own weights in 100 steps, which is three to five times
        // what the same loop wants at the 4 layer default. Measured, not assumed:
        // 0.01 leaves the loss at 5.76 and 0.2 gives 4.57.
        .lr = 0.1,
        .warmup = 20,
        .weight_decay = 0.0,
        .max_grad_norm = 1.0,
        .seed = 7,
        .log_every = 1,
    };
    var res = try train.run(gpa, cfg, ids, ids);
    defer res.deinit();

    const first = res.rows[0].train_loss;
    const last = res.rows[res.rows.len - 1].train_loss;
    // A measurement, not a constant. 1500 corpus bytes, 64 merges, 852 tokens,
    // a 320 entry vocabulary, 100 steps at lr 0.1, seed 7: first logged step
    // 5.7585, final epoch mean 4.2523, drop 1.5062, val 4.2261. The assertion
    // asks for a third of the drop the run made, so a loop that stalls fails and
    // platform noise in the last decimals does not.
    if (last >= first - 0.4) {
        std.debug.print("\ncorpus: {d} bytes, {d} tokens, {d} steps, " ++
            "first {d:.4} final {d:.4} drop {d:.4} val {d:.4}\n", .{
            text.len, ids.len, res.steps, first, last, first - last, res.val_loss,
        });
        return error.LossDidNotFall;
    }
}

test "loss collapses to zero on a few hundred bytes of repeated text" {
    // 512 bytes of one repeating pair. The untrained tokenizer maps each byte to
    // itself, so the stream is two distinct ids in a fixed phase and the model
    // has to read the phase off the position. A trained tokenizer would merge
    // the pair into one id and hand the run a constant sequence, which a model
    // scores well without learning anything.
    const text: []const u8 = "ab" ** 256;
    var tk = try Tokenizer.init(gpa);
    defer tk.deinit();
    const ids = try tk.encode(gpa, text);
    defer gpa.free(ids);

    const ctx = 32;
    const cfg = train.Config{
        .model = tiny_model(tk.vocab.items.len, ctx, 8, 2),
        .epochs = 10,
        .ctx = ctx,
        .lr = 0.1,
        .warmup = 5,
        .weight_decay = 0.0,
        .max_grad_norm = 1.0,
        .seed = 3,
        .log_every = 1,
    };
    var res = try train.run(gpa, cfg, ids, ids);
    defer res.deinit();

    const first = res.rows[0].train_loss;
    const last = res.rows[res.rows.len - 1].train_loss;
    // A stream with two live tokens cannot report a mean below ln(2) = 0.6931 by
    // more than its own entropy, so a final loss under 0.01 is the model
    // predicting the right token every time and not the marginal. Measured over
    // 512 tokens in 150 steps at lr 0.1, seed 3: first 5.4971, final 2.725e-7,
    // val 2.715e-7. d_model 8 stalls at 0.6931 here however long it is given, so
    // the capacity is what this measures.
    if (!(last < 0.01 and last < first / 4.0)) {
        std.debug.print("\nrepeated ab: {d} tokens, {d} steps, first {d:.4} " ++
            "final {e:.3} val {e:.3}\n", .{ ids.len, res.steps, first, last, res.val_loss });
        return error.LossDidNotCollapse;
    }
}

test "two runs of one seed write a byte-identical csv" {
    const vocab = 32;
    const ctx = 32;
    const tokens = try syntheticTokens(tokensFor(12, ctx), vocab, 99);
    defer gpa.free(tokens);
    const val = try syntheticTokens(tokensFor(5, ctx), vocab, 5);
    defer gpa.free(val);

    const cfg = train.Config{
        .model = tiny_model(vocab, ctx, 4, 1),
        .epochs = 3,
        .ctx = ctx,
        .lr = 0.1,
        .warmup = 5,
        .weight_decay = 0.01,
        .max_grad_norm = 1.0,
        .seed = 7,
        .log_every = 1,
    };
    const path_a = try scratchPath("a.csv");
    defer gpa.free(path_a);
    defer removeFile(path_a);
    const path_b = try scratchPath("b.csv");
    defer gpa.free(path_b);
    defer removeFile(path_b);
    const path_c = try scratchPath("c.csv");
    defer gpa.free(path_c);
    defer removeFile(path_c);

    {
        var res = try train.run(gpa, cfg, tokens, val);
        defer res.deinit();
        try train.writeCsv(path_a, res.rows);
    }
    {
        var res = try train.run(gpa, cfg, tokens, val);
        defer res.deinit();
        try train.writeCsv(path_b, res.rows);
    }
    {
        var other = cfg;
        other.seed = 8;
        var res = try train.run(gpa, other, tokens, val);
        defer res.deinit();
        try train.writeCsv(path_c, res.rows);
    }

    const text_a = try readFileAt(path_a);
    defer gpa.free(text_a);
    const text_b = try readFileAt(path_b);
    defer gpa.free(text_b);
    const text_c = try readFileAt(path_c);
    defer gpa.free(text_c);

    // Compared as bytes rather than field by field, because the exit criterion
    // is that the file is the same file. A field-wise comparison would pass two
    // runs whose formatting drifted between epochs.
    try std.testing.expect(text_a.len > 0);
    try std.testing.expectEqualStrings(text_a, text_b);
    try std.testing.expect(!std.mem.eql(u8, text_a, text_c));
}

test "a parameter that received a gradient has moved" {
    const vocab = 32;
    const ctx = 32;
    const tokens = try syntheticTokens(tokensFor(1, ctx), vocab, 1);
    defer gpa.free(tokens);
    const val = try syntheticTokens(tokensFor(3, ctx), vocab, 2);
    defer gpa.free(val);

    const cfg = train.Config{
        .model = tiny_model(vocab, ctx, 4, 1),
        .epochs = 1,
        .ctx = ctx,
        .lr = 0.01,
        .warmup = 0,
        .weight_decay = 0.0,
        .max_grad_norm = 1.0,
        .seed = 11,
        .log_every = 1,
    };

    // `initParams` is a pure function of the config and the seed, so this is the
    // tensor `run` starts from and every difference below is one step's work.
    var before = try model.initParams(gpa, cfg.model, cfg.seed);
    defer before.deinit();

    var res = try train.run(gpa, cfg, tokens, val);
    defer res.deinit();

    // Every group, not one tensor: a loop that refilled the gradient buffer and
    // left one tensor out still moves the other ten, so a single tensor would
    // pass a half applied backward pass.
    try expectSomeElementMoved(&before.tok_embed, &res.params.tok_embed);
    for (before.layers, res.params.layers) |*b, *r| try expectLayerMoved(b, r);
    try expectSomeElementMoved(&before.final_norm, &res.params.final_norm);
}

test "a gradient cap below the norm stops the step, an uncapped one moves it by lr" {
    const vocab = 32;
    const ctx = 32;
    // One window, so the run is one step and the delta is one step's delta.
    const tokens = try syntheticTokens(tokensFor(1, ctx), vocab, 1);
    defer gpa.free(tokens);
    const val = try syntheticTokens(tokensFor(3, ctx), vocab, 2);
    defer gpa.free(val);
    const lr: f32 = 0.01;

    const base = train.Config{
        .model = tiny_model(vocab, ctx, 4, 1),
        .epochs = 1,
        .ctx = ctx,
        .lr = lr,
        .warmup = 0,
        // Zero, so the decoupled decay cannot move a parameter on its own and
        // the gradient step is the only thing under test.
        .weight_decay = 0.0,
        // Above every gradient norm this model produces, so nothing is clipped.
        .max_grad_norm = 1e9,
        .seed = 11,
        .log_every = 1,
    };
    var before = try model.initParams(gpa, base.model, base.seed);
    defer before.deinit();

    {
        var res = try train.run(gpa, base, tokens, val);
        defer res.deinit();
        // AdamW's first step on a gradient g is lr * g / (|g| + eps), which is lr
        // for any gradient not within eps of zero, so the largest change in the
        // tensor is lr to within eps/|g|. That is the step a run that never
        // clips has to show, and the cap in the block below is the only thing
        // that can take it away. Measured: moved 9.999873116612434e-3 at lr 1e-2,
        // short of lr by the eps/|g| the formula predicts.
        const moved = maxAbsDelta(&before.layers[0].w_gate, &res.params.layers[0].w_gate);
        try std.testing.expectApproxEqAbs(@as(f64, lr), moved, @as(f64, lr) * 0.001);
    }

    {
        // The same run with the cap 1e9 times smaller. Clipping scales every
        // gradient to 1e-30, AdamW's first step on that is lr * 1e-22, and the
        // f32 ulp of a weight of order 0.02 is 1e-9, so the tensor cannot move
        // at all. A loop that dropped the clip shows the same lr move here.
        var capped_cfg = base;
        capped_cfg.max_grad_norm = 1e-30;
        var res = try train.run(gpa, capped_cfg, tokens, val);
        defer res.deinit();
        try expectEqualBits(&before.layers[0].w_gate, &res.params.layers[0].w_gate);
    }
}

test "the gradient buffers are cleared between steps, so a step is one backward pass" {
    const vocab = 32;
    const ctx = 32;
    // One window and two epochs: the same batch twice, so both walks below see
    // the same tokens and the only thing that can move the two apart is what the
    // loop did to the gradient buffers between the steps.
    const tokens = try syntheticTokens(tokensFor(1, ctx), vocab, 1);
    defer gpa.free(tokens);
    const val = try syntheticTokens(tokensFor(3, ctx), vocab, 2);
    defer gpa.free(val);

    const cfg = train.Config{
        .model = tiny_model(vocab, ctx, 4, 1),
        .epochs = 2,
        .ctx = ctx,
        .lr = 0.1,
        .warmup = 0,
        .weight_decay = 0.0,
        // Above every gradient norm this model makes, so `clipByNorm` returns the
        // norm and scales nothing. This is a property of the hand walk, not a
        // loosening: the loop clips on a global norm over all eleven tensors, and
        // reproducing that scale by hand is a second thing to get right.
        .max_grad_norm = 1e9,
        .seed = 23,
        .log_every = 1,
    };
    var res = try train.run(gpa, cfg, tokens, val);
    defer res.deinit();
    try std.testing.expectEqual(cfg.epochs, res.steps);

    // The same two steps by hand: forward, backward, optimizer, and then this
    // step's gradient is gone before the next backward pass starts.
    //
    // The loss cannot see this. `backward` accumulates into whatever the buffers
    // hold, so a loop that never clears hands AdamW the sum of both steps on the
    // second one, and AdamW's m/sqrt(v) normalises an accumulated sum back to
    // about the same size step, so the curve falls either way. The weights are
    // where the difference is, compared bit for bit because both walks run the
    // same code on the same numbers in the same order.
    //
    // Every tensor is updated, not one. Step two's gradient is taken at the
    // weights step one left behind, so a walk that moved only `w_gate` would
    // diverge on a detail this test is not about.
    const inputs = tokens[0..ctx];
    const targets = tokens[1 .. ctx + 1];
    var p = try model.initParams(gpa, cfg.model, cfg.seed);
    defer p.deinit();
    var g = try autograd.zeroGrads(gpa, p);
    defer g.deinit();

    // One window per epoch, so the run's whole step count is its epoch count and
    // the schedule below is the one the loop computed.
    const total: u64 = @intCast(cfg.epochs);
    const states = try clearedWalk(p, &g, cfg, inputs, targets, total);
    defer {
        for (states) |*one| one.deinit();
        gpa.free(states);
    }
    try expectParamsEqualBits(&p, &res.params);
}

/// The two steps of the test above, driven by hand through the same optimizer
/// the loop uses, with the gradient buffers cleared after every step.
///
/// The flat view is built in the order `train.run` builds it, so the pairing of a
/// parameter with its gradient and the order the optimizer walks them are the
/// loop's own and not this file's guess at them.
fn clearedWalk(
    p: model.Params,
    g: *autograd.Grads,
    cfg: train.Config,
    inputs: []const u32,
    targets: []const u32,
    total: u64,
) ![]optim.AdamW {
    const n = 2 + 9 * p.layers.len;
    const flat_p = try gpa.alloc(Tensor, n);
    defer gpa.free(flat_p);
    const flat_g = try gpa.alloc(Tensor, n);
    defer gpa.free(flat_g);
    flatten(&p, g, flat_p, flat_g);

    const states = try gpa.alloc(optim.AdamW, n);
    errdefer gpa.free(states);
    // A half-built set of states is freed by the caller's loop only if it gets
    // all of them, so the ones that exist are the ones the errdefer hands back.
    var built: usize = 0;
    errdefer for (states[0..built]) |*one| one.deinit();
    while (built < n) : (built += 1) states[built] = try optim.AdamW.init(gpa, flat_p[built]);

    for (0..cfg.epochs) |at| {
        var logits = try model.forward(gpa, p, cfg.model, inputs);
        defer logits.deinit();
        var dlogits = try autograd.dLossDLogits(gpa, logits, targets);
        defer dlogits.deinit();
        try autograd.backward(gpa, p, g, cfg.model, inputs, dlogits);
        const lr = optim.cosineLR(@intCast(at), total, 0, cfg.lr);
        for (flat_p, flat_g, states) |*one, grad, *s| {
            try s.step(one, grad, lr, cfg.weight_decay);
        }
        // The line under test, at the same point in the step as the loop's.
        for (flat_g) |*one| one.fill(0);
    }
    return states;
}

/// Parameters and gradients in one fixed order: tok_embed, then each layer's
/// nine in field order, then final_norm. The same order `train.run` uses, so the
/// optimizer pairs each parameter with its own gradient here too.
fn flatten(p: *const model.Params, g: *autograd.Grads, ps: []Tensor, gs: []Tensor) void {
    std.debug.assert(ps.len == gs.len);
    ps[0] = p.tok_embed;
    gs[0] = g.tok_embed;
    for (p.layers, g.layers, 0..) |l, gl, i| {
        const at = 1 + 9 * i;
        ps[at + 0] = l.attn_norm;
        gs[at + 0] = gl.attn_norm;
        ps[at + 1] = l.wq;
        gs[at + 1] = gl.wq;
        ps[at + 2] = l.wk;
        gs[at + 2] = gl.wk;
        ps[at + 3] = l.wv;
        gs[at + 3] = gl.wv;
        ps[at + 4] = l.wo;
        gs[at + 4] = gl.wo;
        ps[at + 5] = l.mlp_norm;
        gs[at + 5] = gl.mlp_norm;
        ps[at + 6] = l.w_gate;
        gs[at + 6] = gl.w_gate;
        ps[at + 7] = l.w_up;
        gs[at + 7] = gl.w_up;
        ps[at + 8] = l.w_down;
        gs[at + 8] = gl.w_down;
    }
    ps[ps.len - 1] = p.final_norm;
    gs[gs.len - 1] = g.final_norm;
}

/// Two parameter sets, every tensor of both, bit for bit. Every tensor and not
/// one: a loop that cleared all but one of the buffers moves the cleared ten the
/// same way it moves the one left dirty, and a single comparison would not see
/// that.
fn expectParamsEqualBits(want: *const model.Params, got: *const model.Params) !void {
    try expectEqualBits(&want.tok_embed, &got.tok_embed);
    for (want.layers, got.layers) |*w, *gt| {
        try expectEqualBits(&w.attn_norm, &gt.attn_norm);
        try expectEqualBits(&w.wq, &gt.wq);
        try expectEqualBits(&w.wk, &gt.wk);
        try expectEqualBits(&w.wv, &gt.wv);
        try expectEqualBits(&w.wo, &gt.wo);
        try expectEqualBits(&w.mlp_norm, &gt.mlp_norm);
        try expectEqualBits(&w.w_gate, &gt.w_gate);
        try expectEqualBits(&w.w_up, &gt.w_up);
        try expectEqualBits(&w.w_down, &gt.w_down);
    }
    try expectEqualBits(&want.final_norm, &got.final_norm);
}

test "the csv lr column is the warmup then cosine schedule" {
    const vocab = 32;
    const ctx = 32;
    const total = 12;
    const epochs = 2;
    const warmup = 4;
    const lr: f32 = 0.1;
    // `total` is the whole run's step count and the schedule is set over it, so
    // the stream holds half of it per epoch and two epochs walk all twelve.
    const tokens = try syntheticTokens(tokensFor(total / epochs, ctx), vocab, 4);
    defer gpa.free(tokens);
    const val = try syntheticTokens(tokensFor(3, ctx), vocab, 6);
    defer gpa.free(val);

    const cfg = train.Config{
        .model = tiny_model(vocab, ctx, 4, 1),
        .epochs = epochs,
        .ctx = ctx,
        .lr = lr,
        .warmup = warmup,
        .weight_decay = 0.0,
        .max_grad_norm = 1.0,
        .seed = 21,
        .log_every = 1,
    };
    var res = try train.run(gpa, cfg, tokens, val);
    defer res.deinit();
    try std.testing.expectEqual(total, res.steps);

    const path = try scratchPath("lr.csv");
    defer gpa.free(path);
    defer removeFile(path);
    try train.writeCsv(path, res.rows);
    const text = try readFileAt(path);
    defer gpa.free(text);

    var lines = std.mem.splitScalar(u8, text, '\n');
    try std.testing.expectEqualStrings("step,train_loss,val_loss,lr", lines.next().?);

    var seen: usize = 0;
    while (lines.next()) |line| {
        if (line.len == 0) continue;
        var fields = std.mem.splitScalar(u8, line, ',');
        try std.testing.expectEqual(seen, try std.fmt.parseInt(usize, fields.next().?, 10));
        _ = try std.fmt.parseFloat(f64, fields.next().?);
        // The val column on a real run's own file: empty until the first epoch
        // ends and a number after it. This test is two epochs over six windows
        // with one row per step, so the boundary is the fifth row -- which is
        // what a run that never measured would break and a run that wrote a
        // zero would also break.
        const val_field = fields.next().?;
        if (seen < total / epochs - 1) {
            try std.testing.expectEqualStrings("", val_field);
        } else {
            _ = try std.fmt.parseFloat(f64, val_field);
        }
        const recorded = try std.fmt.parseFloat(f64, fields.next().?);
        try std.testing.expect(fields.next() == null);
        try std.testing.expect(seen < total);

        // The column carries eight decimals, so half of the last printed digit is
        // the most a round trip through the file can be off, and the hand formula
        // is a product of 0.1, which is a further ulp away.
        try std.testing.expectApproxEqAbs(
            @as(f64, optim.cosineLR(seen, total, warmup, lr)),
            recorded,
            1e-8,
        );
        if (seen < warmup) {
            // By hand, lr * (step + 1) / warmup: 0.025, 0.05, 0.075, 0.1.
            try std.testing.expectApproxEqAbs(
                @as(f64, lr) * (@as(f64, @floatFromInt(seen)) + 1.0) / @as(f64, @floatFromInt(warmup)),
                recorded,
                1e-8,
            );
        }
        seen += 1;
    }
    try std.testing.expectEqual(total, seen);

    // Rises across the warmup, holds one step at the peak, then falls. Steps
    // warmup - 1 and warmup are both exactly lr, so the peak is a plateau of two
    // and not a single maximum.
    for (0..total - 1) |i| {
        const at = recordedLrAt(text, i);
        const next = recordedLrAt(text, i + 1);
        if (i < warmup - 1) {
            try std.testing.expect(next > at);
        } else if (i == warmup - 1) {
            try std.testing.expectEqual(at, next);
        } else {
            try std.testing.expect(next < at);
        }
    }
    // The last step is progress 7/8 of the decay, so the rate is
    // lr * 0.5 * (1 + cos(0.875 pi)) = 0.0038 against a base of 0.1.
    try std.testing.expect(recordedLrAt(text, total - 1) < @as(f64, lr) / 10.0);
}

test "more epochs than the stream holds batches reshuffles and keeps going" {
    const vocab = 32;
    const ctx = 32;
    const per_epoch = 2;
    const epochs = 5;
    const tokens = try syntheticTokens(tokensFor(per_epoch, ctx), vocab, 8);
    defer gpa.free(tokens);
    const val = try syntheticTokens(tokensFor(3, ctx), vocab, 12);
    defer gpa.free(val);

    const cfg = train.Config{
        .model = tiny_model(vocab, ctx, 4, 1),
        .epochs = epochs,
        .ctx = ctx,
        .lr = 0.01,
        .warmup = 2,
        .weight_decay = 0.0,
        .max_grad_norm = 1.0,
        .seed = 13,
        .log_every = 1,
    };
    var res = try train.run(gpa, cfg, tokens, val);
    defer res.deinit();

    // Ten steps out of a stream that holds two: the run reshuffles and walks on
    // instead of ending at the first exhaustion.
    try std.testing.expectEqual(per_epoch * epochs, res.steps);
    try std.testing.expectEqual(per_epoch * epochs, res.rows.len);
    // The validation number lands on the last row of an epoch, so every row from
    // the first epoch end onward carries a measurement and only the rows before
    // it are unmeasured. Row 1 is where the first measurement goes, which is
    // also the check that the run measured something rather than walking past
    // it, and `null` on the earlier rows is the check that the run did not
    // write a zero it never took.
    for (res.rows, 0..) |row, i| {
        if (i < per_epoch - 1) {
            try std.testing.expectEqual(@as(?f64, null), row.val_loss);
        } else {
            try std.testing.expect(row.val_loss.? > 0.0);
        }
    }
    try std.testing.expectEqual(res.val_loss, res.rows[res.rows.len - 1].val_loss.?);
}

test "validation comes from the val tokens and nothing else" {
    const vocab = 32;
    const ctx = 32;
    const tokens = try syntheticTokens(tokensFor(6, ctx), vocab, 31);
    defer gpa.free(tokens);
    const val_a = try syntheticTokens(tokensFor(6, ctx), vocab, 77);
    defer gpa.free(val_a);
    const val_b = try syntheticTokens(tokensFor(6, ctx), vocab, 0);
    defer gpa.free(val_b);

    const cfg = train.Config{
        .model = tiny_model(vocab, ctx, 4, 1),
        .epochs = 2,
        .ctx = ctx,
        .lr = 0.1,
        .warmup = 2,
        .weight_decay = 0.0,
        .max_grad_norm = 1.0,
        .seed = 17,
        .log_every = 1,
    };
    var with_a = try train.run(gpa, cfg, tokens, val_a);
    defer with_a.deinit();
    var with_b = try train.run(gpa, cfg, tokens, val_b);
    defer with_b.deinit();

    // Swapping the validation stream cannot touch the training loss, so the two
    // training losses are the same bits. A run that drew its validation batches
    // from the training stream would report the same val loss here too.
    try std.testing.expectEqual(with_a.train_loss, with_b.train_loss);
    try std.testing.expect(with_a.val_loss != with_b.val_loss);
    try std.testing.expect(with_a.train_loss != with_a.val_loss);
}

test "the csv is a header and one line per row, and it round trips" {
    // Every value is a binary fraction, so the file holds exactly the doubles
    // the rows hold and the round trip below is an equality, not a tolerance.
    // The three rows carry all three states the val column can be in, so the
    // round trip covers the whole of the distinction in one pass: unmeasured,
    // measured and exactly zero, and measured and not.
    const rows = [_]train.Row{
        .{ .step = 0, .train_loss = 5.5, .val_loss = null, .lr = 0.025 },
        .{ .step = 1, .train_loss = 5.25, .val_loss = 0, .lr = 0.05 },
        .{ .step = 9, .train_loss = 1.125, .val_loss = 0.625, .lr = 0.1 },
    };
    const path = try scratchPath("rows.csv");
    defer gpa.free(path);
    defer removeFile(path);
    try train.writeCsv(path, &rows);
    const text = try readFileAt(path);
    defer gpa.free(text);

    try std.testing.expectEqualStrings(
        "step,train_loss,val_loss,lr\n" ++
            "0,5.500000,,0.02500000\n" ++
            "1,5.250000,0.000000,0.05000000\n" ++
            "9,1.125000,0.625000,0.10000000\n",
        text,
    );

    var lines = std.mem.splitScalar(u8, text, '\n');
    try std.testing.expectEqualStrings("step,train_loss,val_loss,lr", lines.next().?);
    var seen: usize = 0;
    while (lines.next()) |line| {
        if (line.len == 0) continue;
        try std.testing.expect(seen < rows.len);
        var fields = std.mem.splitScalar(u8, line, ',');
        try std.testing.expectEqual(rows[seen].step, try std.fmt.parseInt(usize, fields.next().?, 10));
        try std.testing.expectEqual(rows[seen].train_loss, try std.fmt.parseFloat(f64, fields.next().?));
        // The val column is the one optional field in the file, so it is read
        // back here the way a consumer reads it: an empty field is the absence
        // of a measurement and `0.000000` is the number zero. The test holds
        // the two apart rather than folding them, so a writer that rendered
        // both as one thing and a reader that parsed both as the other thing
        // would each fail here -- which is the whole point of the column, since
        // a round trip that cannot tell them apart proves nothing.
        const val_field = fields.next().?;
        const read_back: ?f64 = if (val_field.len == 0)
            null
        else
            try std.fmt.parseFloat(f64, val_field);
        try std.testing.expectEqual(rows[seen].val_loss, read_back);
        // The rate is the one column that does not come back exactly: it is an
        // f32, and eight decimals is fewer digits than an f32 near 0.025 needs,
        // so the file carries the f32 rounded to 1e-8 and the test reads it at
        // that resolution.
        try std.testing.expectApproxEqAbs(
            @as(f64, rows[seen].lr),
            try std.fmt.parseFloat(f64, fields.next().?),
            1e-8,
        );
        try std.testing.expect(fields.next() == null);
        seen += 1;
    }
    try std.testing.expectEqual(rows.len, seen);
}

test "a measured zero and an unmeasured cell are not the same file" {
    // THE NEGATIVE CONTROL, and the reason the column is optional at all. Two
    // curves that differ in nothing but this -- one measured a validation loss
    // of exactly zero, the other never measured one -- carry identical numbers
    // in every other column. A reader that folded the empty field onto zero
    // would report them as byte-for-byte the same curve, at an `absolute` of
    // zero, which is a claim about a run that never happened. So: the two
    // renderings are different bytes, and the comparison refuses the pair.
    const measured = [_]train.Row{
        .{ .step = 0, .train_loss = 5.5, .val_loss = 0, .lr = 0.025 },
    };
    const unmeasured = [_]train.Row{
        .{ .step = 0, .train_loss = 5.5, .val_loss = null, .lr = 0.025 },
    };

    const measured_path = try scratchPath("zero.csv");
    defer gpa.free(measured_path);
    defer removeFile(measured_path);
    try train.writeCsv(measured_path, &measured);
    const measured_text = try readFileAt(measured_path);
    defer gpa.free(measured_text);

    const unmeasured_path = try scratchPath("empty.csv");
    defer gpa.free(unmeasured_path);
    defer removeFile(unmeasured_path);
    try train.writeCsv(unmeasured_path, &unmeasured);
    const unmeasured_text = try readFileAt(unmeasured_path);
    defer gpa.free(unmeasured_text);

    // The whole column, both ways, so the distinction is pinned on the bytes a
    // reader sees rather than on the struct that produced them. `0.000000` is
    // six digits of a number that was measured; `,,` is a field with no
    // number in it, and the two lines differ in a character a person can see.
    try std.testing.expectEqualStrings(
        "step,train_loss,val_loss,lr\n0,5.500000,0.000000,0.02500000\n",
        measured_text,
    );
    try std.testing.expectEqualStrings(
        "step,train_loss,val_loss,lr\n0,5.500000,,0.02500000\n",
        unmeasured_text,
    );

    // `divergence` is the consumer that would have been fooled, so it is the
    // consumer that grades this. Both orders, because a check that only reads
    // one side of the pair is half a check.
    try std.testing.expectError(error.CurveShape, train.divergence(measured_text, unmeasured_text));
    try std.testing.expectError(error.CurveShape, train.divergence(unmeasured_text, measured_text));

    // Each file against itself is still a clean zero. Without this the two
    // refusals above would also be satisfied by a `divergence` that refused
    // every pair, which is a comparison that measures nothing.
    const zero_measured = try train.divergence(measured_text, measured_text);
    try std.testing.expectEqual(0.0, zero_measured.absolute);
    const zero_unmeasured = try train.divergence(unmeasured_text, unmeasured_text);
    try std.testing.expectEqual(0.0, zero_unmeasured.absolute);
    // A measured zero is a magnitude and raises the scale; a cell that was
    // never measured is not a loss at all and cannot. Same train loss on the
    // row either way, so this reads the val column and nothing else.
    try std.testing.expectEqual(5.5, zero_measured.max_loss);
    try std.testing.expectEqual(5.5, zero_unmeasured.max_loss);

    // An empty field is the absence of a measurement in the val column and
    // nowhere else. In either column that is logged on every row it is a
    // malformed curve, and accepting it would let a run whose losses were never
    // written compare clean against one that wrote them. Each is compared with
    // itself so that the row counts match and the parse is what refuses, rather
    // than the row count.
    for ([_][]const u8{
        "step,train_loss,val_loss,lr\n0,,0.000000,0.02500000\n",
        "step,train_loss,val_loss,lr\n0,5.500000,0.000000,\n",
        "step,train_loss,val_loss,lr\n0,5.500000,,,0.02500000\n",
    }) |malformed| {
        try std.testing.expectError(error.CurveShape, train.divergence(malformed, malformed));
    }
}

// The divergence tests below use the committed curve itself, read from the tree
// rather than written as a literal here. A literal in this file would be a second
// copy of the artifact that can drift from the first, and the point of the
// measurement is that it describes the curve that is actually committed.
// `zig build test` runs with the build root as the working directory, the same
// place `zig build train` writes.
//
// The val column in those fixtures is written the way `writeCsv` writes it, so
// that what was measured and what was not agrees with the committed curve. A
// fixture that spelled an unmeasured row `0.000000` would be a measured zero
// against a cell the committed curve never measured, which `divergence`
// refuses rather than compares -- and the refusal is the point, so the fixture
// has to be the shipped spelling for the comparison to mean anything.
const committed_curve = "outputs/loss.csv";

fn readCommittedCurve() ![]u8 {
    return std.Io.Dir.cwd().readFileAlloc(std.testing.io, committed_curve, gpa, .unlimited);
}

test "a curve compared with itself disagrees by nothing" {
    const text = try readCommittedCurve();
    defer gpa.free(text);
    const d = try train.divergence(text, text);

    try std.testing.expectEqual(0.0, d.absolute);
    // Read off the curve, not asserted as a restatement: 123 steps and a largest
    // logged loss of 6.489256 are what `outputs/loss.csv` holds, and the
    // measurement carries them so a reader can judge a disagreement as a
    // fraction of the run rather than only as an absolute number.
    try std.testing.expectEqual(123, d.steps);
    try std.testing.expectEqual(6.489256, d.max_loss);
}

test "a wrong gradient is measured at the step it appears, not merely as a total" {
    // The critic's experiment, as a fixture: `optim.AdamW.beta1` at 0.85 rather
    // than 0.9 moves step 24's loss from 6.489256 to 6.694191. Measured on the
    // host of record against the curve that host committed: 0.204935, the
    // largest disagreement on the curve, at the first logged row. The magnitude
    // and the step are both asserted because a report that named only "they
    // differ" would leave the reader to recompute the one thing that tells them
    // which kind of difference they are looking at.
    //
    // Both curves are the CUDA host's, re-measured rather than carried over. This
    // fixture was a Mac run, so against the committed curve it was measuring the
    // optimizer defect and the host at the same time, and the two tests in this
    // file that quote a "host difference" would each have been partly reporting
    // the other's. A fixture from one host and a committed curve from another
    // cannot separate two causes, so both are now the same host's.
    const text = try readCommittedCurve();
    defer gpa.free(text);
    const broken =
        "step,train_loss,val_loss,lr\n" ++
        "24,6.694191,,0.29888502\n" ++
        "49,6.110091,,0.24504843\n" ++
        "74,5.833082,,0.13857324\n" ++
        "99,5.683854,,0.03842629\n" ++
        "122,5.588424,5.143903,0.00006977\n";

    const d = try train.divergence(text, broken);
    try std.testing.expectEqual(@as(usize, 24), d.step);
    try std.testing.expectEqual(train.Column.train_loss, d.column);
    try std.testing.expectEqual(6.489256, d.committed);
    try std.testing.expectEqual(6.694191, d.fresh);
    try std.testing.expectApproxEqAbs(@as(f64, 0.204935), d.absolute, 1e-9);
    // The budget this repository used to gate on, kept here as a measured
    // distance rather than a gate: `steps * f32_epsilon * max_loss` on this
    // curve, 9.5e-5. 2153x under the broken run, and the Debug build below sits
    // inside the same two orders of magnitude. A number both cases clear by so
    // much is not a number that tells them apart, which is why nothing branches
    // on it any more.
    const budget = @as(f64, 123) * std.math.floatEps(f32) * 6.489256;
    try std.testing.expectApproxEqAbs(@as(f64, 0.0000951502905), budget, 1e-12);
    try std.testing.expect(d.absolute / budget > 1000);
}

test "a host difference is measured, and it is the same order as a broken run" {
    // This repository's two hosts: the Mac's curve against the CUDA host's
    // committed one, and the reason the budget was removed rather than
    // widened. It disagrees by 0.017975 at step 74 -- 189x the budget, so no
    // derived threshold separates them -- and its rows disagree by -31971,
    // -4635, +37696, -2288 and -6648 f32 ulp. Those are the ulp of a value
    // near 6, which is 2^-21 and four times what `floatEps(f32)` reports; the
    // budget below counts 2^-23 per step, and the two are not the same unit.
    // The sign changes, so the difference is trajectory chaos rather than
    // accumulated rounding, and it is unbounded in the step count. Asserted as
    // facts about the two cases because they are the whole argument, and
    // because a future change that reintroduced a threshold would be
    // reintroducing a constant these numbers refute.
    //
    // This fixture was captioned "a Debug build of the identical source on this
    // host", and it was neither that nor reproducible as one: `zig build
    // dbg-train` on this toolchain writes a curve byte-identical to the
    // ReleaseFast one, so `settleCsv` promotes it and leaves no pending file,
    // and there is no Debug curve to quote. What it actually held was the other
    // host's values, which is the difference this test's name claims. The
    // 0.017975 at step 74 and the 189x survive the correction exactly, because
    // they were always this comparison.
    const text = try readCommittedCurve();
    defer gpa.free(text);
    const host =
        "step,train_loss,val_loss,lr\n" ++
        "24,6.474011,,0.29888502\n" ++
        "49,6.059562,,0.24504843\n" ++
        "74,5.832430,,0.13857324\n" ++
        "99,5.686193,,0.03842629\n" ++
        "122,5.593455,5.154903,0.00006977\n";

    const d = try train.divergence(text, host);
    try std.testing.expectEqual(@as(usize, 74), d.step);
    try std.testing.expectApproxEqAbs(@as(f64, 0.017975), d.absolute, 1e-9);

    const budget = @as(f64, 123) * std.math.floatEps(f32) * 6.489256;
    try std.testing.expect(d.absolute / budget > 100);
    // The gap the reader is asked to judge by: a wrong optimizer constant is
    // 11.4x a host difference here, which is a measurement on two hosts, one
    // seed and one run length. Not a margin to gate on, and the code does not.
    try std.testing.expectApproxEqAbs(@as(f64, 0.204935 / 0.017975), 11.4, 0.05);
    // Step 24 is down, step 74 is up, and the largest of them is the up one, so
    // the two curves are not shifted one way.
    try std.testing.expect(d.fresh > d.committed);
    // WHERE the worst cell is, is a property of the two curves and not a
    // constant this file pins: constructed here rather than measured, this
    // curve departs only at the first logged row and is reported at step 24,
    // where the host difference above is reported at step 74.
    const early = try train.divergence(
        text,
        "step,train_loss,val_loss,lr\n" ++
            "24,7.000000,,0.29888502\n" ++
            "49,6.061772,,0.24504843\n" ++
            "74,5.814455,,0.13857324\n" ++
            "99,5.687284,,0.03842629\n" ++
            "122,5.596625,5.155396,0.00006977\n",
    );
    try std.testing.expectEqual(@as(usize, 24), early.step);
    try std.testing.expect(early.fresh > early.committed);
    try std.testing.expectEqual(@as(usize, 74), d.step);
}

test "a curve of a different shape is refused rather than half compared" {
    const text = try readCommittedCurve();
    defer gpa.free(text);

    // A row count that differs, which is what more or fewer epochs looks like.
    try std.testing.expectError(
        error.CurveShape,
        train.divergence(text, "step,train_loss,val_loss,lr\n24,6.474011,,0.29888502\n"),
    );
    // A header that is not the one `writeCsv` writes.
    try std.testing.expectError(
        error.CurveShape,
        train.divergence(text, "step,loss,val,lr\n24,6.474011,,0.29888502\n"),
    );
    // A row with a field that is not a number.
    try std.testing.expectError(
        error.CurveShape,
        train.divergence(text, "step,train_loss,val_loss,lr\n24,6.474011,,oops\n"),
    );
    // A row with a field too few, and one with a field too many: both are a
    // header and its rows disagreeing, and reading past the end of the former
    // would compare a train loss against whatever came next. The first carries
    // the unmeasured val field the shipped format writes, and the second a
    // measured one, so the two cases differ in the way that matters and still
    // have to be refused.
    try std.testing.expectError(
        error.CurveShape,
        train.divergence(text, "step,train_loss,val_loss,lr\n24,6.474011,\n"),
    );
    try std.testing.expectError(
        error.CurveShape,
        train.divergence(text, "step,train_loss,val_loss,lr\n24,6.474011,5.154903,0.29888502,1\n"),
    );
    // No rows at all: no step count and no scale, so there is nothing to measure
    // against, and an empty comparison would report a zero disagreement.
    try std.testing.expectError(
        error.CurveShape,
        train.divergence(text, "step,train_loss,val_loss,lr\n"),
    );
}

test "a full run and its csv leak nothing" {
    const vocab = 32;
    const ctx = 32;
    const tokens = try syntheticTokens(tokensFor(4, ctx), vocab, 41);
    defer gpa.free(tokens);
    const val = try syntheticTokens(tokensFor(4, ctx), vocab, 42);
    defer gpa.free(val);

    const path = try scratchPath("leak.csv");
    defer gpa.free(path);
    defer removeFile(path);
    {
        const cfg = train.Config{
            .model = tiny_model(vocab, ctx, 4, 1),
            .epochs = 2,
            .ctx = ctx,
            .lr = 0.1,
            .warmup = 2,
            .weight_decay = 0.01,
            .max_grad_norm = 0.5,
            .seed = 19,
            .log_every = 1,
        };
        // gpa is std.testing.allocator, so the loop allocating per step and
        // missing a free on any path fails this test when it returns.
        var res = try train.run(gpa, cfg, tokens, val);
        defer res.deinit();
        try train.writeCsv(path, res.rows);
    }
    const text = try readFileAt(path);
    defer gpa.free(text);
    try std.testing.expect(text.len > 0);
}

test "a non-finite weight decay is refused, so it cannot poison the weights" {
    const vocab = 32;
    const ctx = 16;
    const tokens = try syntheticTokens(tokensFor(4, ctx), vocab, 51);
    defer gpa.free(tokens);
    const val = try syntheticTokens(tokensFor(4, ctx), vocab, 52);
    defer gpa.free(val);

    const cfg = train.Config{
        .model = tiny_model(vocab, ctx, 4, 1),
        .epochs = 1,
        .ctx = ctx,
        .lr = 0.1,
        .warmup = 2,
        // The one Config float the loop never checked. Decoupled decay
        // multiplies the rate by it, so a NaN lands in every weight on the
        // first step and every later step then reads a NaN back.
        .weight_decay = std.math.nan(f32),
        .max_grad_norm = 1.0,
        .seed = 3,
        .log_every = 1,
    };
    try std.testing.expectError(error.BadWeightDecay, train.run(gpa, cfg, tokens, val));
}

test "a run that overflows stops instead of logging non-finite rows forever" {
    const vocab = 32;
    const ctx = 16;
    const tokens = try syntheticTokens(tokensFor(4, ctx), vocab, 61);
    defer gpa.free(tokens);
    const val = try syntheticTokens(tokensFor(4, ctx), vocab, 62);
    defer gpa.free(val);

    // A huge but finite rate, which is the one way to reach a non-finite
    // gradient without editing a module that is not this test's: the weights
    // overflow on the first step, so the second step's forward and backward
    // are non-finite. The loop has to stop there. It used to keep going, and
    // every row it wrote from then on was a NaN.
    const cfg = train.Config{
        .model = tiny_model(vocab, ctx, 4, 1),
        .epochs = 2,
        .ctx = ctx,
        .lr = 3.0e38,
        .warmup = 0,
        .weight_decay = 0.0,
        .max_grad_norm = 1.0,
        .seed = 4,
        .log_every = 1,
    };
    // The guard that fires is the one on the loss, because the loss goes
    // non-finite on the same step the weights do. Whichever guard catches it
    // first, the run ends in an error and never returns a Result.
    const outcome = train.run(gpa, cfg, tokens, val);
    if (outcome) |*res| {
        // A run that completed is the bug: either it returned NaN rows, or it
        // returned finite rows, which would mean the overflow never happened
        // and the test is not testing what it says it is.
        var owned = res.*;
        defer owned.deinit();
        for (owned.rows) |row| {
            try std.testing.expect(std.math.isFinite(row.train_loss));
            // An unmeasured cell is not a non-finite one: the rows before the
            // first epoch ends carry no val loss at all, and saying so is what
            // they are for.
            try std.testing.expect(row.val_loss == null or std.math.isFinite(row.val_loss.?));
        }
        // Finite rows out of an overflowing run means the numbers are simply
        // wrong, so there is nothing to assert about them.
        return error.OverflowRunCompletedWithFiniteLoss;
    } else |err| {
        // The one error, not either of two. A disjunction here passed whether
        // the loss guard or the gradient guard caught it, and those are different
        // defects at different lines in the loop: the loss going non-finite
        // makes the gradient non-finite on the same step, so deleting the loss
        // guard outright still satisfied the old assertion. The comment above
        // says which guard is under test, so this names it.
        //
        // `NonFiniteLogits`, not `NonFiniteLoss`, because the guard MOVED rather
        // than changed: `loss.forward` used to return the NaN and `train.run:187`
        // caught it afterwards, which left the other two callers of `loss.forward`
        // -- `gradcheck`'s central differences and `train.zig:488`'s seam gate --
        // consuming it silently. The guard now sits inside `loss.forward`, so it
        // names the cause rather than the symptom and fires for all three callers.
        // With every logit finite the logsumexp cannot produce a non-finite loss:
        // `max` is the row max, `sum` is at least 1 and at most `v_count`, and
        // `total` sums at most `t_count` terms each bounded by the f32 range, so
        // `train.run:187` is defence in depth against a future change to that
        // arithmetic rather than the guard that catches this overflow.
        try std.testing.expectEqual(error.NonFiniteLogits, err);
    }
}

test "a path that cannot be written is an error, not a panic" {
    try std.testing.expectError(
        error.FileNotFound,
        train.writeCsv("no_such_dir_9f2b/curve.csv", &.{}),
    );
}

fn recordedLrAt(text: []const u8, step: usize) f64 {
    var lines = std.mem.splitScalar(u8, text, '\n');
    _ = lines.next();
    var i: usize = 0;
    while (lines.next()) |line| : (i += 1) {
        if (line.len == 0) continue;
        if (i != step) continue;
        var fields = std.mem.splitScalar(u8, line, ',');
        _ = fields.next();
        _ = fields.next();
        _ = fields.next();
        return std.fmt.parseFloat(f64, fields.next().?) catch unreachable;
    }
    unreachable;
}

fn maxAbsDelta(before: *const Tensor, after: *const Tensor) f64 {
    var biggest: f64 = 0;
    for (before.data, after.data) |a, b| {
        biggest = @max(biggest, @abs(@as(f64, b) - @as(f64, a)));
    }
    return biggest;
}

fn expectSomeElementMoved(before: *const Tensor, after: *const Tensor) !void {
    try std.testing.expectEqual(before.data.len, after.data.len);
    for (before.data, after.data) |a, b| {
        if (@as(u32, @bitCast(a)) != @as(u32, @bitCast(b))) return;
    }
    return error.NoParameterMoved;
}

fn expectEqualBits(before: *const Tensor, after: *const Tensor) !void {
    try std.testing.expectEqual(before.data.len, after.data.len);
    for (before.data, after.data) |a, b| {
        try std.testing.expectEqual(@as(u32, @bitCast(a)), @as(u32, @bitCast(b)));
    }
}

fn expectLayerMoved(before: *const model.Layer, after: *const model.Layer) !void {
    const pairs = [9][2]*const Tensor{
        .{ &before.attn_norm, &after.attn_norm },
        .{ &before.wq, &after.wq },
        .{ &before.wk, &after.wk },
        .{ &before.wv, &after.wv },
        .{ &before.wo, &after.wo },
        .{ &before.mlp_norm, &after.mlp_norm },
        .{ &before.w_gate, &after.w_gate },
        .{ &before.w_up, &after.w_up },
        .{ &before.w_down, &after.w_down },
    };
    for (pairs) |pair| try expectSomeElementMoved(pair[0], pair[1]);
}
