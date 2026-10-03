//! The training loop: batch, forward, loss, hand-written backward, clip, rate,
//! AdamW, clear. In that order, every step.
//!
//! Reproducibility is the property this module exists to hold. One seed on one
//! host produces one CSV byte for byte: the batch order is a named Xoshiro256
//! permutation in `data.zig`, every reduction inside the numerics walks a fixed
//! order, the two loss accumulators are f64 added in step order, and nothing
//! here depends on a hash table's iteration order or on a second thread.
//!
//! Gradients come from `autograd.backward` and nowhere else. There is no tape
//! and no second derivation, so a run that trains is a run whose loss curve was
//! produced by the same code `gradcheck.zig` differences against finite
//! differences.
const std = @import("std");
const model = @import("model.zig");
const autograd = @import("autograd.zig");
const data = @import("data.zig");
const profile = @import("profile.zig");
const loss = @import("loss.zig");
const optim = @import("optim.zig");
const tensor = @import("tensor.zig");
const attention = @import("attention.zig");
const Tensor = tensor.Tensor;

pub const Config = struct {
    model: model.Config,
    epochs: usize,
    ctx: usize,
    lr: f32,
    warmup: usize,
    weight_decay: f32,
    max_grad_norm: f32,
    seed: u64,
    log_every: usize,
};

/// One logged step. `writeCsv` turns a run's rows into the loss curve.
pub const Row = struct {
    /// 0-based index of the optimizer step, which is also the schedule step, so
    /// `optim.cosineLR(row.step, ...)` is the rate the run used here.
    step: usize,
    /// Mean training loss over the steps of the epoch this row falls in, one
    /// f64 accumulator in step order and restarted at every epoch. A mean and
    /// not the single step's loss, because one batch of a reshuffled stream is
    /// the noisiest number in the file, and per epoch rather than per run so
    /// the column answers how the model is doing now instead of trailing the
    /// plateau the run started on.
    train_loss: f64,
    /// Validation loss of the most recent epoch end, so the last row of every
    /// epoch carries the number measured after it and the rows before the first
    /// epoch ends are `null`.
    ///
    /// `null`, not `0`. It used to be an `f64` initialised to 0, and every row
    /// logged before the first epoch ended printed `0.000000` -- a file in
    /// which "the validation loss was exactly zero" and "nothing has been
    /// measured yet" are the same bytes, so a reader cannot tell which of the
    /// two it is looking at. `writeCsv` prints an empty field for `null` and
    /// `cell` reads one back, so the file says what it means. A measured `0.0`
    /// still prints `0.000000` and still parses as a number, which is the whole
    /// difference.
    val_loss: ?f64,
    lr: f32,
};

pub const Result = struct {
    /// Mean training loss over every step of the run.
    train_loss: f64,
    /// Validation loss of the last epoch.
    val_loss: f64,
    steps: usize,
    /// The trained weights. The loop allocates these from `allocator` and the
    /// caller frees them, so a checkpoint can be written straight out of a run.
    params: model.Params,
    rows: []Row,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *Result) void {
        self.params.deinit();
        self.allocator.free(self.rows);
        self.* = undefined;
    }
};

/// Validation batches measured at every epoch end.
///
/// A fixed count, not the whole stream, so the number is comparable epoch to
/// epoch and a val loss does not move when the corpus does. Eight is enough to
/// average a batch's noise down and small enough to stay a rounding error next
/// to a training step.
const val_batches: usize = 8;

/// Trains for `cfg.epochs` epochs over `train_tokens` and reports the mean
/// training loss, the final validation loss measured on `val_tokens`, and the
/// per-logged-step log.
///
/// The two streams are walked by two batchers over two token slices and share
/// nothing but the parameters, so a validation loss is never a number the run
/// optimised against.
pub fn run(
    allocator: std.mem.Allocator,
    cfg: Config,
    train_tokens: []const u32,
    val_tokens: []const u32,
) !Result {
    // Every test is a negated comparison so a NaN is rejected too: NaN fails
    // each `>` it meets, and a NaN rate or cap then poisons every weight in the
    // run without a single error.
    if (!(cfg.lr > 0)) return error.BadLr;
    if (!(cfg.max_grad_norm > 0)) return error.BadMaxGradNorm;
    // Weight decay is a multiplier on the rate, so a NaN here lands in every
    // weight on the first step and the run then reads a NaN back on every step
    // after it. It needs the same guard as the two above, and the loop's own
    // finiteness check below is the second line of defence, not the first.
    if (!(cfg.weight_decay >= 0)) return error.BadWeightDecay;
    if (cfg.log_every == 0) return error.BadLogEvery;
    if (cfg.epochs == 0) return error.NoEpochs;
    // Every epoch walks every window, so a training stream shorter than one
    // window has nothing to train on and no loss to report.
    if (train_tokens.len < cfg.ctx + 1) return error.NoTrainingBatches;

    var params = try model.initParams(allocator, cfg.model, cfg.seed);
    errdefer params.deinit();
    var grads = try autograd.zeroGrads(allocator, params);
    defer grads.deinit();

    // The optimizer and the clipper both want one flat view of the parameters,
    // and the shapes of the optimizer states are read off it, so it is built
    // before the states and refreshed on every step.
    const n_tensors = 2 + 9 * cfg.model.n_layers;
    const flat_params = try allocator.alloc(Tensor, n_tensors);
    defer allocator.free(flat_params);
    const flat_grads = try allocator.alloc(Tensor, n_tensors);
    defer allocator.free(flat_grads);
    flatten(&params, &grads, flat_params, flat_grads);

    var states = try allocator.alloc(optim.AdamW, n_tensors);
    var built: usize = 0;
    // `built` is read when the scope exits, so the unwind frees the prefix that
    // exists rather than the whole slice of half-built states.
    defer {
        for (states[0..built]) |*one| one.deinit();
        allocator.free(states);
    }
    while (built < n_tensors) : (built += 1) {
        states[built] = try optim.AdamW.init(allocator, flat_params[built]);
    }

    var batcher = try data.Batcher.init(allocator, train_tokens, cfg.ctx, cfg.seed);
    defer batcher.deinit();
    // Its own batcher over its own tokens, never the training one. It reshuffles
    // per epoch like the training one, and it is seeded with the same `cfg.seed`,
    // so the two permutations are the same sequence rather than two independent
    // draws. The guarantee that matters does not depend on that: the two batchers
    // index disjoint token slices, so no validation token is ever trained on
    // whatever order either of them picks.
    var val_batcher = try data.Batcher.init(allocator, val_tokens, cfg.ctx, cfg.seed);
    defer val_batcher.deinit();
    if (val_batcher.order.len == 0) return error.EmptyValidation;

    var rows: std.ArrayList(Row) = .empty;
    // A no-op after `toOwnedSlice` hands the buffer over, and the only thing
    // that frees the list if any step below fails.
    defer rows.deinit(allocator);

    const windows = batcher.order.len;
    // The schedule is set over the whole run, so `total` is known before the
    // first step and the rate never depends on how far the loop has got.
    const total: u64 = @intCast(windows * cfg.epochs);
    const warmup: u64 = @intCast(cfg.warmup);
    var step: usize = 0;
    // Two accumulators, both f64 and both added in step order: the run's, which
    // `Result.train_loss` reports, and the epoch's, which the log column carries.
    var run_loss_sum: f64 = 0;
    var epoch_loss_sum: f64 = 0;
    // `null` until the first epoch ends, so a row logged before that carries
    // "not measured" and not a zero.
    var val_loss: ?f64 = null;
    var in_epoch: usize = 0;

    for (0..cfg.epochs) |_| {
        while (true) {
            // FIRST probe of the iteration, before anything is allocated. Every
            // `defer` in this body fires at the closing brace, which is AFTER the
            // last probe below, so without this row the previous step's loop
            // -- `Cache.deinit` alone frees eleven tensors per block -- would
            // land in `fetch`, and the row would be named after something other
            // than what it measures.
            if (profile.active) |pr| pr.stop(.loop);
            var batch = (try batcher.next()) orelse break;
            defer batch.deinit();
            // Each probe closes the gap since the previous one and names what
            // it was spent on, so the reading that ends one op opens the next
            // and nothing is counted twice. `active` is null unless
            // `ZTRANSFORMER_PROFILE=1`, so the committed curve takes a null check
            // and no clock read at all.
            if (profile.active) |pr| pr.stop(.fetch);

            // The one forward pass, handing its intermediates to the backward
            // pass as they are produced. `model.forward` plus a backward that
            // rebuilt the same tensors was two forwards per step, and the
            // rebuild was the larger half of the step.
            var cache = try autograd.Cache.init(allocator, params, cfg.model, batch.inputs);
            defer cache.deinit();
            var logits = try model.forwardWith(allocator, params, cfg.model, batch.inputs, &cache.sink);
            defer logits.deinit();
            if (profile.active) |pr| pr.stop(.forward);
            const batch_loss = try loss.forward(logits, batch.targets);
            if (profile.active) |pr| pr.stop(.loss);
            // Checked before the loss is used, not after. A non-finite loss is
            // already a non-finite gradient, and the row it would be logged
            // into is a NaN that reads like a measurement. Stopping here is
            // what keeps one bad step from writing NaN into every weight and
            // logging a NaN curve for the rest of the run.
            if (!std.math.isFinite(batch_loss)) return error.NonFiniteLoss;
            run_loss_sum += batch_loss;
            epoch_loss_sum += batch_loss;
            var dlogits = try autograd.dLossDLogits(allocator, logits, batch.targets);
            defer dlogits.deinit();
            if (profile.active) |pr| pr.stop(.dlogits);
            try autograd.backwardFrom(allocator, params, &grads, cfg.model, batch.inputs, dlogits, &cache);
            if (profile.active) |pr| pr.stop(.backward);

            // One global norm over every tensor, so the cap is the whole model's
            // and not a per-tensor budget. The norm is checked but not logged:
            // the log carries the four columns below and a fifth would be a
            // column nothing reads.
            //
            // `clipByNorm` cannot scale a non-finite norm: a NaN fails its
            // `norm > max_norm` test, so the gradients leave the clip exactly
            // as NaN and the step below would spread them across every
            // parameter. Caught here, where the run can still be abandoned with
            // the weights intact.
            const norm = optim.clipByNorm(flat_grads, cfg.max_grad_norm);
            if (profile.active) |pr| pr.stop(.clip);
            if (!std.math.isFinite(norm)) return error.NonFiniteGradient;
            const at = step;
            const lr = optim.cosineLR(@intCast(at), total, warmup, cfg.lr);
            for (flat_params, flat_grads, states) |*p, g, *s| try s.step(p, g, lr, cfg.weight_decay);
            if (profile.active) |pr| pr.stop(.adam);
            // `backward` accumulates, so the buffers are cleared every step and
            // not once per epoch: otherwise every step after the first adds its
            // gradient to the whole history behind it.
            //
            // Zeroed in place through the flat view, which already names every
            // gradient buffer. Allocating a fresh set here would churn about
            // 5 MB per step at the 4 layer default, and it would also leave the
            // flat view pointing at freed memory, which is what forced a second
            // `flatten` on every step to rebuild it. Clearing the bytes the
            // view already names removes both problems, and cannot fail.
            for (flat_grads) |*g| g.fill(0);
            if (profile.active) |pr| pr.stop(.zero);

            step = at + 1;
            in_epoch += 1;
            // The last step of an epoch is always logged, because that is where
            // the validation number lands and a row per `log_every` steps alone
            // would leave a short epoch with no row to carry it.
            if (in_epoch % cfg.log_every == 0 or in_epoch == windows) {
                try rows.append(allocator, .{
                    .step = at,
                    .train_loss = epoch_loss_sum / @as(f64, @floatFromInt(in_epoch)),
                    .val_loss = val_loss,
                    .lr = lr,
                });
            }
        }

        // The stream is exhausted, so the epoch is over: measure validation,
        // put it on the row the epoch ended on, and reshuffle for the next one.
        val_loss = try evalLoss(allocator, params, cfg.model, &val_batcher);
        if (profile.active) |p| p.stop(.eval);
        rows.items[rows.items.len - 1].val_loss = val_loss;
        val_batcher.reset();
        batcher.reset();
        in_epoch = 0;
        epoch_loss_sum = 0;
    }

    return .{
        .train_loss = run_loss_sum / @as(f64, @floatFromInt(step)),
        // The run's own number rather than the rows': the `epochs == 0` guard at
        // the top means the loop above ran and measured at least once, so this
        // is not an `orelse` that could hand back a zero the loop never took.
        .val_loss = val_loss.?,
        .steps = step,
        .params = params,
        .rows = try rows.toOwnedSlice(allocator),
        .allocator = allocator,
    };
}

/// The header `writeCsv` writes and `divergence` reads. One spelling, so a
/// column cannot be renamed in one place and looked for in another.
pub const csv_header = "step,train_loss,val_loss,lr";

/// A step, and the three numbers logged at it in the order `csv_header` names.
///
/// `values[1]` is optional because the column can carry an empty field. The
/// other two are logged on every row, so their cells are never `null` and
/// unwrapping them is not a guess.
const Cell = struct {
    step: usize,
    values: [3]?f64,
};

/// Which logged column a disagreement was found in. The order is `csv_header`'s,
/// because that is the order `row` fills `Cell.values` in.
pub const Column = enum {
    train_loss,
    val_loss,
    lr,
};

/// How far apart two runs' curves are, and where: the largest disagreement over
/// every cell, and the row and column it is at.
///
/// A measurement, deliberately not a verdict. There was a budget here once —
/// `steps * f32_epsilon * max_loss`, one f32 ulp per step added, which is what
/// `src/README.md` documents — and it was refuted by this repository's own two
/// cases. `optim.AdamW.beta1` at 0.85 lands 1667x outside it. So does a Debug
/// build of the identical source, at 189x, and the run is fine: its successive
/// rows disagree by 39507, 6119, -51706, 3219 and 9508 ulp, changing sign,
/// because a one-ulp difference in an early weight moves every weight after it.
/// That is trajectory chaos and it is unbounded in the step count, so an additive
/// bound cannot describe it, and any threshold separating the two cases would be
/// a number fitted to this host's seed, libm and run length.
///
/// What is left is what a reader needs to judge the difference themselves, which
/// is the number itself and the step it is at. The caller decides what it means.
pub const Divergence = struct {
    /// Largest absolute disagreement over every cell, 0 for identical curves.
    absolute: f64,
    /// The row and column that disagree most, and both sides of it.
    step: usize,
    column: Column,
    committed: f64,
    fresh: f64,
    /// The size of the curve it was measured on, so a disagreement can be read
    /// as a fraction of the run rather than only as an absolute number.
    steps: usize,
    max_loss: f64,
};

/// Measures how far `fresh` is from `committed`, cell by cell.
///
/// `error.CurveShape` when the two are not the same shape of file: a different
/// header, a different number of rows, a field that is not a number, or one
/// cell that was measured on one side and not on the other. All four are
/// refused rather than tolerated, because a curve this cannot read is not one
/// it can measure, and reading past the end of a row is how a difference gets
/// attributed to the wrong column and reported as rounding.
///
/// THE UNMEASURED CELL, which is the fourth refusal and the only one this
/// signature could not have had before `val_loss` became optional. Two cells
/// that were both never measured agree, and contribute a difference of zero
/// like any other pair of equal numbers. One that was measured and one that was
/// not have no difference between them to report: there is no number on one
/// side of the subtraction, and treating the absent one as 0 would be to read
/// a file this module has already refused to read. That is the defect this
/// closes, and it is why the answer is an error rather than a sentinel. A
/// sentinel would have to be a number, every caller would have to know to
/// disregard it, and a caller that did not would report a run that measured
/// its validation loss at some point in one curve and never in the other as
/// agreeing to within a rounding error -- which is exactly the claim the
/// `0.000000` file used to make, one layer down.
///
/// Nothing allocates: both curves are read into fixed arrays, and `max_rows` is
/// refused rather than grown, so a longer curve fails instead of being silently
/// compared only in part.
pub fn divergence(committed: []const u8, fresh: []const u8) !Divergence {
    var a: [max_rows]Cell = undefined;
    var b: [max_rows]Cell = undefined;
    const n = try parse(committed, &a);
    if (try parse(fresh, &b) != n) return error.CurveShape;

    var steps: usize = 0;
    var max_loss: f64 = 0;
    for (a[0..n]) |c| {
        steps = @max(steps, c.step + 1);
        // The two loss columns, not `lr`: the budget is in loss units. A cell
        // that was never measured is not a magnitude, so it does not raise the
        // scale of the curve; a curve with nothing measured in it is smaller
        // than one that read a number.
        max_loss = @max(
            max_loss,
            @abs(c.values[0].?),
            if (c.values[1]) |v| @abs(v) else 0.0,
        );
    }

    var worst: Divergence = .{
        .absolute = 0,
        .step = a[0].step,
        .column = .train_loss,
        .committed = a[0].values[0].?,
        .fresh = a[0].values[0].?,
        .steps = steps,
        .max_loss = max_loss,
    };
    for (a[0..n], b[0..n]) |ca, cb| {
        for (ca.values, cb.values, 0..) |va, vb, i| {
            if ((va == null) != (vb == null)) return error.CurveShape;
            // Both unmeasured is agreement, the same as two equal numbers:
            // there is a magnitude on neither side to subtract. Unwrapping
            // here panicked on every curve whose `val_loss` is empty, which
            // is four of the five rows the shipped run writes.
            if (va == null) continue;
            const d = @abs(va.? - vb.?);
            if (d > worst.absolute) {
                worst.absolute = d;
                worst.step = ca.step;
                worst.column = switch (i) {
                    0 => .train_loss,
                    1 => .val_loss,
                    else => .lr,
                };
                worst.committed = va.?;
                worst.fresh = vb.?;
            }
        }
    }
    return worst;
}

/// A curve longer than this is refused. The shipped run logs five rows; the cap
/// is a stack array rather than an allocation, so that `divergence` needs no
/// allocator argument and cannot leak one.
const max_rows = 256;

fn parse(text: []const u8, out: *[max_rows]Cell) !usize {
    var lines = std.mem.splitScalar(u8, text, '\n');
    const header = lines.next() orelse return error.CurveShape;
    if (!std.mem.eql(u8, header, csv_header)) return error.CurveShape;
    var n: usize = 0;
    while (lines.next()) |line| {
        // The newline the last row ends on leaves one empty field behind.
        if (line.len == 0) continue;
        if (n == out.len) return error.CurveShape;
        out[n] = try cell(line);
        n += 1;
    }
    // A curve with no rows has no step count and no scale, so there is nothing
    // to measure against and nothing for a budget to be computed from.
    if (n == 0) return error.CurveShape;
    return n;
}

fn cell(line: []const u8) !Cell {
    var fields = std.mem.splitScalar(u8, line, ',');
    var c: Cell = undefined;
    c.step = std.fmt.parseInt(usize, fields.next() orelse return error.CurveShape, 10) catch
        return error.CurveShape;
    for (&c.values, 0..) |*v, i| {
        const text = fields.next() orelse return error.CurveShape;
        // An empty field is "not measured", and it is `val_loss` alone that is
        // allowed to be one: every row is logged with a train loss and a rate,
        // so an empty field in either of those is a malformed row rather than
        // a measurement that was not taken, and reading it as the latter would
        // accept a curve whose losses were never written. A measured zero is
        // `0.000000`, which is a number, and it comes back as one -- the empty
        // field is a different thing and not another spelling of it.
        if (i == 1 and text.len == 0) {
            v.* = null;
            continue;
        }
        v.* = std.fmt.parseFloat(f64, text) catch return error.CurveShape;
    }
    // A fourth field means the header and the rows disagree about the shape,
    // which is the one case where reading the first four and moving on would
    // compare a train loss against whatever happened to be next.
    if (fields.next() != null) return error.CurveShape;
    return c;
}

/// Writes a header line and one line per row, truncating `path` first.
///
/// Losses carry six decimals and the rate eight, fixed, so the file is a
/// readable curve and two runs of one seed produce the same bytes. Nothing
/// here allocates: each line is formatted into a stack buffer and streamed.
///
/// ponytail: the signature takes a path and no `Io`, so it reaches for the
/// standard library's single-threaded instance, which supports no concurrency
/// and no cancelation. A caller that wants its own `Io` writes the same rows
/// with `std.Io.Dir.createFile`; the upgrade is a fifth parameter, not a
/// second writer.
pub fn writeCsv(path: []const u8, rows: []const Row) !void {
    const threaded = std.Io.Threaded.global_single_threaded;
    const io = threaded.io();
    // `truncate` defaults to true, so a shorter run over a longer file leaves
    // no tail of the previous curve behind.
    const file = try std.Io.Dir.cwd().createFile(io, path, .{});
    defer file.close(io);
    var buffer: [128]u8 = undefined;
    var out: std.Io.File.Writer = .init(file, io, &buffer);
    const w = &out.interface;
    try w.print(csv_header ++ "\n", .{});
    for (rows) |row| {
        // The empty field, and the reason `writeCsv` builds the column rather
        // than printing the optional: a row logged before the first epoch end
        // has no validation measurement, and `0.000000` would be a claim that
        // it measured exactly zero. A measured zero still prints `0.000000`,
        // so the two are different bytes. One format string for both, so the
        // column count cannot depend on which of the two a row carries.
        var val: [64]u8 = undefined;
        const val_text = if (row.val_loss) |v|
            try std.fmt.bufPrint(&val, "{d:.6}", .{v})
        else
            "";
        try w.print("{d},{d:.6},{s},{d:.8}\n", .{
            row.step,
            row.train_loss,
            val_text,
            row.lr,
        });
    }
    try w.flush();
}

/// The parameters and their gradients in one fixed order: tok_embed, then each
/// layer's nine in field order, then final_norm.
///
/// The two arrays have to line up index for index, because the optimizer pairs
/// them, and the order has to be the same on every run, because the clipper's
/// single f64 norm sums them in index order. Copies of the `Tensor` headers,
/// which share the buffers they name, so writing through one writes the
/// parameter itself.
fn flatten(p: *model.Params, g: *autograd.Grads, ps: []Tensor, gs: []Tensor) void {
    std.debug.assert(ps.len == gs.len);
    std.debug.assert(ps.len == 2 + 9 * p.layers.len);

    ps[0] = p.tok_embed;
    gs[0] = g.tok_embed;
    for (p.layers, g.layers, 0..) |*l, *gl, i| {
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

/// Mean loss over the first `val_batches` validation batches. No backward pass
/// and no optimizer call: this measures, it does not train.
fn evalLoss(
    allocator: std.mem.Allocator,
    p: model.Params,
    cfg: model.Config,
    batcher: *data.Batcher,
) !f64 {
    var total: f64 = 0;
    var n: usize = 0;
    while (n < val_batches) {
        var batch = (try batcher.next()) orelse break;
        defer batch.deinit();

        var logits = try model.forward(allocator, p, cfg, batch.inputs);
        defer logits.deinit();
        const batch_loss = try loss.forward(logits, batch.targets);
        // Same guard as the training step. A validation loss is written into
        // the row, so a NaN here is a NaN in the file, and the next epoch's
        // training step is going to be non-finite too.
        if (!std.math.isFinite(batch_loss)) return error.NonFiniteLoss;
        total += batch_loss;
        n += 1;
    }
    // A zero count would divide to a NaN and hand back a loss that reads like a
    // measurement, so the empty stream is refused here and at the batcher.
    if (n == 0) return error.EmptyValidation;
    return total / @as(f64, @floatFromInt(n));
}

// THE ONE GATE ON THE CUDA ATTENTION SEAM, and it is in this file because it
// needs BOTH halves at once: `autograd.Cache` for the tensors a real training
// step's forward produced, and `attention.forward` for the CPU twin that grades
// them. `src/cuda/run-attn.sh` can only have the second -- it grades uniform
// +/-0.5 inputs and knows nothing about a step.
//
// THE TWO SIDES, and the earlier version of this test had both of them on the
// GPU: `want` was read out of the same cache the CUDA arm had just written, and
// `got` was a second launch of the same kernel, so `attention.forward` -- the
// implementation this is a claim about -- was never called and the test could
// only fail if the kernel were non-deterministic. It printed "ok" and meant
// nothing. So the CPU side is called here, directly, on the three tensors the
// CUDA arm was handed, and the CUDA side is what the MODEL got: `b.ctx` is
// `model.forwardWith`'s own `.attn_ctx`, which under `cuda_attn` is
// `cudaForward`'s return value. Reading the answer out of the model rather than
// launching the kernel again from here is what makes it a gate on the SEAM --
// `Attn.init`, the three uploads, the launch config, the download and the shape
// checks are all inside `cudaForward` and all of them are graded by this
// comparison, none of which a hand-rolled second launch in the test would have
// covered.
//
// WHY THE `comptime`, which is the part that is load-bearing. This test is
// compiled into `tests.o` in EVERY configuration, because `src/train_test.zig`
// imports this file and the CPU suite is rooted at `src/tests.zig`. An `extern fn`
// becomes a link-time requirement as soon as the function calling it is
// ANALYSED, so a body that merely NAMES a kernel entry point behind a runtime
// `if` -- or that takes its address, the shape `src/cuda/device.zig` carries a
// paragraph about having shipped once and reverted -- puts `zt_attn_forward` and
// `cudaFree` into that object and takes `zig build test` down on every host with
// no CUDA toolchain, which is every GitHub runner. Zig does not analyse the
// untaken arm of a comptime-known `if`, so the arm above is the whole of this
// file's contribution to a CPU link. Checked rather than assumed: a
// `@compileError` put in it does not fire.
//
// WHERE IT RUNS, and the CPU run is not asked to report a pass it did not earn.
// `zig build cuda-attn-check` and nothing else: it is the only build in the
// graph that links `src/cuda/attn_kernels.cu`, and on a tree where
// `src/model.zig:cuda_attn` is false it EXITS 1 naming the line to edit rather
// than skipping quietly. Everywhere else the arm below returns
// `error.SkipZigTest`, which the test runner reports as `SKIP` and does NOT count
// as passed -- Zig prints `... SKIP` on its own line and a summary that reads
// `passed; 1 skipped; 0 failed`. A CPU-only `zig build test` therefore cannot be
// read as having compared anything.
test "the CUDA attention forward agrees with the CPU one on a real training step" {
    if (comptime model.cuda_attn) {
        const cfg = model.defaultConfig();
        const t_count = cfg.n_ctx;
        const gpa = std.testing.allocator;

        var params = try model.initParams(gpa, cfg, 7);
        defer params.deinit();

        // Real ids in range, and that is the whole of what a batch has to be: the
        // embedding is indexed by them and every check on them is a bound. Nothing
        // here needs a corpus, and reading one would tie the gate to a file the
        // numerics do not care about.
        const tokens = try gpa.alloc(u32, t_count);
        defer gpa.free(tokens);
        for (tokens, 0..) |*tok, i| tok.* = @intCast(i % cfg.vocab_size);

        // One real step's forward pass, through `model.forwardWith`, which is what
        // decides which arm of the seam ran. The cache copies the intermediates
        // out because the pass frees each layer's buffers before it returns.
        var cache = try autograd.Cache.init(gpa, params, cfg, tokens);
        defer cache.deinit();
        var logits = try model.forwardWith(gpa, params, cfg, tokens, &cache.sink);
        defer logits.deinit();
        const b = cache.blocks[0];

        // THE CUDA SIDE: what the model got. `b.ctx` is the cache's copy of the
        // pass's `.attn_ctx`, which `cudaForward` produced because `cuda_attn` is
        // true -- the same constant the arm this test is in reads.
        const got = b.ctx;

        // THE CPU SIDE: `src/attention.zig`'s own forward, called here on `q_pos`,
        // `k_pos` and `v` exactly as the CUDA arm received them, with the same
        // `attention.Config` `model.forwardWith` builds for itself. Nothing in this
        // test reaches the device to produce this number.
        var want = try attention.forward(gpa, b.q_pos, b.k_pos, b.v, .{
            .n_heads = cfg.n_heads,
            .n_kv_heads = cfg.n_kv_heads,
            .head_dim = cfg.head_dim,
        });
        defer want.deinit();
        try std.testing.expectEqual(want.data.len, got.data.len);

        // RELATIVE, and the reason is this repository's own: `run-attn.sh`'s
        // `ATTN_TOL` is an ABSOLUTE bound calibrated on that script's own uniform
        // +/-0.5 inputs, so it says nothing about a kernel handed the magnitudes a
        // step actually produces, and an absolute gate here would be either
        // vacuous or unreachable depending on the answer. `max|a-b| <= tol *
        // max|b|` means the same thing at any scale, which is the only property a
        // forward comparison needs.
        var worst: f32 = 0;
        var scale: f32 = 0;
        for (want.data, got.data) |w, h| {
            worst = @max(worst, @abs(w - h));
            scale = @max(scale, @abs(w));
        }
        // The reference has to carry a magnitude at all. `scale == 0` would make
        // the gate below an equality test against a pair of zero tensors, which
        // passes, and a comparison that can pass on nothing is the defect this
        // test was written to stop repeating.
        if (!(scale > 0)) return error.EmptyReference;
        if (!(worst <= attn_rel_tol * scale)) {
            std.debug.print(
                "\nCUDA attention forward differs from the CPU one by {e} against a" ++
                    " reference of at most {e},\nwhich is {e} times the relative gate" ++
                    " of {e}.\n",
                .{ worst, scale, worst / scale, attn_rel_tol },
            );
            return error.AttentionMismatch;
        }
    } else {
        return error.SkipZigTest;
    }
}

// The relative gate above, as a named constant because the failure message quotes
// it and a literal in two places is one of them to forget.
//
// MEASURED, not assumed. On the 32-core NVIDIA host at seed 7 and the shipped
// shape -- 256 tokens, 4 heads over 2 kv heads, head_dim 32, `group_q` 1,
// `max_tile` 64 -- the worst absolute difference between the CUDA answer and
// `attention.forward` is 2.9802322e-08 against a largest reference value of
// 7.8568566e-01: a ratio of 3.793161e-08, so this gate sits about 2600x above the
// noise, and what is left of that gap is the CPU twin's f64 accumulation.
//
// 1e-4 rather than something tighter, and the reason is what the gate has to
// catch. The defects this seam can carry -- a dropped causal mask, a collapsed GQA
// group, a missing or doubled `1/sqrt(head_dim)` -- each move the answer by a
// fraction of itself rather than by a part in a million of it, so 1e-4 clears all
// of them by more than an order of magnitude and leaves three orders of magnitude
// for a different card, a different libm and a different summation order inside
// the kernel. It is also the constant `sh src/cuda/run-attn.sh` gates that same
// kernel at, so a reader comparing the two gates is comparing like with like.
const attn_rel_tol: f32 = 1e-4;
