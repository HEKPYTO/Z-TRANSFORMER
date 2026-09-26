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
const loss = @import("loss.zig");
const optim = @import("optim.zig");
const tensor = @import("tensor.zig");
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
    /// epoch ends read 0 because nothing has been measured yet.
    val_loss: f64,
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
    // per epoch like the training one, from a stream the training batcher's does
    // not touch, so the two orders stay independent.
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
    var val_loss: f64 = 0;
    var in_epoch: usize = 0;

    for (0..cfg.epochs) |_| {
        while (true) {
            var batch = (try batcher.next()) orelse break;
            defer batch.deinit();

            var logits = try model.forward(allocator, params, cfg.model, batch.inputs);
            defer logits.deinit();
            const batch_loss = try loss.forward(logits, batch.targets);
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
            try autograd.backward(allocator, params, &grads, cfg.model, batch.inputs, dlogits);

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
            if (!std.math.isFinite(norm)) return error.NonFiniteGradient;
            const at = step;
            const lr = optim.cosineLR(@intCast(at), total, warmup, cfg.lr);
            for (flat_params, flat_grads, states) |*p, g, *s| try s.step(p, g, lr, cfg.weight_decay);
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
        rows.items[rows.items.len - 1].val_loss = val_loss;
        val_batcher.reset();
        batcher.reset();
        in_epoch = 0;
        epoch_loss_sum = 0;
    }

    return .{
        .train_loss = run_loss_sum / @as(f64, @floatFromInt(step)),
        .val_loss = val_loss,
        .steps = step,
        .params = params,
        .rows = try rows.toOwnedSlice(allocator),
        .allocator = allocator,
    };
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
    try w.print("step,train_loss,val_loss,lr\n", .{});
    for (rows) |row| {
        try w.print("{d},{d:.6},{d:.6},{d:.8}\n", .{
            row.step,
            row.train_loss,
            row.val_loss,
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
