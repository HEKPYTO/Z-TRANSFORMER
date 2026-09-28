//! Finite-difference gradient check against the real forward pass.
//!
//! The oracle is `model.forward` re-run with one parameter element moved, which
//! is a different method from the algebra in `autograd.zig` and not a second
//! copy of it. Nothing here reuses a backward function to decide whether a
//! gradient is right.
const std = @import("std");
const model = @import("model.zig");
const autograd = @import("autograd.zig");
const loss = @import("loss.zig");
const tensor = @import("tensor.zig");
const Tensor = tensor.Tensor;

/// Central-difference step for every parameter.
///
/// f32 tensors and f32 transcendental calls bound this from both sides. The
/// difference of two losses carries the f32 representation error of the logits,
/// which dividing by 2h turns into the floor in `floorOf`; the truncation error
/// is h^2 times the third derivative. One step is used for every element rather
/// than one per tensor, and the floor it is read against is computed rather than
/// guessed: at this step the floor is three orders of magnitude above the
/// truncation term h^2/6 unless a third derivative exceeds a thousand, so the
/// total is flat here and the cheaper of the two errors is not worth trading.
const step: f32 = 1e-3;

/// f32 has a 24 bit significand, so a value carries a representation error of
/// at most 2^-24 of itself. This is that number, named so the floor below reads
/// as what it is.
const f32_epsilon: f64 = 1.0 / 16777216.0;

const Group = struct {
    param: *const Tensor,
    grad: *const Tensor,
    layer: ?usize,
    field: []const u8,
    worst: f64 = 0,
    worst_at: usize = 0,
};

/// The first parameter element that did not fit its budget, or null when every
/// element agreed. This is everything the failure line names, as data.
pub const Mismatch = struct {
    layer: ?usize,
    field: []const u8,
    index: usize,
    analytic: f64,
    numeric: f64,
    diff: f64,
    budget: f64,
};

/// What one sweep found: the per tensor worst discrepancy, the largest budget
/// any element was read against, and the first element that did not fit.
/// Nothing here has been written. `report` writes the table to the writer it
/// is given and the caller judges `mismatch`.
pub const Report = struct {
    allocator: std.mem.Allocator,
    groups: []Group,
    floor: f64,
    mismatch: ?Mismatch,

    pub fn deinit(self: Report) void {
        self.allocator.free(self.groups);
    }
};

const layer_fields = [_][]const u8{
    "attn_norm", "wq", "wk", "wv", "wo", "mlp_norm", "w_gate", "w_up", "w_down",
};

/// Differences every parameter element of `p` against the analytic gradient in
/// `g` and returns what it found. Silent, and never an error: a disagreement is
/// `Report.mismatch` rather than a return, so a caller can read the margins of
/// a passing sweep as well as a failing one.
pub fn compare(
    allocator: std.mem.Allocator,
    cfg: model.Config,
    p: model.Params,
    tokens: []const u32,
    targets: []const u32,
    g: *const autograd.Grads,
) !Report {
    var base_scale: f64 = 0;
    {
        // The logit scale at the base point, which is the scale the sweep's
        // budget is anchored on. The two losses an element is differenced over
        // are read at points a step away from here and can carry a different
        // scale, which is what `Reading` below exists to notice.
        var logits = try model.forward(allocator, p, cfg, tokens);
        defer logits.deinit();
        for (logits.data) |z| base_scale = @max(base_scale, @abs(@as(f64, @floatCast(z))));
    }

    var groups = try allocator.alloc(Group, 2 + 9 * p.layers.len);
    errdefer allocator.free(groups);
    var n: usize = 0;
    groups[n] = .{ .param = &p.tok_embed, .grad = &g.tok_embed, .layer = null, .field = "tok_embed" };
    n += 1;
    for (p.layers, 0..) |*l, li| {
        const params = tensorsOf(l);
        const grads = gradsOf(&g.layers[li]);
        for (params, grads, layer_fields) |param, grad, field| {
            groups[n] = .{ .param = param, .grad = grad, .layer = li, .field = field };
            n += 1;
        }
    }
    groups[n] = .{ .param = &p.final_norm, .grad = &g.final_norm, .layer = null, .field = "final_norm" };
    n += 1;

    var mismatch: ?Mismatch = null;
    var floor: f64 = 0;
    for (groups[0..n]) |*group| {
        for (group.param.data, group.grad.data, 0..) |*element, analytic, i| {
            const original = element.*;
            // Registered before the first evaluation, because a loss that fails
            // returns before the put-back below and would otherwise leave the
            // caller holding a parameter a step off in a direction it never
            // asked for.
            errdefer element.* = original;
            element.* = original + step;
            const up = try lossAt(allocator, cfg, p, tokens, targets);
            element.* = original - step;
            const down = try lossAt(allocator, cfg, p, tokens, targets);
            element.* = original;

            const numeric = (up.loss - down.loss) / (2.0 * @as(f64, @floatCast(step)));
            const a: f64 = @floatCast(analytic);
            const diff = @abs(a - numeric);
            // The floor is the whole budget. A relative term beside it would be a
            // tolerance someone chose, and it would buy nothing: the analytic
            // gradient is itself an f32 sum, so its own representation error is
            // f32_epsilon times its own magnitude, which is orders of magnitude
            // below the floor at every magnitude this check sees. What the floor
            // cannot resolve, no chosen fraction of the gradient rescues.
            //
            // Per element, because the three evaluations it is read off are this
            // element's own. A base point whose logits vanish still has a
            // gradient, and reading the scale off the base point alone there
            // gives a floor of zero, which is not a resolution, it is a claim
            // that f32 resolved something it did not.
            const budget = floorOf(
                @max(base_scale, @max(up.logit_scale, down.logit_scale)),
                tokens.len,
                cfg.vocab_size,
            );
            floor = @max(floor, budget);
            const scale = @max(maxAbs(group.grad), budget);

            // The first one, not the only one: the sweep runs to the end either
            // way, so a failure reports a whole table and the failing row is the
            // one that stands out.
            if (diff > budget and mismatch == null) {
                mismatch = .{
                    .layer = group.layer,
                    .field = group.field,
                    .index = i,
                    .analytic = a,
                    .numeric = numeric,
                    .diff = diff,
                    .budget = budget,
                };
            }
            // A tensor whose analytic gradient is identically zero has no scale
            // of its own, and the floor is the only thing left to measure
            // against, which is the honest reading of that case.
            const relative = diff / scale;
            if (relative > group.worst) {
                group.worst = relative;
                group.worst_at = i;
            }
        }
    }

    return .{ .allocator = allocator, .groups = groups[0..n], .floor = floor, .mismatch = mismatch };
}

/// The failure line: which parameter, which element, the two values, the gap
/// and the budget, so a reader never re-derives any of it. A function rather
/// than a format at the print, so the text is assertable without capturing
/// stderr.
pub fn line(allocator: std.mem.Allocator, m: Mismatch) ![]u8 {
    var name: [48]u8 = undefined;
    return std.fmt.allocPrint(
        allocator,
        "gradient mismatch at {s}[{d}]: analytic {d:.9}  numeric {d:.9}  diff {d:.9}  budget {d:.9}",
        .{ label(&name, m.layer, m.field), m.index, m.analytic, m.numeric, m.diff, m.budget },
    );
}

/// The per tensor worst discrepancies, on demand, into `w`. A passing run writes
/// nothing at all, so this is how a developer who wants the margins asks for
/// them: call it on the `Report` a `compare` returned.
///
/// The destination is a parameter because a printer hardcoded to the process's
/// stderr cannot be tested. The only way to cover such a printer is to let its
/// output escape, and escaping inside a test puts bytes on the channel the
/// build runner reads, which is what breaks the build and fails CI on a
/// passing suite. Naming the writer makes the table assertable on captured
/// bytes and keeps the escape to `checkAll`, the one caller that wants it.
pub fn report(w: *std.Io.Writer, r: Report) std.Io.Writer.Error!void {
    var buf: [48]u8 = undefined;
    for (r.groups) |group| {
        try w.print("gradcheck {s:<24} worst {e:>10.3} of max grad  at {d:>5}  h {e:.0}  floor {e:.3}\n", .{
            label(&buf, group.layer, group.field),
            group.worst,
            group.worst_at,
            step,
            r.floor,
        });
    }
}

/// Runs one backward pass and grades the gradient with it, writing nothing
/// unless an element is outside its budget, in which case the table goes to
/// `report` and the offending element names itself, both on stderr.
///
/// An element passes when
///     |analytic - numeric| <= floor
/// where `floor` is the most an f32 central difference of this loss can resolve
/// at that element, derived in `floorOf` from the logit scale and the step. It is
/// the whole budget, with no chosen relative term beside it. It is also the limit
/// of the check: an element whose true gradient is far below the floor cannot be
/// checked elementwise in f32 at all, and a budget that claimed otherwise would
/// be a number somebody picked.
///
/// `p` is restored element by element as it goes, error path included, so the
/// caller's parameters come back unchanged. Every element of every parameter is
/// visited either way; only who hears the verdict changes.
pub fn checkAll(
    allocator: std.mem.Allocator,
    cfg: model.Config,
    p: model.Params,
    tokens: []const u32,
    targets: []const u32,
) !void {
    var g = try autograd.zeroGrads(allocator, p);
    defer g.deinit();

    {
        var logits = try model.forward(allocator, p, cfg, tokens);
        defer logits.deinit();

        var dl = try autograd.dLossDLogits(allocator, logits, targets);
        defer dl.deinit();
        try autograd.backward(allocator, p, &g, cfg, tokens, dl);
    }

    const r = try compare(allocator, cfg, p, tokens, targets, &g);
    defer r.deinit();
    if (r.mismatch) |m| {
        // The only place the table and the line are meant to be read, and so
        // the only place that names the process's stderr. Silence is the
        // contract, so the destination is claimed here rather than in the
        // printer, which is what lets the printer be tested at all.
        var buf: [512]u8 = undefined;
        const locked = std.debug.lockStderr(&buf);
        defer std.debug.unlockStderr();
        const w = &locked.file_writer.interface;

        try report(w, r);
        const text = try line(allocator, m);
        defer allocator.free(text);
        try w.print("{s}\n", .{text});
        return error.GradientMismatch;
    }
}

/// One loss evaluation together with the size of the logits it was read from.
/// The scale travels with the loss because the budget is read off it, and the
/// two losses a difference is made of are read at points a step either side of
/// the base point, where the logits need not be the same size.
const Reading = struct {
    loss: f64,
    logit_scale: f64,
};

/// The most an f32 central difference of this loss can resolve: an absolute
/// bound on the disagreement between the analytic gradient and the numeric one,
/// per element, from the numbers the numeric one is made of.
///
/// The loss is `(1/T) * sum_t (logsumexp z[t] - z[t][target])`, so:
///
///   - every f32 logit carries a representation error of at most `f32_epsilon`
///     times its own magnitude, and the largest logit of the three evaluations
///     bounds all of them;
///   - the `T * vocab` of those roundings are independent, so they compose in
///     quadrature as `sqrt(T * vocab)` rather than as `T * vocab`. Summing is the
///     worst case of an ensemble of independent errors, every one of them
///     maximal and in the same direction, and it overstates the resolvable
///     precision by `sqrt(vocab)`;
///   - the mean divides the sum by `T`;
///   - the difference of two losses carries that error, and the difference
///     divides it by `2 * step`.
///
/// Nothing here is chosen. It is the widest disagreement a correct gradient can
/// have and still be a correct gradient, and it is what the truncation error
/// would have to be measured against.
fn floorOf(logit_scale: f64, tokens: usize, vocab: usize) f64 {
    const t: f64 = @floatFromInt(tokens);
    const v: f64 = @floatFromInt(vocab);
    return @sqrt(v / t) * f32_epsilon * logit_scale / (2.0 * @as(f64, @floatCast(step)));
}

fn lossAt(
    allocator: std.mem.Allocator,
    cfg: model.Config,
    p: model.Params,
    tokens: []const u32,
    targets: []const u32,
) !Reading {
    var logits = try model.forward(allocator, p, cfg, tokens);
    defer logits.deinit();
    var scale: f64 = 0;
    for (logits.data) |z| scale = @max(scale, @abs(@as(f64, @floatCast(z))));
    return .{ .loss = try loss.forward(logits, targets), .logit_scale = scale };
}

fn maxAbs(t: *const Tensor) f64 {
    var m: f64 = 0;
    for (t.data) |v| m = @max(m, @abs(@as(f64, @floatCast(v))));
    return m;
}

fn tensorsOf(l: *model.Layer) [9]*const Tensor {
    return .{
        &l.attn_norm, &l.wq,     &l.wk,   &l.wv,     &l.wo,
        &l.mlp_norm,  &l.w_gate, &l.w_up, &l.w_down,
    };
}

fn gradsOf(l: *const autograd.LayerGrads) [9]*const Tensor {
    return .{
        &l.attn_norm, &l.wq,     &l.wk,   &l.wv,     &l.wo,
        &l.mlp_norm,  &l.w_gate, &l.w_up, &l.w_down,
    };
}

/// A group name, in a caller supplied buffer so nothing here allocates. The
/// field name is a literal that always fits; a formatted name cannot overrun
/// 48 bytes, and the fallback is the bare field name rather than a panic.
fn label(buf: []u8, layer: ?usize, field: []const u8) []const u8 {
    const l = layer orelse return std.fmt.bufPrint(buf, "{s}", .{field}) catch field;
    return std.fmt.bufPrint(buf, "layers[{d}].{s}", .{ l, field }) catch field;
}
