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
/// which dividing by 2h turns into the budget in `floorOf`; the truncation error
/// is h^2 times the third derivative. One step is used for every element rather
/// than one per tensor, and the budget is computed rather than guessed: measured
/// by Richardson on the real f32 forward, truncation sits near 1e-6 at this step.
/// The gap to the roundoff term is shape-dependent rather than one number, since
/// `floorOf` divides by sqrt(T): about 1.5 orders of magnitude at the shipped
/// context (T=256) and about 2.5 at the fixtures' T=4. "Three orders" held at no
/// shape that can be measured here, so the conclusion this comment draws is
/// correspondingly weaker than the one it used to -- the truncation term is
/// OMITTED from the budget rather than bounded by it, and it stays negligible
/// only while a third derivative is a few tens of what the shipped context
/// produces and a few hundred of what T=4 produces. The cheaper error is still
/// not worth trading, but it is not three orders of slack.
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
///
/// `headroom` is `1 / max(diff / budget)` over every element read: the factor
/// the whole budget could be multiplied by before the check would begin to
/// fail, and 1.0 the largest value a correct gradient may report. It is here
/// because a budget is only as good as its tightness. A budget that had grown
/// ten times looser would still pass every element in the suite while no longer
/// being able to see a one-percent gradient error, and a test asserting only
/// `mismatch == null` cannot tell those two apart.
pub const Report = struct {
    allocator: std.mem.Allocator,
    groups: []Group,
    floor: f64,
    headroom: f64,
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
    // The tightest element in the sweep, as a fraction of its own budget. The
    // whole report is judged off this as well as off `mismatch`: an element at
    // 0.65 and one at 0.01 both pass, and only one of them is a budget with any
    // margin left in it.
    var tightest: f64 = 0;
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
            // The budget is the whole floor. A relative term beside it would be a
            // tolerance someone chose, and it would buy nothing: the analytic
            // gradient is itself an f32 sum, so its own representation error is
            // f32_epsilon times its own magnitude, which is orders of magnitude
            // below the budget at every magnitude this check sees. What the
            // budget cannot resolve, no chosen fraction of the gradient rescues.
            //
            // Per element, because the three evaluations it is read off are this
            // element's own. A base point whose logits vanish still has a
            // gradient, and reading the scale off the base point alone there
            // gives a budget of zero, which is not a resolution, it is a claim
            // that f32 resolved something it did not.
            const budget = floorOf(
                @max(base_scale, @max(up.logit_scale, down.logit_scale)),
                tokens.len,
            );
            floor = @max(floor, budget);
            tightest = @max(tightest, diff / budget);
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

    return .{
        .allocator = allocator,
        .groups = groups[0..n],
        .floor = floor,
        .headroom = 1.0 / tightest,
        .mismatch = mismatch,
    };
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
        try w.print("gradcheck {s:<24} worst {e:>10.3} of max grad  at {d:>5}  h {e:.0}  floor {e:.3}  headroom {e:.2}x\n", .{
            label(&buf, group.layer, group.field),
            group.worst,
            group.worst_at,
            step,
            r.floor,
            r.headroom,
        });
    }
}

/// Runs one backward pass and grades the gradient with it, writing nothing
/// unless an element is outside its budget, in which case the table goes to
/// `report` and the offending element names itself, both on stderr.
///
/// An element passes when
///     |analytic - numeric| <= budget
/// where `budget` is the most an f32 central difference of this loss can resolve
/// at that element, derived in `floorOf` from the logit scale, the token count and
/// the step. It is the whole budget, with no chosen relative term beside it. It is
/// also the limit of the check: an element whose true gradient is far below the
/// budget cannot be checked elementwise in f32 at all, and a budget that claimed
/// otherwise would be a number somebody picked.
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

/// How many sigma of per-element spread the derivation is short by. Measured,
/// not chosen; see `floorOf`.
const k: f64 = 12.0;

/// The most an f32 central difference of this loss can resolve: an absolute
/// bound on the disagreement between the analytic gradient and the numeric one,
/// per element, from the numbers the numeric one is made of.
///
///     budget = sqrt(2) * k * f32_epsilon * logit_scale / (2 * step * sqrt(T))
///
/// The loss is `(1/T) * sum_t (logsumexp z[t] - z[t][target])`, and every f32
/// logit carries a representation error of at most `f32_epsilon` times its own
/// magnitude, which the largest logit of the three evaluations bounds. The
/// absence of a `vocab` is the correction, and it is not a loosening: the
/// derivative `dL/dz_v = (p_v - 1[v=tgt]) / T` already carries the
/// `1/sqrt(vocab)` a near-uniform softmax imposes, so a `sqrt(vocab)` out here
/// counts it twice. That one term is why the old budget was roughly a thousand
/// times loose at vocab 16 and 5.6% over at vocab 256.
///
/// Cauchy-Schwarz over the independent per-logit errors `eta[t][v]` bounds the
/// loss error at `eps * scale * sqrt( sum ((p_v - 1[v=tgt]) / T)^2 ) / T`, and
/// the sum under that root is `T * vocab * E_v[(p - onehot)^2]`. For a
/// near-uniform softmax `p_v ~ 1/vocab`, so the per-token sum is
///
///     1/vocab + (1 - 1/vocab)^2 * (vocab - 1)  ~=  1
///
/// independent of `vocab`, and the whole thing collapses to `eps * scale /
/// sqrt(T)`. The two losses a difference is made of are separate evaluations, so
/// their roundoffs add in quadrature: the `sqrt(2)`.
///
/// `k` is where the derivation ends and measurement begins, and it is the one
/// constant here that is not a function of the config. Over 550 sweeps
/// (110 configs x 5 seeds, 2.2M elements) the largest per-element discrepancy
/// ran 3.8 to 8.4 sigma, while at one fixed config sigma itself swung 500 to
/// 1000 across seeds. No config-only formula can be tight against a spread that
/// depends on the weights, so `k` absorbs all of it. At `k = 12` the worst of
/// those 550 sweeps sat at 0.651 of budget, which bounds it with 35% to spare.
///
/// The width is real and is not removed by pinning a test fixture: `checkAll`
/// runs on trained weights, which are arbitrary draws, and a budget sized for
/// the median draw would report correct gradients as mismatches on the tail. A
/// 4000-draw study of the headroom under redrawn seeds runs 0.46 to 10.2, and
/// the 0.46 end is a correct gradient reported as a mismatch, which is why `k`
/// is set from the worst case and not the typical one. That study bounds `k`;
/// it does not license a loose assertion about it, which is why the assertion
/// in `gradcheck_test.zig` brackets one pinned draw instead.
///
/// Two things the derivation omits. Truncation, `h^2/6 * f'''`, is 1.5 orders of
/// magnitude below the roundoff term at the shipped context and 2.5 at the test
/// fixtures' T=4, so a shape that pushes it up would bring the two together, and
/// it is not in the total. If a future shape ever makes them comparable, add a
/// term for it rather than widening `k`, which measures weight spread and would
/// then hide both. Gradient path accumulation depth needs no term: at d_model 32,
/// sweeping kv-group 1 to 4 over five draws, per-element sigma was flat with no
/// trend.
fn floorOf(logit_scale: f64, tokens: usize) f64 {
    const t: f64 = @floatFromInt(tokens);
    return @sqrt(2.0) * k * f32_epsilon * logit_scale / (2.0 * @as(f64, @floatCast(step)) * @sqrt(t));
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
