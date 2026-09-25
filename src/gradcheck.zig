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
/// which dividing by 2h turns into the floor below; the truncation error is
/// h^2 times the third derivative. Measured on the tiny config, the two cross
/// between 1e-3 and 3e-4 and the total error is flat there, so one step is used
/// rather than one per tensor, and the floor is computed rather than guessed.
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

/// What one sweep found: the per tensor worst discrepancy, the floor it is
/// read against, and the first element that did not fit. Nothing here has been
/// printed. `report` prints the table and the caller judges `mismatch`.
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
    tol: f32,
    g: *const autograd.Grads,
) !Report {
    var floor: f64 = 0;
    {
        var logits = try model.forward(allocator, p, cfg, tokens);
        defer logits.deinit();

        var logit_scale: f64 = 0;
        for (logits.data) |z| logit_scale = @max(logit_scale, @abs(@as(f64, @floatCast(z))));
        floor = @as(f64, @floatFromInt(logits.cols)) * f32_epsilon * logit_scale /
            (2.0 * @as(f64, @floatCast(step)));
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
    for (groups[0..n]) |*group| {
        const scale = @max(maxAbs(group.grad), floor);
        for (group.param.data, group.grad.data, 0..) |*element, analytic, i| {
            const original = element.*;
            element.* = original + step;
            const up = try lossAt(allocator, cfg, p, tokens, targets);
            element.* = original - step;
            const down = try lossAt(allocator, cfg, p, tokens, targets);
            element.* = original;

            const numeric = (up - down) / (2.0 * @as(f64, @floatCast(step)));
            const a: f64 = @floatCast(analytic);
            const diff = @abs(a - numeric);
            const budget = @as(f64, @floatCast(tol)) * @max(@abs(a), @abs(numeric)) + floor;

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

/// The per tensor worst discrepancies, on demand. A passing run prints nothing
/// at all, so this is how a developer who wants the margins asks for them:
/// call it on the `Report` a `compare` returned.
pub fn report(r: Report) void {
    var buf: [48]u8 = undefined;
    for (r.groups) |group| {
        std.debug.print("gradcheck {s:<24} worst {e:>10.3} of max grad  at {d:>5}  h {e:.0}  floor {e:.3}\n", .{
            label(&buf, group.layer, group.field),
            group.worst,
            group.worst_at,
            step,
            r.floor,
        });
    }
}

/// Runs one backward pass and grades the gradient with it, printing nothing
/// unless an element is outside its budget, in which case the table goes to
/// `report` and the offending element names itself.
///
/// An element passes when
///     |analytic - numeric| <= tol * max(|analytic|, |numeric|) + floor
/// where `floor` is the most a central difference of this loss can resolve: the
/// softmax sums `vocab` f32 logits, each carrying a representation error of
/// `f32_epsilon` times its own magnitude, and the difference divides that by
/// `2 * step`. An element whose true gradient is far below the floor cannot be
/// checked elementwise in f32 at all, and pretending otherwise would mean a
/// tolerance that is tight for the large elements and impossible for the small
/// ones.
///
/// `p` is restored element by element as it goes, so the caller's parameters
/// come back unchanged. Every element of every parameter is visited either way;
/// only who hears the verdict changes.
pub fn checkAll(
    allocator: std.mem.Allocator,
    cfg: model.Config,
    p: model.Params,
    tokens: []const u32,
    targets: []const u32,
    tol: f32,
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

    const r = try compare(allocator, cfg, p, tokens, targets, tol, &g);
    defer r.deinit();
    if (r.mismatch) |m| {
        report(r);
        const text = try line(allocator, m);
        defer allocator.free(text);
        std.debug.print("{s}\n", .{text});
        return error.GradientMismatch;
    }
}

fn lossAt(
    allocator: std.mem.Allocator,
    cfg: model.Config,
    p: model.Params,
    tokens: []const u32,
    targets: []const u32,
) !f64 {
    var logits = try model.forward(allocator, p, cfg, tokens);
    defer logits.deinit();
    return loss.forward(logits, targets);
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
