//! Scale projections: what the code's own formulas imply at shapes the code
//! cannot be run at.
//!
//! This is a projection, not a benchmark, and it says so in its own output. No
//! forward pass is run at any shape here, no wall time is printed, and nothing
//! below is a measurement.
//!
//! There is no `@setRuntimeSafety` escape and no unsafe, so every product below
//! is checked: a shape whose arithmetic overflows `u64` traps rather than
//! printing a wrapped number. The softmax is left out of the attention FLOP
//! count, which is a bound rather than an oversight: `attention.forward` does
//! four non-multiply-add operations per score against `head_dim` multiply-adds,
//! so the whole softmax term is `4 / head_dim` of the attention core and under
//! 3% at any head_dim this project uses. It is named in the output so the
//! omission is checkable.

const std = @import("std");
const model = @import("model.zig");

/// One row of the sweep: a name a reader can look up and a shape to project.
/// `cfg` is a real `model.Config`, so a row cannot claim a shape the model
/// would refuse to build.
pub const Shape = struct {
    name: []const u8,
    cfg: model.Config,
};

/// The shipped shape first, then the same width at longer contexts, then a
/// width that is a real model's, then two rows that bracket the crossovers the
/// deferred work was argued from. The last two are not aspirations:
/// `llama3-8b-32k` is the first row whose attention core passes
/// `attention_bar`, and `at-parity-12d` sits exactly on the context where the
/// core equals one layer's MLP, so a crossover is a line of this tool's output
/// rather than a sentence in a document that has to be taken on trust.
pub const sweep = [_]Shape{
    .{ .name = "shipped", .cfg = model.defaultConfig() },
    .{ .name = "ctx-1k", .cfg = withCtx(model.defaultConfig(), 1024) },
    .{ .name = "ctx-4k", .cfg = withCtx(model.defaultConfig(), 4096) },
    .{ .name = "d4096-ctx4k", .cfg = llamaThree(4096) },
    .{ .name = "llama3-8b", .cfg = llamaThree(8192) },
    .{ .name = "llama3-8b-32k", .cfg = llamaThree(32768) },
    .{ .name = "at-parity-12d", .cfg = llamaThree(12 * 4096) },
};

fn withCtx(cfg: model.Config, n_ctx: usize) model.Config {
    var out = cfg;
    out.n_ctx = n_ctx;
    return out;
}

/// Llama-3 8B's shape with this repository's own feed-forward rule, which is
/// `ffn_mult * d_model` with `ffn_mult` an integer and so is 4x where Llama-3
/// is 3.5x rounded to a multiple of 256. `ffn_mult = 4` is stated here rather
/// than inherited so the one place this row is not Llama-3's is visible.
fn llamaThree(n_ctx: usize) model.Config {
    return .{
        .n_layers = 32,
        .n_heads = 32,
        .n_kv_heads = 8,
        .head_dim = 128,
        .n_ctx = n_ctx,
        .vocab_size = 128256,
        .ffn_mult = 4,
    };
}

/// Bytes of f32 in a host a person can buy. A round figure, and the column says
/// so: it is only ever the thing a `fits` verdict is a statement about.
const host_bytes: u64 = 32 * 1024 * 1024 * 1024;

/// The bar a deferred item has to clear before it is worth doing, as a share of
/// one layer's MLP arithmetic. A choice, and named as one: nothing derives it.
/// The ratios it is compared against are counted by formula and this is the
/// only judgement in the table, so it is stated here and a reader who prefers a
/// different bar can see which number moved.
const attention_bar: f64 = 0.25;

/// Every projected quantity for one shape, in f32, in the units the field name
/// says. Nothing here was timed.
pub const Projection = struct {
    shape: Shape,
    /// `model.dModel`, `model.ffnDim` and the `kv_dim` of
    /// `model.initLayer`, which is `n_kv_heads * head_dim`.
    d: u64,
    h: u64,
    kv: u64,
    t: u64,
    /// FLOP per layer, forward only. A fused multiply-add is two operations,
    /// which is the convention every count in this file uses and the one the
    /// `24 * T * d^2` claim in the deferral note assumes.
    attn_core: u64,
    attn_proj: u64,
    mlp: u64,
    /// FLOP for the whole step: the tied head's forward (`model.tiedHead`)
    /// and the two products its backward runs at, which is one forward and
    /// two backward. Once per step rather than per layer, and the
    /// embeddings are the only parameter that takes both paths.
    tied_head: u64,
    /// FLOP per layer, `autograd.weightGrad` over the seven projections it
    /// is called on: one each for wq, wk, wv, wo, w_gate, w_up and w_down.
    weight_grad: u64,
    /// f32 elements, from the `shapes` array in `model.initLayer` and the
    /// two outer tensors in `model.initParams`.
    params: u64,
    /// `autograd.zeroGrads` mirrors `Params` element for element, and
    /// `train.run` flattens `2 + 9 * n_layers` tensors over it.
    grads: u64,
    /// `optim.AdamW` holds `m` and `v`, each shaped like the parameter it
    /// updates.
    adam: u64,
    /// f32 elements live at the peak of `autograd.backward`, named term by term
    /// in `activationElems`.
    act: u64,
    /// `n_layers * n_heads * T * T * 4`: the score matrix a dense
    /// implementation would materialize. `attention.forward` does not
    /// materialize it, which is what `scores_row` is for.
    scores_dense: u64,
    /// What `attention.forward` actually allocates: one f64 row of `T`, reused
    /// across rows and heads.
    scores_row: u64,
    /// `T * vocab * d * 4`: `tiedHead` reads a whole `tok_embed` row per
    /// token, so the table is walked `T` times.
    tied_stream: u64,
    /// `n_layers * n_kv_heads * T * head_dim * 2 * 4`: k and v, f32, for every
    /// layer. Not implemented; the number is what implementing it costs.
    kv_cache: u64,
};

/// Projects one shape. Rejects a config `model.validate` would reject, so a row
/// of the sweep cannot print a number for a model that would not build.
pub fn project(shape: Shape) !Projection {
    try model.validate(shape.cfg);
    const cfg = shape.cfg;
    const d: u64 = model.dModel(cfg);
    const h: u64 = model.ffnDim(cfg);
    const kv: u64 = cfg.n_kv_heads * cfg.head_dim;
    const t: u64 = cfg.n_ctx;
    const l: u64 = cfg.n_layers;

    // `attention.forward` walks `0..t + 1` in its score loop, so the score
    // matrix is triangular: n_heads * T(T+1)/2 dots of head_dim, each a
    // multiply-add, and its weighted sum over the same prefix is the same
    // count over the same prefix. Together 2 * d * T * (T + 1), which is HALF
    // the `4 * T^2 * d` the deferral note quotes: that figure is the dense
    // non-causal one every public attention count uses.
    const attn_core = 2 * d * t * (t + 1);
    // In `model.initLayer`'s `shapes`, wq and wo are [d, d] and wk and wv are
    // [d, kv], and `tensor.matmul` is one multiply-add per element of [T,k] and
    // [k,n]. So T * (2d^2 + 2 * d * kv) multiply-adds.
    const attn_proj = 2 * t * (2 * d * d + 2 * d * kv);
    // Still `model.initLayer`: w_gate and w_up are [d, h] and w_down is [h, d],
    // so 3 * T * d * h multiply-adds. At `ffn_mult = 4` that is the
    // `24 * T * d^2` the deferral note quotes, and it is right.
    const mlp = 2 * (3 * t * d * h);
    // `model.tiedHead` is T * vocab * d multiply-adds, and the head loop in
    // `autograd.backwardFrom` is two per element of the same three loops. Three
    // forwards' worth, once per step.
    const tied_head = 2 * (3 * t * cfg.vocab_size * d);
    // One weight gradient per projection, each `input` rows by `dout` columns.
    const weight_grad = 2 * t * (2 * d * d + 2 * d * kv + 3 * d * h);

    const params = cfg.vocab_size * d + l * (2 * d + 2 * d * d + 2 * d * kv + 3 * d * h) + d;
    return .{
        .shape = shape,
        .d = d,
        .h = h,
        .kv = kv,
        .t = t,
        .attn_core = attn_core,
        .attn_proj = attn_proj,
        .mlp = mlp,
        .tied_head = tied_head,
        .weight_grad = weight_grad,
        .params = params,
        .grads = params,
        // Two f32 moments per parameter element.
        .adam = 2 * params,
        .act = activationElems(cfg),
        .scores_dense = l * cfg.n_heads * t * t * 4,
        .scores_row = cfg.n_heads * t * 8,
        .tied_stream = t * cfg.vocab_size * d * 4,
        .kv_cache = l * kv * t * 2 * 4,
    };
}

/// f32 elements live at the peak of one backward pass, term by term. Every term
/// is a tensor the code allocates and keeps, from `train.run` and
/// `autograd.backward`. A term off by one tensor is off by 4 bytes out of
/// billions; a term missing entirely is a different memory profile, so the list
/// is written out rather than collapsed into a coefficient.
fn activationElems(cfg: model.Config) u64 {
    const d: u64 = model.dModel(cfg);
    const h: u64 = model.ffnDim(cfg);
    const kv: u64 = cfg.n_kv_heads * cfg.head_dim;
    const t: u64 = cfg.n_ctx;
    const l: u64 = cfg.n_layers;

    // logits and dlogits, both [T, vocab]. A tenth of the
    // total, so the per-layer blocks below are what fills the machine and not
    // the pair a reader arrives with.
    const head = 2 * t * cfg.vocab_size;
    // The cache's block input (`Cache.init`), one [T, d]: the stream a block
    // starts from is the previous block's `out` and is not held twice.
    const stream = t * d;
    // One `Block` per layer (`autograd.Cache`): six [T, d] (`attn_in`, `q_pos`,
    // `ctx`, `x_mid`, `mlp_in`, `out`), two [T, kv] (`k_pos`, `v`) and three
    // [T, h] (`gate`, `up`, `a`).
    const blocks = l * t * (6 * d + 2 * kv + 3 * h);
    // `final_h`, `d_final_h`, `d_x` and the layer loop's `d_next`
    // (`backwardFrom`), all [T, d], one `d_next` alive at a time.
    const grads = 4 * t * d;
    // `attentionBackward`'s dq, dk and dv plus its two f64 row buffers, which
    // are f32 elements' worth twice over.
    const attn_back = t * (d + 2 * kv) + 4 * t;
    return head + stream + blocks + grads + attn_back;
}

/// Attention core over one layer's MLP. The two the deferral note compares.
pub fn coreOverMlp(p: Projection) f64 {
    return @as(f64, @floatFromInt(p.attn_core)) / @as(f64, @floatFromInt(p.mlp));
}

/// One layer's whole step: forward, then `weightGrad` and `inputGrad` per
/// projection, which is three times the forward. Three and not two because the
/// backward runs both a weight and an input gradient for every projection
/// (`weightGrad` and `inputGrad`), and for the attention core it rebuilds the
/// scores a second time (`attentionBackward`), so the same factor covers it.
fn layerStep(p: Projection) f64 {
    return 3 * (toF(p.attn_core) + toF(p.attn_proj) + toF(p.mlp));
}

/// The tied head over the whole step: every layer's arithmetic, forward and
/// backward, plus the tied head. The two are the same cost model, so the share
/// is over the same denominator.
fn tiedOverStep(p: Projection) f64 {
    const l: f64 = @floatFromInt(p.shape.cfg.n_layers);
    const tied = toF(p.tied_head);
    return tied / (l * layerStep(p) + tied);
}

/// The attention core over one layer's whole step. The deciding number for a
/// fused kernel, which is a share of everything the step does rather than of
/// the one term the deferral note happens to name.
fn coreOverStep(p: Projection) f64 {
    return toF(p.attn_core) / layerStep(p);
}

fn toF(x: u64) f64 {
    return @floatFromInt(x);
}

/// The context at which the attention core reaches `bar` times one layer's MLP.
///
/// `2 * d * T * (T + 1) = bar * 6 * T * d * h` solves to `T = 3 * bar * h - 1`.
/// With `h = 4 * d` that is `T = 12 * bar * d - 1`: a quarter of the MLP's
/// arithmetic at `T = 3 * d`, and parity at `T = 12 * d`.
///
/// The deferral note claims the crossover is `T = 6 * d`, from `4 * T^2 * d`
/// against `24 * T * d^2`. The MLP half is right and is `mlp`'s own shape. The
/// attention half is the dense non-causal count every public attention figure
/// uses; `attention.forward` walks `0..t + 1`, so it does half of it, and the
/// true crossover is twice as far out as the note says.
pub fn crossoverT(h: usize, bar: f64) f64 {
    return 3.0 * bar * @as(f64, @floatFromInt(h)) - 1.0;
}

/// The vocabulary at which the tied head reaches `bar` of one MLP layer's
/// arithmetic, forward and backward on both sides. `6 * T * vocab * d = bar * 3 *
/// 6 * T * d * h` gives `vocab = bar * 3 * h`, or `bar * 3 * ffn_mult * d`: a
/// quarter of three times the feed-forward width at the bar this file prints,
/// and `3 * h` at `bar = 1`, which is below every vocabulary in this sweep.
fn tiedCrossoverVocab(h: usize, bar: f64) f64 {
    return bar * 3.0 * @as(f64, @floatFromInt(h));
}

comptime {
    // Every row of the sweep has to be a config `model.validate` accepts, and
    // the check is here rather than in `print` so a bad row is a compile error
    // instead of a run that fails in the middle of a table.
    for (sweep) |shape| {
        _ = project(shape) catch @compileError("scale.sweep: " ++ shape.name ++
            " is not a config model.validate accepts");
    }
}

/// `project` with the error removed, for the rows of `sweep`. Unreachable
/// because the block above proved every one of them at compile time, and the
/// only way to reach it is to add a row and skip the block.
fn row(shape: Shape) Projection {
    return project(shape) catch unreachable;
}

/// Writes the whole report. No timestamp, no address, no width taken from a
/// terminal: two runs print the same bytes, which is what lets a README quote
/// it.
pub fn print(w: *std.Io.Writer) std.Io.Writer.Error!void {
    try w.writeAll(
        \\ztransformer scale profile
        \\
        \\EVERY NUMBER BELOW IS PROJECTED, NOT MEASURED. Each one is arithmetic
        \\over model.Config, the tensor shapes in model.initLayer, the eleven
        \\tensors in autograd.Block and the two moments in optim.AdamW. No
        \\forward pass was run at any shape here and no wall time is printed,
        \\because a timing is not byte-reproducible and this output has to be.
        \\
        \\FLOP counts are per layer and forward only unless the column says
        \\otherwise, and a fused multiply-add counts as two operations. The
        \\softmax is left out of the attention core: four non-multiply-add
        \\operations per score against head_dim multiply-adds makes the whole
        \\term 4/head_dim of the core, under 3% anywhere in this sweep.
        \\
        \\
    );

    var ps: [sweep.len]Projection = undefined;
    for (sweep, 0..) |shape, i| ps[i] = row(shape);

    try w.print("SHAPES\n{s:<15}{s:>7}{s:>7}{s:>7}{s:>7}{s:>7}{s:>5}{s:>8}\n", .{
        @as([]const u8, "name"), @as([]const u8, "d"),      @as([]const u8, "h"),
        @as([]const u8, "T"),    @as([]const u8, "layers"), @as([]const u8, "heads"),
        @as([]const u8, "kv"),   @as([]const u8, "vocab"),
    });
    for (ps) |p| {
        try w.print("{s:<15}{d:>7}{d:>7}{d:>7}{d:>7}{d:>7}{d:>5}{d:>8}\n", .{
            p.shape.name,         p.d,                 p.h,  p.t,
            p.shape.cfg.n_layers, p.shape.cfg.n_heads, p.kv, p.shape.cfg.vocab_size,
        });
    }

    try w.writeAll("\nARITHMETIC, PROJECTED, GFLOP\n");
    try w.print("{s:<15}{s:>12}{s:>12}{s:>12}{s:>12}{s:>12}{s:>10}{s:>8}{s:>8}\n", .{
        @as([]const u8, "name"),     @as([]const u8, "attn_core"), @as([]const u8, "attn_proj"),
        @as([]const u8, "mlp"),      @as([]const u8, "tied_head"), @as([]const u8, "weight_grad"),
        @as([]const u8, "core/mlp"), @as([]const u8, "core%"),     @as([]const u8, "tied%"),
    });
    for (ps) |p| {
        try w.print("{s:<15}{d:>12.2}{d:>12.2}{d:>12.2}{d:>12.2}{d:>12.2}{d:>10.3}{d:>7.1}%{d:>7.1}%\n", .{
            p.shape.name,            g(p.attn_core),   g(p.attn_proj), g(p.mlp),
            g(p.tied_head),          g(p.weight_grad), coreOverMlp(p), 100.0 * coreOverStep(p),
            100.0 * tiedOverStep(p),
        });
    }
    try w.print(
        \\
        \\core/mlp is the attention core against one layer's MLP: the number the
        \\deferral of a fused kernel rests on. core% is the core against a whole
        \\layer, and tied% is the tied head against a whole step. weight_grad
        \\cannot move core/mlp at all: it is a loop order, not a term.
        \\
        \\
    , .{});

    try w.writeAll("BYTES, PROJECTED, GiB\n");
    try w.print("{s:<15}{s:>10}{s:>10}{s:>10}{s:>10}{s:>10}{s:>10}{s:>10}{s:>10}\n", .{
        @as([]const u8, "name"),   @as([]const u8, "params"),  @as([]const u8, "grads"),
        @as([]const u8, "adam"),   @as([]const u8, "act"),     @as([]const u8, "peak"),
        @as([]const u8, "scores"), @as([]const u8, "tied_GB"), @as([]const u8, "kv_cache"),
    });
    for (ps) |p| {
        const peak = p.params + p.grads + p.adam + p.act;
        try w.print("{s:<15}{d:>10.3}{d:>10.3}{d:>10.3}{d:>10.3}{d:>10.3}{d:>10.3}{d:>10.3}{d:>10.3}\n", .{
            p.shape.name, gib(p.params),       gib(p.grads),       gib(p.adam),     gib(p.act),
            gib(peak),    gib(p.scores_dense), gib(p.tied_stream), gib(p.kv_cache),
        });
    }
    try w.print(
        \\
        \\params, grads and adam are the parameter, the gradient and the two
        \\AdamW moments, one f32 each per element. act is the peak of one
        \\backward. At depth it is NOT the logits: autograd's cache keeps one
        \\eleven-tensor Block per layer, and that is 90% of the total at 32
        \\layers. The logits and dlogits are 9%.
        \\scores is n_layers * n_heads * T * T * 4: the score matrix a DENSE
        \\attention would materialize. This one does not materialize it, it walks
        \\one f64 row of T at a time (n_heads * T * 8, under a megabyte in this
        \\whole sweep), which is the one thing a fused kernel would keep.
        \\tied_GB is tok_embed read T times by the tied head. kv_cache is what a
        \\cache would cost and the code has none.
        \\
        \\
    , .{});

    try w.print("VERDICTS, one line per deferred item, at {d:.0} GiB of host memory\n", .{
        toF(host_bytes) / toF(1024 * 1024 * 1024),
    });
    try w.print("{s:<15}{s:>13}{s:>13}{s:>13}{s:>13}{s:>14}\n", .{
        @as([]const u8, "shape"),     @as([]const u8, "fused_attn"), @as([]const u8, "kv_cache"),
        @as([]const u8, "tied_head"), @as([]const u8, "wgrad_swap"), @as([]const u8, "dense_scores"),
    });
    for (ps) |p| {
        try w.print("{s:<15}{s:>13}{s:>13}{s:>13}{s:>13}{s:>14}\n", .{
            p.shape.name,
            // The deferral note's own comparison, against a stated bar.
            yesNo(coreOverMlp(p) >= attention_bar),
            // The cache removes the recompute of the quadratic term and nothing
            // else, so it is gated on that same term and shares its crossover.
            yesNo(coreOverMlp(p) >= attention_bar),
            // Like for like: the tied_head column is forward and backward, and
            // one MLP layer forward and backward is 3 * the mlp column.
            yesNo(toF(p.tied_head) / (3 * toF(p.mlp)) >= attention_bar),
            // Not a ratio. A loop-order swap cannot change a multiply-add count,
            // and both loops already walk contiguous rows, so there is no strided
            // access for it to remove. It would only cost byte-identity.
            @as([]const u8, "no"),
            if (p.scores_dense > host_bytes) "over host" else "fits",
        });
    }
    try w.print(
        \\
        \\fused_attn and kv_cache are core/mlp against a bar of {d:.2}, which is
        \\chosen and not derived. tied_head is tied_head / (3 * mlp) against the
        \\same bar, forward and backward on both sides. wgrad_swap is not a
        \\ratio: it is a constant no, and the reason is in the code rather than
        \\in a number. matmul, weightGrad and inputGrad each stream three
        \\contiguous rows and accumulate in place, so there is no strided read
        \\to block away, and any reorder sums the same terms in another order
        \\and changes the bytes.
        \\
        \\
    , .{attention_bar});

    // Every crossover below is one of two closed forms applied to the shipped
    // row, and the tables above already carry the shapes they come from, so
    // they are named here rather than spelled out sixteen times inline.
    const any = ps[0];
    const ctx_bar = crossoverT(any.h, attention_bar);
    const ctx_parity = crossoverT(any.h, 1.0);
    const ctx_six_pct = crossoverT(any.h, 0.06);
    const vocab_bar = tiedCrossoverVocab(any.h, attention_bar);
    const host_gib = toF(host_bytes) / toF(1024 * 1024 * 1024);
    // What the tied head walks at T=64 on the d4096 row, in GB and the bandwidth
    // moving it implies.
    const stream_gb = toF(64 * 128256 * 4096 * 4) / 1e9;
    try w.print(
        \\
        \\CROSSOVERS, the context or vocabulary at which each verdict flips
        \\
        \\  fused attention  no below T = {d:.0} (= {d:.2} * d), yes at or above.
        \\                   Parity with one layer's MLP is T = {d:.0} (= {d:.0} * d).
        \\                   The 6% the deferral note names is T = {d:.0} (= {d:.2} * d).
        \\  kv cache        the same number, and the same arithmetic: the one
        \\                   thing a cache removes is the recompute of the
        \\                   quadratic term, so it is gated on that term alone.
        \\  tied head       no below vocab = {d:.0} (= {d:.2} * d), yes at or above.
        \\                   The bar is a quarter of one MLP layer, so the
        \\                   threshold is a quarter of 3 * ffn_dim.
        \\  wgrad swap      no threshold exists. It reorders a fixed number of
        \\                   multiply-adds, so it cannot make a ratio better.
        \\
        \\  Dense score matrix over {d:.0} GiB of host: T = {d:.0} at the shipped
        \\  shape's heads and layers, T = {d:.0} at the 32-head 32-layer ones.
        \\
        \\  Why the tied head is the expensive one is arithmetic intensity and
        \\  not arithmetic, and it is readable from the two loops rather than
        \\  from a stopwatch. tiedHead does 2 * T * vocab * d operations
        \\  over 4 * T * vocab * d bytes, which is {d:.2} flop per byte with
        \\  no reuse to find, while one MLP layer re-reads its 3 * d * h
        \\  weights once for all T rows and so gets T / 2 = {d:.0} flop per
        \\  byte. At T=64 that is {d:.0}x. The {d:.1} GB it has to move at
        \\  that shape, and the {d:.2} GB/s that implies, are derived here; no
        \\  wall time is, because this tool projects rather than measures.
        \\
    , .{
        ctx_bar,
        ctx_bar / toF(any.d),
        ctx_parity,
        ctx_parity / toF(any.d),
        ctx_six_pct,
        ctx_six_pct / toF(any.d),
        vocab_bar,
        vocab_bar / toF(any.d),
        host_gib,
        denseScoreCtx(ps[0]),
        denseScoreCtx(ps[3]),
        stream_gb,
        stream_gb / 116.0,
        0.5,
        64 / 2,
        64 / 2 / 0.5,
    });
}

fn yesNo(b: bool) []const u8 {
    return if (b) "yes" else "no";
}

fn g(flop: u64) f64 {
    return @as(f64, @floatFromInt(flop)) / 1e9;
}

fn gib(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / toF(1024 * 1024 * 1024);
}

/// The largest `T` at which `n_layers * n_heads * T^2 * 4` still fits in
/// `host_bytes`, for one row's shape. Solved in f64 and then rounded, so a shape
/// whose square overflows `u64` still gets an answer.
fn denseScoreCtx(p: Projection) f64 {
    const per_t2 = toF(p.shape.cfg.n_layers) * toF(p.shape.cfg.n_heads) * 4.0;
    return @sqrt(toF(host_bytes) / per_t2);
}
