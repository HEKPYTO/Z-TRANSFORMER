# src

Thirteen modules, plus the `main.zig` binary, `lib.zig` and the `tests.zig` root. 29 `.zig`
files: 7,123 lines outside the 13 `*_test.zig` files, 4,244 inside them.

    wc -l src/*.zig | grep -v _test.zig | tail -1    # 7123 total
    wc -l src/*_test.zig | tail -1                    # 4244 total

Every module that allocates takes a `std.mem.Allocator` first and returns an error union. Two take
no allocator because they build nothing: `tensor.matmul` reads it off `a` and `loss.forward`
allocates nothing at all. No module holds global state or spawns a thread. Every reduction runs in
a fixed order, so two runs in one build configuration are bit-identical.

In-tree code reaches the implementation by relative path, `@import("tensor.zig")`. Only
`src/tests.zig` imports the `ztransformer` module, and only for `version` and `name`. `lib.zig`
deliberately exports no modules, because a file that is both re-exported by the library and
imported directly puts one file in two Zig modules, which the build refuses.

## Numerics

| Symbol | Signature | Purpose |
|---|---|---|
| `tensor.Tensor` | `{ data: []f32, rows, cols, allocator }` | Dense row-major `f32`. 2D only, by choice. |
| `tensor.Tensor.init` | `init(allocator, rows, cols) !Tensor` | Allocates `rows * cols` zeros. Errors on overflow. |
| `tensor.Tensor.deinit` | `deinit(*Tensor) void` | Frees the buffer. |
| `tensor.Tensor.at` | `at(Tensor, r, c) f32` | Reads one element. Panics out of range. |
| `tensor.Tensor.set` | `set(*Tensor, r, c, v) void` | Writes one element. Panics out of range. |
| `tensor.Tensor.row` | `row(*Tensor, r) []f32` | Mutable view of row `r`. |
| `tensor.Tensor.rowConst` | `rowConst(Tensor, r) []const f32` | Immutable view of row `r`. |
| `tensor.Tensor.fill` | `fill(*Tensor, v) void` | Sets every element. |
| `tensor.matmul` | `matmul(a, b) !Tensor` | `[m,k] @ [k,n] -> [m,n]`. Errors on shape mismatch. No allocator argument. |
| `norm.forward` | `forward(allocator, x, weight) !Tensor` | RMSNorm, eps 1e-5, per row, weight fused. |
| `rope.forward` | `forward(allocator, x, pos, theta, head_dim) !Tensor` | Rotary embedding, Llama-3 half-split. Row `r` sits at `pos + r`. |
| `mlp.forward` | `forward(allocator, x, w_gate, w_up, w_down) !Tensor` | SwiGLU feed-forward. |
| `mlp.silu` | `silu(z: f32) f32` | `z * sigmoid(z)`, safe in the negative tail. |
| `attention.Config` | `{ n_heads, n_kv_heads, head_dim }` | Grouped-query shape. |
| `attention.forward` | `forward(allocator, q, k, v, cfg) !Tensor` | Causal GQA, two passes over the scores. |
| `loss.forward` | `forward(logits, targets) !f64` | Mean cross-entropy, `logsumexp` form. No allocator argument. |

`rope.forward` takes the head width because the row is `n_heads` head blocks laid end to end, each
`head_dim` wide, and each rotates on its own: element `i` pairs with `i + head_dim / 2` of the same
head, never with the neighbour `i + 1` the way the original paper interleaves. A width of zero or
odd is `error.OddHeadDim` and a row that is not a whole number of head blocks is
`error.DimensionMismatch`; neither is a silent truncate.

`attention.forward` walks the scores twice per query row. The first pass takes the row max, the
second exponentiates against it and divides by the denominator. The max is subtracted rather than the
exponentials clamped, so the softmax is unchanged and every exponent stays in range. Scores are `f64`
and narrow to `f32` only on the store.

`at` and `set` panic on an out-of-range index. They are typed `f32` and `void`, so an error return
is not available, and the guard is explicit rather than a bare slice index because `at(0, 3)` on a
three-column tensor would otherwise read `data[3]`, a valid element of a different row. The panic is
a precondition violation and stays live in `ReleaseFast`. An out-of-range index panics in `at`, `set`
and the two row views, and nowhere else in the codebase. It is deliberate: shapes come from `Config`
and from internal arithmetic, not from data. Everywhere a value from data reaches an index, the check
returns an error instead: `model.forward` and `autograd.backward` on a token id
(`error.TokenOutOfRange`), `loss.forward` and `autograd.dLossDLogits` on a target id
(`error.TargetOutOfRange`), and `tokenizer.decode` on an id it was asked to rebuild from
(`error.TokenOutOfRange`). Each widens the `u32` to `usize` before comparing, which is lossless on
every Zig target and is what stops a corrupt id from wrapping down into the valid range.

## Training

| Symbol | Signature | Purpose |
|---|---|---|
| `data.Batch` | `{ inputs, targets, allocator }` | One window of `ctx` inputs and `ctx` targets. |
| `data.Corpus` | `{ train, val }` | Subslices of a caller's token array. No `deinit`, it owns nothing. |
| `data.split` | `split(tokens, val_fraction) !Corpus` | Positional split, before any shuffling. |
| `data.Batcher` | `init(allocator, tokens, ctx, seed)`, `next`, `reset`, `deinit` | Yields shuffled batches; `next` returns null when spent. |
| `train.Config` | `{ model, epochs, ctx, lr, warmup, weight_decay, max_grad_norm, seed, log_every }` | One run's settings, validated on the way in. |
| `train.Row` | `{ step, train_loss, val_loss, lr }` | One logged step. `writeCsv` turns a run's rows into the curve. |
| `train.run` | `run(allocator, cfg, train_tokens, val_tokens) !Result` | The whole loop: batch, forward, loss, backward, clip, rate, step, clear. |
| `train.Result` | `{ train_loss, val_loss, steps, params, rows, allocator }`, with `deinit` | The run's numbers plus the trained weights, so a checkpoint can be written straight out. |
| `train.writeCsv` | `writeCsv(path, rows) !void` | Header plus one line per row, truncating the file first. |
| `optim.AdamW` | `init(allocator, like)`, `step(p, g, lr, weight_decay)` | Decoupled weight decay, bias-corrected moments. |
| `optim.cosineLR` | `cosineLR(step, total, warmup, base_lr) f32` | Linear warmup then cosine to zero. Never negative. |
| `optim.clipByNorm` | `clipByNorm(grads, max_norm) f32` | Scales in place by the GLOBAL norm. Returns the PRE-clip norm. |

`data.Batcher` shuffles batch order, never the tokens inside a batch. Shuffling inside a batch would
destroy the sequence and train on noise. `reset` draws from the batcher's live PRNG rather than
re-seeding it, so every epoch is a different permutation and two runs of one seed still agree epoch
for epoch; a fresh PRNG per reset would replay the first epoch's order for all of them, which is a
shuffle that stops shuffling. `Corpus` deliberately has no `deinit`: `split` returns subslices, and
a deinit that frees nothing misstates ownership.

`clipByNorm` returns the norm it measured before scaling, so the caller reads back the value the
decision was made on. A non-finite norm comes back unscaled: the scaling is guarded by
`norm > max_norm`, and a NaN fails that test, so there is no scale to divide by.

`train.run` refuses to carry a number it cannot trust. `lr`, `max_grad_norm`, `weight_decay`,
`log_every`, `epochs` and the length of the training stream are all checked before the first step
(`error.BadLr`, `error.BadMaxGradNorm`, `error.BadWeightDecay`, `error.BadLogEvery`,
`error.NoEpochs`, `error.NoTrainingBatches`), each written as a negated comparison so a NaN is
rejected with them. In the loop, a non-finite loss on a training or a validation batch returns
`error.NonFiniteLoss` and a non-finite gradient norm returns `error.NonFiniteGradient`, both before
the step is applied, so one bad step cannot write NaN into every weight and then log a NaN curve for
the rest of the run. The norm is checked because `clipByNorm` cannot scale it: the gradients would
leave the clip still NaN and the next step would spread them across every parameter.

Validation is measured at every epoch end by `evalLoss`, over a fixed eight batches of a batcher
that shares nothing with the training stream but the parameters. It runs no backward pass and no
optimizer call.

## Model

| Symbol | Signature | Purpose |
|---|---|---|
| `model.Config` | `{ n_layers, n_heads, n_kv_heads, head_dim, n_ctx, vocab_size, ffn_mult }` | Model shape. |
| `model.dModel` | `dModel(cfg) usize` | `n_heads * head_dim`. |
| `model.ffnDim` | `ffnDim(cfg) usize` | `ffn_mult * dModel(cfg)`. |
| `model.defaultConfig` | `defaultConfig() Config` | 4 layers, 4 heads, 2 kv heads, head_dim 32, ctx 256. |
| `model.Layer` | 9 tensors | `attn_norm`, `wq`, `wk`, `wv`, `wo`, `mlp_norm`, `w_gate`, `w_up`, `w_down`. |
| `model.Params` | `{ tok_embed, layers, final_norm }` | Tied embeddings, so there is no `lm_head`. |
| `model.initParams` | `initParams(allocator, cfg, seed) !Params` | Deterministic from `seed`. |
| `model.forward` | `forward(allocator, p, cfg, tokens) !Tensor` | Returns logits `[T, vocab]`. |

Pre-norm: the norm sits inside the residual branch, not on the sum. Tied embeddings mean `tok_embed`
takes gradient from both the input lookup and the output projection, and the second path is the one
that is easy to miss. Norm weights initialise to 1.0, not 0.0: at zero every branch output is zero,
so no gradient reaches any of the 28 projection tensors or the embedding, and only the norm weights
move. Both PRNGs name `Xoshiro256` explicitly rather than reaching for `DefaultPrng`, whose stream
Zig documents as an implementation choice rather than a guarantee.

## Gradients

| Symbol | Signature | Purpose |
|---|---|---|
| `autograd.LayerGrads` | 9 tensors | Mirrors `model.Layer`, one gradient each. |
| `autograd.Grads` | `{ tok_embed, layers, final_norm }` | Mirrors `model.Params`. Accumulates. |
| `autograd.zeroGrads` | `zeroGrads(allocator, like: Params) !Grads` | Exact zeros, shaped like the parameters. |
| `autograd.dLossDLogits` | `dLossDLogits(allocator, logits, targets) !Tensor` | `(softmax - onehot) / T`. |
| `autograd.backward` | `backward(allocator, p, g, cfg, tokens, dlogits) !void` | Accumulates into `g`. Never zeroes it. |
| `gradcheck.Mismatch` | `{ layer: ?usize, field, index, analytic, numeric, diff, budget }` | One failing parameter element. |
| `gradcheck.Report` | `{ allocator, groups, floor, mismatch }`, with `deinit` | The whole sweep, as data. |
| `gradcheck.compare` | `compare(allocator, cfg, p, tokens, targets, g) !Report` | Central differences, no printing, no tolerance argument. |
| `gradcheck.line` | `line(allocator, m) ![]u8` | One diagnostic line for one mismatch. |
| `gradcheck.report` | `report(w: *std.Io.Writer, r: Report) std.Io.Writer.Error!void` | Writes the table to the writer it is given. |
| `gradcheck.checkAll` | `checkAll(allocator, cfg, p, tokens, targets) !void` | Runs one backward pass, then prints and fails on mismatch. |

Gradients are hand-written per module, not produced by a tape. A tape would need every tensor to
become a graph node, which rewrites every numerics module and changes every signature they expose.
Hand-written backward leaves those untouched and makes each gradient independently checkable. The
cost is one backward function per forward function, and that cost is smaller than the rewrite.

`checkAll` is silent when every parameter matches, and prints the full table plus a diagnostic line
naming the parameter, index, analytic value, numeric value, difference and budget when one does not.
`report` takes a writer rather than naming the process's stderr, because a printer that can only
write to stderr cannot be tested: the only way to cover one is to let its output escape, and inside
a test those bytes land on the channel the build runner reads, which fails the build. Naming the
writer makes the table assertable on captured bytes, and `checkAll` is the one caller that hands it
the real stderr.

### The budget is derived, not chosen

`compare` takes no tolerance. An element passes when

    |analytic - numeric| <= budget

where `budget` comes from `floorOf`, and nothing in it was picked:

- the loss is `(1 / T) * sum_t (logsumexp z[t] - z[t][target])`, so every `f32` logit carries a
  representation error of at most `f32_epsilon` (2^-24) of itself, and the largest logit of the
  three evaluations that element is differenced over bounds all of them;
- the `T * vocab` of those roundings are independent, so they compose in quadrature as
  `sqrt(T * vocab)` and not as a worst-case sum `T * vocab`. Summing is the case where every error
  is maximal and in the same direction, and it overstates the resolvable precision by `sqrt(vocab)`;
- the mean divides by `T`;
- the difference of two losses carries that error, and the difference divides it by `2 * step`.

so `budget = sqrt(vocab / T) * f32_epsilon * logit_scale / (2 * step)`. It is read per element, off
the largest of the three evaluations that element is differenced over, because a base point whose
logits vanish still has a gradient and reading the scale off the base point alone would report a
floor of zero there, which is not a resolution but a claim that `f32` resolved something it did not.

Two limits are worth stating plainly. The derivation bounds the logit-representation term only, not
the forward pass's own `f32` roundings, so the floor overstates the disagreement a correct gradient
can have, by a wide margin rather than a close one: on the suite's one-layer fixture a clean sweep's
largest gap comes in under a thousandth of the scale it is read against. And an element whose true
gradient is far below the floor cannot be checked elementwise in `f32` at all; no derived budget can
see it, and a chosen one that could would be a chosen one. The suite pins the floor's useful scale
rather than a tolerance: it is under a tenth of one percent of the largest gradient in the model, so
a one-percent error on that element is more than ten floors past the budget and a wrong gradient is
named rather than absorbed.

`compare` returns the table as data and takes the gradient tensor as a parameter, so a test can
corrupt one element and assert on the returned `Mismatch` rather than scraping stderr. It also
restores `p` element by element as it goes, error path included, which is what let the `tol`
parameter go: the caller never has to put the parameters back, so no call site has to be told to.

## Tokenizer

| Symbol | Signature | Purpose |
|---|---|---|
| `tokenizer.byte_vocab_size` | `256` | Every byte is a token, so encoding is total. |
| `tokenizer.Merge` | `{ left: u32, right: u32 }` | One learned merge, named by the two ids it joins. |
| `tokenizer.Tokenizer` | `init`, `deinit`, `train`, `encode`, `decode`, `save`, `load` | Byte-level BPE. |

There is no pre-tokenizer, so merges may cross any byte boundary and the merge list is a property of
the corpus alone. That costs some compression against a GPT-2-style split. There is no `ranks` array
either: the rank is the merge index, merge `r` yields id `256 + r`, and a parallel array is state
that can go stale. The persisted file is the merge list and nothing else, so a saved vocabulary
cannot carry a byte table that disagrees with its merges; `load` treats the file as untrusted input
and refuses a forward reference, where a merge names an id built after it.

## Tests

One test file per module, collected by `comptime` blocks in `tests.zig`. Zig has no test globbing,
so a new test file is inert until it is named there, and nothing fails when a name goes missing.
`tests.zig` is 68 lines and holds 3 tests of its own, the third being the guard that closes that gap:
Zig 0.16 has no comptime filesystem, so it walks `src/` at test time against its own source embedded
with `@embedFile`, and fails with `error.TestUnreferencedTestFile` on any `*_test.zig` the blocks
above do not name. It is a test rather than a compile error, which means a cached run can skip it, so
a developer who adds a test file may have to re-run before believing a green. It also asserts the
walk found a directory, because an empty listing passes every check above it vacuously.

Expected values are hand-computed literals, never the output of another function in this repo. A test
comparing an implementation against itself passes when both halves are wrong. Three tests here were
written specifically to defeat a mutation that the original suite could not see: a 512-token
non-dyadic attention case that fails if the accumulator drops to `f32`, a 512-element RMSNorm row
that fails if `sum_sq` drops to `f32`, and a `init` overflow case that fails only in the release modes
where the multiply used to wrap.

Reproducibility is per build configuration and per host, and the distinction is measured rather
than assumed. Two runs at one seed in one build produce byte-identical output. Across optimization
levels it does not hold: Debug differs from Release by about one f32 ulp per step, because the
compiler contracts the element-wise accumulation loops into fused multiply-add under optimization
and not under Debug. One ulp is harmless over a hundred steps and unbounded over ten thousand, so
the training criterion is stated per build configuration. `@exp`, `@sqrt` and `@cos` additionally
resolve to the platform libm, so output is not comparable across libm versions either.
