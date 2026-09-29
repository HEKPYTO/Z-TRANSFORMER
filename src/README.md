# src

Fourteen modules, plus the `main.zig` binary, `lib.zig` and the `tests.zig` root. 31 `.zig` files:
3,316 lines outside the 16 `*_test.zig` files, 5,228 inside them.

    cat $(ls src/*.zig | grep -v _test) | wc -l    # 3316
    cat src/*_test.zig | wc -l                    # 5228

Those two commands are the source of the two numbers, and they are the `cat` form on purpose.
`wc -l src/*.zig | grep -v _test.zig | tail -1` looks equivalent and is not: `grep -v` filters
lines, and the `total` line that `wc` appends does not contain `_test.zig`, so it survives and
`tail -1` hands back the grand total of every file including the tests. An earlier version of this
block used that command and labelled its output as the non-test count, which is how 7,123 became
7,676 became 8,645 while the number underneath it was always the sum of both columns.

Every module that allocates takes a `std.mem.Allocator` first and returns an error union. Two take
no allocator because they build nothing: `tensor.matmul` reads it off `a` and `loss.forward`
allocates nothing at all. No module holds global state or spawns a thread. Every reduction runs in
a fixed order, so two runs in one build configuration are bit-identical.

In-tree code reaches the implementation by relative path, `@import("tensor.zig")`. Only
`src/tests.zig` imports the `ztransformer` module, and only for `version` and `name`. `lib.zig`
deliberately exports no modules, because a file that is both re-exported by the library and
imported directly puts one file in two Zig modules, which the build refuses.

One directory here is not Zig. `src/cuda/` holds the CUDA source and the container recipe that
compiles and runs it, because `nvcc` cannot be installed on a GPU host without root, and the CUDA
the distribution's NVIDIA repository ships is version-skewed against the one this repository targets.
No `src/*.zig` file imports it and `build.zig` does not reference it yet, so the other 30 `.zig`
files above are still the whole compiled surface, and their line counts have not moved. `sh
src/cuda/run-probe.sh` compiles the probe and runs it on the local GPU; `src/cuda/README.md` says
what that does and does not establish.

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
| `norm.forward` | `forward(allocator, x, weight) !Tensor` | RMSNorm, per row, weight fused. |
| `norm.eps` | `pub const eps: f64 = 1e-5` | The additive epsilon, public because the gradient and the parity export are the same constant. `autograd.normBackward` differentiating a private copy of it would differentiate a different function than the forward pass evaluates, and no finite difference would catch it. |
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

That `>` is not `>=`, and no test here can tell the two apart, so there is no test for it.
`tools/mutation`'s `clip-ge` makes the change and survives; the reason is arithmetic rather than a
missing assertion. The two spellings differ only when `norm == max_norm` exactly, and there
`scale = max / norm = 1` exactly in f64, so every element is multiplied by `1.0` and both return
the same norm. That is a bit-for-bit identical run, not a close one, and no tolerance catches a
difference of zero. A test that passed there would be passing for a reason unrelated to the
mutation, which is worse than no test.

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
| `model.forwardWith` | `forwardWith(allocator, p, cfg, tokens, ?*Sink) !Tensor` | The same pass, handing each intermediate to a `Sink`. `forward` is this with a null sink. |
| `model.Name` | 14 values | Which intermediate a `Sink.put` call is about. |
| `model.Sink` | `{ put }` | A callback, not a bag of pointers: the intermediates live in buffers the pass frees before it returns. |
| `parity.run` | `run(allocator, io, s: Sweep) !Summary` | Writes the weights, intermediates, token ids and shape to `outputs/parity/`. |
| `parity.runInto` | `runInto(allocator, io, out_dir, s: Sweep) !Summary` | `run` with the output directory as an argument. One caller overrides it: `removed_test.zig`, which would otherwise overwrite the sweep the Python oracle reads with a one-layer fixture. |
| `parity.sweep` | `sweep() Sweep` | The shape and the sequence lengths and seeds the harness compares. |

Pre-norm: the norm sits inside the residual branch, not on the sum. Tied embeddings mean `tok_embed`
takes gradient from both the input lookup and the output projection, and the second path is the one
that is easy to miss. Norm weights initialise to 1.0, not 0.0: at zero every branch output is zero,
so no gradient reaches any of the 28 projection tensors or the embedding, and only the norm weights
move. Both PRNGs name `Xoshiro256` explicitly rather than reaching for `DefaultPrng`, whose stream
Zig documents as an implementation choice rather than a guarantee.

`model.Sink` exists for one caller, `parity.run`, and changes no arithmetic: the callback is the
only thing the pass does that the arithmetic does not already do, so a null sink walks the same
statements in the same order. It is a function pointer rather than a set of tensor pointers because
the intermediates live in buffers the pass frees before it returns, and a pointer recorded during the
pass would dangle by the time a reader got to it. The two tensors the design names and this cannot
reach are the attention probabilities and the SwiGLU hidden state: `attention.forward` and
`mlp.forward` reduce them internally and return only the result, so reaching either means changing
those two files. `parity.run` writes raw little-endian f32 with a text index rather than
safetensors, because the oracle never calls `from_pretrained` and there is one dtype and no mmap on
either side. `tools/README.md` says what the harness checks and what it does not.

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

## Scale

`zig build scale-profile` prints what the code's own formulas imply at shapes the code cannot be run
at. It is a projection, not a benchmark, it says so in its own output, and it times nothing: no
forward pass runs at any shape here, so no wall time can leak into a number. Everything it prints is
arithmetic over `model.Config`, the shapes in `model.initLayer`, the eleven tensors in
`autograd.Block` and the two moments in `optim.AdamW`, and every term in `scale.zig` names the line
it came from. The output carries no timestamp and no address, so two runs are byte-identical and a
README can quote it.

| Symbol | Signature | Purpose |
|---|---|---|
| `scale.Shape` | `{ name, cfg }` | One row of the sweep. A real `model.Config`, so a row cannot claim a shape the model would refuse to build. |
| `scale.sweep` | 7 rows | The shipped shape, then the same width at 1k and 4k context, then a 32-layer `d_model 4096` at 4k, 8k and 32k, then the row that sits on the crossover. |
| `scale.project` | `project(Shape) !Projection` | Every term for one shape. Rejects a config `model.validate` rejects. |
| `scale.Projection` | 20 fields | FLOPs per term, element counts for parameters, gradients, AdamW state and activations, and the four byte counts. |
| `scale.coreOverMlp` | `coreOverMlp(Projection) f64` | The attention core against one layer's MLP. The number the fused-kernel deferral rests on. |
| `scale.layerStep` | `layerStep(Projection) f64` | One layer's whole step: forward plus `weightGrad` and `inputGrad`, so three times the forward. |
| `scale.tiedOverStep` | `tiedOverStep(Projection) f64` | The tied head against every layer's step plus itself. |
| `scale.crossoverT` | `crossoverT(h, bar) f64` | The `T` at which `core/mlp` reaches `bar`: `3 * bar * h - 1`. |
| `scale.tiedCrossoverVocab` | `tiedCrossoverVocab(h, bar) f64` | The `vocab` at which the tied head reaches `bar` of one MLP layer: `bar * 3 * h`. |
| `scale.attention_bar` | `0.25` | The bar a deferred item has to clear. A choice, and named as one. |
| `scale.host_bytes` | 32 GiB | The memory a `fits` verdict is a statement about. |
| `scale.print` | `print(w: *std.Io.Writer) !void` | The whole report to the writer it is given, never to a stream. |

### What it says about the deferred work

Three items were deferred on arithmetic asserted in prose. This is the arithmetic, and one of the
three claims does not survive it.

The tables below are copied from `zig build scale-profile`, not written here.

```
ARITHMETIC, PROJECTED, GFLOP
name              attn_core   attn_proj         mlp   tied_head weight_grad  core/mlp   core%   tied%
shipped                0.02        0.03        0.10        0.20        0.13     0.167    3.9%   10.5%
ctx-1k                 0.27        0.10        0.40        0.81        0.50     0.667   11.6%    8.0%
ctx-4k                 4.30        0.40        1.61        3.22        2.01     2.667   22.7%    4.1%
d4096-ctx4k          137.47      343.60     1649.27    12910.67     1992.86     0.083    2.2%    5.9%
llama3-8b            549.82      687.19     3298.53    25821.34     3985.73     0.167    4.0%    5.6%
llama3-8b-32k       8796.36     2748.78    13194.14   103285.37    15942.92     0.667   11.9%    4.2%
at-parity-12d      19791.61     4123.17    19791.21   154928.06    23914.38     1.000   15.1%    3.6%

BYTES, PROJECTED, GiB
name               params     grads      adam       act      peak    scores   tied_GB  kv_cache
shipped             0.001     0.001     0.002     0.003     0.007     0.004     0.125     0.000
ctx-1k              0.001     0.001     0.002     0.013     0.017     0.063     0.500     0.002
ctx-4k              0.001     0.001     0.002     0.050     0.054     1.000     2.000     0.008
d4096-ctx4k         7.740     7.740    15.479    10.830    41.788    64.000  8016.000     1.000
llama3-8b           7.740     7.740    15.479    21.660    52.618   256.000 16032.000     2.000
llama3-8b-32k       7.740     7.740    15.479    86.641   117.599  4096.000 64128.000     8.000
at-parity-12d       7.740     7.740    15.479   129.961   160.919  9216.000 96192.000    12.000

VERDICTS, one line per deferred item, at 32 GiB of host memory
shape             fused_attn     kv_cache    tied_head   wgrad_swap  dense_scores
shipped                   no           no          yes           no          fits
ctx-1k                   yes          yes          yes           no          fits
ctx-4k                   yes          yes          yes           no          fits
d4096-ctx4k               no           no          yes           no     over host
llama3-8b                 no           no          yes           no     over host
llama3-8b-32k            yes          yes          yes           no     over host
at-parity-12d            yes          yes          yes           no     over host
```

**The `T > 6 * d` claim is wrong by a factor of two, and it is the deferral's own arithmetic.**
`attention.forward` walks `0..t + 1` (attention.zig:54), so it computes half of a dense score
matrix, and the core is `2 * d * T * (T + 1)` rather than `4 * T^2 * d`. The MLP half of the
comparison is correct and is `mlp`'s own shape: three matmuls of a `[T, d]` input against
`[d, h]`, `[d, h]` and `[h, d]` (model.zig:376) is `6 * T * d * h`, which is `24 * T * d^2` at
`ffn_mult = 4`. So `core / mlp = (T + 1) / (3 * h)`, parity is `T = 12 * d` and not `6 * d`, and
the 6% the note names is `T = 0.71 * d` rather than whatever the dense count gives. The direction of
the deferral survives, and it was the right call anyway: the core is 3.9% of a layer's step at the
shipped shape and 4.0% at Llama-3 8B. The threshold was wrong.

The three thresholds, as the tool prints them:

| Deferred item | Worth doing at | Deciding number |
|---|---|---|
| Fused IO-aware attention | `T >= 3 * d` | `core/mlp >= 0.25`. No at the shipped shape (0.167) and no at 8B (0.167), yes at 32k (0.667). |
| KV cache | `T >= 3 * d` | The same term, because the only thing a cache removes is the recompute of the quadratic part. It also costs 2 GiB at 8B and 12 GiB at the parity row. |
| Tied-head restructure | `vocab >= 3 * d` | `tied / (3 * mlp) >= 0.25`. Yes at every row here, including the shipped one at 10.5% of the step. |
| `weightGrad` loop-order swap | never | Not a ratio. It reorders a fixed multiply-add count, so no shape improves it, and `matmul`, `weightGrad` and `inputGrad` already stream contiguous rows. |

The tied head is the one that should not have waited. It is 10.5% of a step at the shipped shape
and 5.6% at 8B, and the 116 s figure is not an arithmetic problem at all: `tiedHead` does
`2 * T * vocab * d` operations over `4 * T * vocab * d` bytes, which is 0.5 flop per byte with no
reuse in it, where one MLP layer re-reads its weights once for all `T` rows and gets `T / 2 = 32` at
`T = 64`. That is a 64x gap and it is the whole of the 116 s.

## Tokenizer

| Symbol | Signature | Purpose |
|---|---|---|
| `tokenizer.byte_vocab_size` | `256` | Every byte is a token, so encoding is total. |
| `tokenizer.Merge` | `{ left: u32, right: u32 }` | One learned merge, named by the two ids it joins. |
| `tokenizer.Tokenizer` | `init`, `deinit`, `train`, `encode`, `decode` | Byte-level BPE. |

There is no pre-tokenizer, so merges may cross any byte boundary and the merge list is a property of
the corpus alone. That costs some compression against a GPT-2-style split. There is no `ranks` array
either: the rank is the merge index, merge `r` yields id `256 + r`, and a parallel array is state
that can go stale. The vocabulary is never written to disk: `src/main.zig` trains the merges in
memory on every run, so there is nothing to persist and nothing to keep in step with the merges.

## Tests

One test file per module, collected by `comptime` blocks in `tests.zig`. Zig has no test globbing,
so a new test file is inert until it is named there, and nothing fails when a name goes missing.
`tests.zig` is 72 lines and holds 3 tests of its own, the third being the guard that closes that gap:
Zig 0.16 has no comptime filesystem, so it walks `src/` at test time against its own source embedded
with `@embedFile`, and fails with `error.TestUnreferencedTestFile` on any `*_test.zig` the blocks
above do not name. It is a test rather than a compile error, which means a cached run can skip it, so
a developer who adds a test file may have to re-run before believing a green. It also asserts the
walk found a directory, because an empty listing passes every check above it vacuously.

Expected values are hand-computed literals, never the output of another function in this repo. A test
comparing an implementation against itself passes when both halves are wrong. Four tests here were
written specifically to defeat a mutation that the original suite could not see: a 512-token
non-dyadic attention case that fails if the accumulator drops to `f32`, a 512-element RMSNorm row
that fails if `sum_sq` drops to `f32`, a `init` overflow case that fails only in the release modes
where the multiply used to wrap, and a 256-by-512-by-128 matmul that fails if the `k` reduction is
carried in `f64`.

That last one is worth the shape, because the five smaller matmul tests could not have caught it and
no tighter version of them would. They reduce `k <= 3` terms, where an `f32` and an `f64` reduction
agree to the last bit. The accumulator's error grows with `k`, and `w_down` at the shipped config
multiplies a 256-token batch through the 512-long reduction into 128 columns, which is the widest
reduction the model has. Its two gates are the same formula with the machine epsilon of the
accumulator swapped in, `(k - 1) * e` against the sum of the absolute products, so nothing is picked:
`1.14e-13` for `f64` and `6.09e-5` for `f32`, with the measured divergence of 1.98e-7 between them.
An `f64` accumulator lands on the exact reduction and the divergence is `0`; an `f32` one is two
hundred times under the `f32` gate. The reference is narrowed to `f32` before it is compared,
because that is what `matmul` returns — comparing an `f32` against an unrounded `f64` sum leaves the
store rounding on one side only, and that term alone is `9.3e-9`, which passes the `f64` gate and
makes the test blind to the thing it is for.

### Two survivors that are not holes

`tools/mutation` leaves three survivors after that test. Two of them are not coverage holes, and
neither is going to become one, so they are recorded here rather than left for the next person to
spend a day on. `clip-ge` is the `>` / `>=` question settled above: bit-identical, uncaughtable.
`norm-reassociate` is `v / rms * w[i]` written as `v * w[i] / rms`, and the measured answer is that
it is below the resolution of any tolerance that could be written down.

The reassociation touches no reduction, so its error is the rounding of one element's three
operations and cannot grow with `d`: measured at 1.1 to 1.6 `f32` ulp for `d` from 4 to 4096, flat.
The gate in `norm_test.zig` is `1e-6`, which is 8.4 ulp at a value of one, so catching it would mean
a sub-ulp tolerance. There is a second and stronger reason, and it is the reason no such test should
be written: every weight in `norm_test.zig` is a power of two — `0.5`, `1`, `2`, `0.25` — and a power
of two commutes exactly with a single `f32` division. Measured on all three hand-computed rows, the
two spellings differ by `0.0`: they are bit-identical, so they are identical at a tolerance of zero
and not merely at `1e-6`. Catching this needs a non-dyadic weight, and then the divergence is one
ulp, which is a statement about `f32` and not about this function. The `tol` is left where it is.

Reproducibility is per build configuration and per host, and the distinction is measured rather
than assumed. Two runs at one seed in one build produce byte-identical output. Across optimization
levels it does not hold: Debug differs from Release by about one f32 ulp per step. One ulp is
harmless over a hundred steps and unbounded over ten thousand, so the training criterion is stated
per build configuration.

The mechanism was attributed to fused multiply-add in the element-wise accumulation loops, and that
attribution was wrong. Each of those loops was extracted and compiled both ways at a size that
shows the difference: `matmul`, `weightGrad` and `inputGrad` are bit-identical between Debug and
ReleaseFast. At full model scale the only tensors that move at all are `w_gate` and `w_down` in
layer 0; the forward pass, `tok_embed`, `wq`, `wk`, `attn_norm` and `final_norm` are all
bit-identical. The phenomenon is real and reproducible, the explanation for it is not established,
and it is recorded here as an observation rather than as a mechanism nobody has verified. `@exp`,
`@sqrt` and `@cos` additionally resolve to the platform libm, so output is not comparable across
libm versions either.
