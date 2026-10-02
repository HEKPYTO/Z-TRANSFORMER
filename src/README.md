# src

Seventeen modules, plus the `main.zig` binary, `lib.zig` and the `tests.zig` root. 36 `.zig` files
directly in `src/`, 16 of them `*_test.zig`. Three more sit in `src/cuda/`: `device.zig` and the two
CPU twins, `norm_twin.zig` and `attn_twin.zig`.

There is deliberately no line count here. Three of them have been written and all three were wrong
within a commit, the last by 64 lines, because any line count is invalidated by every commit that
adds a line and nothing in the build notices. The count that did survive being wrong is the file
count, which only moves when a file is added or removed.

The trap that made the first two wrong is worth keeping, because it is the same shape as a gate
that measures the wrong thing. `wc -l src/*.zig | grep -v _test.zig | tail -1` looks like a
non-test line count and is not: `grep -v` filters lines, and the `total` line that `wc` appends
does not contain `_test.zig`, so it survives and `tail -1` hands back the grand total of every
file including the tests. A version of this block used that command and labelled its output as the
non-test count, which is how 7,123 became 7,676 became 8,645 while the number underneath was
always the sum of both columns.

Every module that allocates takes a `std.mem.Allocator` first and returns an error union. Two take
no allocator because they build nothing: `tensor.matmul` reads it off `a` and `loss.forward`
allocates nothing at all. No module holds global state or spawns a thread. Every reduction runs in
a fixed order, so two runs in one build configuration are bit-identical.

In-tree code reaches the implementation by relative path, `@import("tensor.zig")`. Only
`src/tests.zig` imports the `ztransformer` module, and only for `version` and `name`. `lib.zig`
deliberately exports no modules, because a file that is both re-exported by the library and
imported directly puts one file in two Zig modules, which the build refuses.

One directory here is mostly not Zig. `src/cuda/` holds the CUDA source and the container recipe that
compiles and runs it. The container is the pinned toolchain rather than the host's: the benchmark
table in `src/cuda/README.md` was measured with it, and a repository that compiles with one CUDA
and measures with another describes a build that never ran. `zig build cuda-check` therefore
compiles three of the four CUDA sources through the same `cuda()` the runners use -- the units are
`norm`, `probe` and `attn`, and `attn_kernels.cu` arrives through the `#include` in `attn.cu` -- so
the directory is reached from the build graph. One file in it is Zig and the directory is not outside
the compiled surface: `src/tests.zig` imports `cuda/device.zig` by path, deliberately, so that its
inline tests are built. The two twins are in none of these builds; `run-attn.sh` and `run-norm.sh`
compile each with its own `zig build-exe`. `sh
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
| `mlp.forwardWith` | `forwardWith(allocator, x, w_gate, w_up, w_down, ?*Sink, layer) !Tensor` | The same, handing `gate`, `up` and the gated activation to the sink. `forward` is this with a null sink. |
| `mlp.silu` | `silu(z: f32) f32` | `z * sigmoid(z)`, safe in the negative tail. |
| `attention.Config` | `{ n_heads, n_kv_heads, head_dim }` | Grouped-query shape. |
| `attention.forward` | `forward(allocator, q, k, v, cfg) !Tensor` | Causal GQA, two passes over the scores. |
| `attention.forwardWith` | `forwardWith(allocator, q, k, v, cfg, ?*model.Sink, layer) !Tensor` | The same pass, materialising the `[T·n_heads, T]` softmax matrix and handing it to `sink`. `forward` is this with a null sink. The training step is **not** on that path: `train.zig` passes a live sink, so it materialises the matrix and pays for it. Only `evalLoss` and `gradcheck` use the null form. |
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

### The allocator `train.run` is handed

`train.run` frees every per-step buffer as the step ends: the cache `Cache.deinit` names, the logits,
the gradient of the logits, and the temporaries inside `backwardFrom`. None of that is a bulk reset,
so the allocator has to be one whose `free` does something, and `main.zig` used to hand it
`init.arena`, whose `free` is a no-op. Nothing here leaked in the sense the suite means by leaking —
`std.testing.allocator` is a real allocator, so `train_test.zig` caught a missing free on any of
those paths then and catches it now. It is only in the binary, where the one allocator in scope was
an arena, that each step's working set survived to the end of the process. `main.zig` now splits the
two the runtime already provides: the corpus, the vocabulary and the merges are read for the whole
process and stay on `init.arena`, the run goes to `init.gpa`, and `Result.deinit` returns the
trained weights to the same one.

`zig build peak-rss` is the gate, `verify` runs it, and the budget is 134,217,728 B — the
`peak_rss_budget` constant at the top of `build.zig`, beside the digests. It execs the ReleaseFast
binary itself under `/usr/bin/time -l`, compares the maximum resident set size that tool reports
against that one number, prints nothing when it passes, and prints the budget, the peak and the way
back when it does not. It measures on Darwin and on Linux, and on Linux it **fails** rather than skips when
`/usr/bin/time` is missing, because CI runs there and a green run that checked nothing is a claim
rather than a gate. The one silent exit is a platform that is neither.

The budget sits between two measured populations rather than beside either one. The fixed build read
45,694,976 to 49,070,080 over five runs, so 128 MiB is 2.7x the worst of them. The arena build read
387,661,824 at its lowest and 3,842,310,144 at its highest, so it is 2.9x under the lowest, and a
budget placed near the 3.72 GiB the defect is famous for would have passed on the run that read
387 MB. Peak resident size does not inflate under load the way a wall clock does, which is what makes
this gate runnable inside `verify` at all, and the 2.7x is headroom for a differing allocator and
libc rather than for a busy machine.

Measured on the shipped 123-step run, the ReleaseFast `ztransformer-train` binary on its own under
`/usr/bin/time -l`, every run taken. The second column says where each number came from, because one
of these rows can no longer be taken by any command in this tree.

| Allocator `train.run` is handed | Peak resident, every run | How it was taken |
|---|---|---|
| `init.gpa` | 49,070,080 / 48,529,408 / 47,841,280 / 47,316,992 / 45,694,976 B | `/usr/bin/time -l` on the binary, five runs; `zig build peak-rss` is that measurement and a budget |
| `init.arena` | 3,842,310,144 / 3,288,449,024 / 3,180,314,624 / 387,661,824 B | the same command on the pre-split `main.zig`, four runs. Observation: this tree has no build that takes them |

The gate's own two runs on the committed tree and on `src/main.zig` reverted to the arena, taken
through `zig build peak-rss` and reported by it:

| Build | Peak resident | Gate |
|---|---|---|
| `init.gpa`, committed | 48,955,392 B | green, 85,262,336 B under budget |
| `init.arena`, `src/main.zig` reverted | 993,738,752 B | red, 7.4x over budget |

That second row is the reason the budget is where it is, and it is not the 3.72 GiB this defect is
known by: the arena's peak is whatever the host chose to keep resident of an unbounded accumulation,
and on this run it reported 993 MB rather than 3,989 MB. A gate that only fires above 3.7 GiB would
have watched that run happen. The run itself is 2:20 here, and the same number on the same box is
what the fixture costs either way.

67x at the median of the first table, and the more useful half of the answer is the shape of its two
rows: the arena's peak is unstable across runs of the identical binary, because how much of an
unbounded accumulation the kernel keeps resident is the host's decision, while the gpa's peak is one
step's working set and moves 7%. Three more numbers were taken alongside those runs and no command in
this tree produces them now, so they are observations and the method is named: a 0.4 s RSS sampler
running next to the run — on the arena the resident set climbed 1,338 MB to 3,032 MB across the run's
second half at about 26 MB per step, and on the gpa it held 45.4 MB to 46.8 MB across the same window
— and the `user`, `sys` and page-reclaim fields of the same `/usr/bin/time -l` report, which read
`sys` 0.83 to 4.73 s against 0.50 to 0.92 s and page reclaims 244,000 to 258,000 against 11,000 to
17,000. `zig build peak-rss` keeps that report and throws the rest of it away, because those fields
move with the host and the budget is about one of them.

`/usr/bin/time -l zig build train` is not the way to measure this, and its number would have hidden
the defect completely: it read 570,769,408 B before this change and 561,856,512 B after it, because
`zig build` runs the binary as a grandchild and the grandchild's peak does not reach the number. The
binary has to be timed on its own, which is why the gate takes the built binary's path and execs it
rather than depending on the `train` step that would have run it as a grandchild. Both of those two
numbers are observations taken that way, the "after" one reproducible with
`/usr/bin/time -l zig build train` and the "before" one not, for the reason in the table above.

There is no step-time cost, and the host this was measured on could not have shown one. `user` time
across both arms spans 71 to 96 s and is a monotone function of `real` time across the two arms
alike, because the machine was at load 8 to 14 on eight cores throughout; at the two most comparable
samples the gpa arm was 3% the faster of the two. What is not contention is the `sys` time and the
page-reclaim count above, both of which fall with the memory.

`outputs/loss.csv` hashes to `f1dd5444...` either way, which is the point worth making explicit: an
allocator decides where the bytes live and not what they add up to. Nothing in the loop reads an
address — the batch order is an index array, the optimizer pairs parameters and gradients by
position, the clipper's single norm sums the flat view in index order, and every reduction walks a
fixed order — so there is no path by which a different `free` could reach the arithmetic.

### Step time, and what it is not

`zig build bench` is the command. It reports the median of N runs of the whole training run, divided
by the steps the run reports, so the denominator comes from the run rather than from a constant here:

```
$ zig build bench -Doptimize=ReleaseFast
bench  3 runs of 123 steps, user CPU seconds per step
       median 0.2793   min 0.2785   max 0.2808
       spread 0.8% of the median
       whole run including tokenizer startup, over the steps it did
```

That is the figure at `HEAD` after both unroll rounds. An earlier version of this block quoted
0.2934, which was measured before either round and was left sitting above a paragraph claiming
0.2721 — a stale transcript contradicting the number next to it by 7.8%. The 0.2721 below is the
same tree measured in a different sitting; this box's load average moves, and the honest reading is
that the step is **0.27 to 0.28** and either single figure is a measurement of one moment.

It is user CPU time and not wall clock, and getting that wrong was the first bug in it: macOS
`/usr/bin/time -l` prints real, user and sys on one line, so a pattern matching `.*user.*` captures
the **first** number on that line, which is real. On `sleep 1` it read 1.00 where user was 0.00,
and the first version of this table was wall clock under a label saying CPU. The pattern now
anchors on the column.

Three direct `/usr/bin/time -l` runs of the same binary read 0.2988, 0.2998 and 0.3038. `bench` has
read 0.2934 and 0.3031 in separate invocations, so it lands in and around that set rather than
systematically above or below it, and the two methods agree to within about 3%. Which direction any
single pair differs by is not stable and is not claimed to be: both were taken on a box whose load
average was moving, and that is the likeliest reason.

It reports and never gates, and the asymmetry with `peak-rss` is deliberate. A peak in bytes is a
property of the program, so it can have a threshold. A time in seconds is a property of the program
and the machine and the hour, so a threshold on it fires on somebody else's load average and teaches
the gate to be ignored. `-Dbench-runs=9` for a number you intend to quote.

Two rounds of unrolling the backward pass took the whole run from **0.4522 to 0.2721 CPU seconds per
step**, both figures the median of three `zig build bench` runs taken in one session on the same
machine at a load average of 4.6, with spreads of 1.8% and 0.4%. The loss curve is byte-identical at
`f1dd5444...` across both. An earlier version of this line quoted 0.563 and 0.373, read off a console
transcript at a load average of 15 to 19; the ratio was 1.51x and this one is 1.66x, and neither pair
should be compared with the other because the load differed. Those two are the numbers
`zig build bench` produced in one sitting, and the rule here is that a number either comes out of
that command or says where it came from.

**The rest of this section does not come out of `bench`, and an earlier version of it implied that
it did.** The per-call and per-phase figures below come from an instrumented copy that is not
committed — two `clock_gettime(PROCESS_CPUTIME_ID)` reads at function entry and exit, none inside
the loops. Treat them as observations with the method named, not as measurements a reader can
re-run. `bench` reports the whole run and nothing below its level.

The first round unrolled `weightGrad`'s column loop and `inputGrad`'s row loop eight ways, which
were 77% of the backward. The second went after `attentionBackward`, at an observed 10.54 ms a call
and took it to 6.11 ms. A step is about 272 ms, so 4.43 ms a call times 4 calls is 17.7 ms of it,
against an observed **18.2 ms reduction in the whole step** — 3% of the gain came from anywhere else. Note that 0.4522 to 0.2721 is the span of **both** rounds, not
of the second alone.

What made the second round different is that its split axis was neither a column nor a row. The two
gathered-row dot products split on `s`, giving eight independent chains of length `dim`; the `d q`
and `d k`/`d v` loops split on `j`. The `d k`/`d v` case is the instructive one, because those
*accumulate* over `s`: the unroll is only over `j` within one `s`, and the `s` loop that does the
accumulating is untouched, so the read-modify-write order is unchanged. Unrolling a reduction is
what moves bytes, and the way not to is to never add one lane to another. Byte-identity is a
property of the split rather than a hope: each lane accumulates exactly the terms its scalar
counterpart did, in the same ascending order, and the lanes are never summed together. A `@Vector`
accumulator over the reduction axis would have reassociated the sum and moved the bytes.

What made it worth doing is that neither kernel was bandwidth-bound. `weightGrad` moves 12 bytes per
FMA and at the shipped shape the whole backward is a few megabytes of L1-resident rows, so bandwidth
was never the constraint — instruction issue was. LLVM declines to vectorise `w_row[j] +=` because it
is a read-modify-write through two pointers it cannot prove disjoint, and Zig has no `noalias` on a
function parameter, so the fix is code rather than a signature.

Three of the seven unrolled loops in `src/` reach a fast path that no finite-difference check
executed for most of this history, and the failures were of two kinds. The `s`-split loops in
`attentionBackward` need at least eight tokens to unroll at all, and every `checkAll` in the
repository was passed a four-token array, so their eight-way bodies ran only in `train_test.zig` --
which compares `train.run` against `clearedWalk`, two calls to the same `backward`, so a wrong
unroll is wrong identically in both arms. The `head_dim`-split bodies need a `head_dim` above
`lanes` and a kv head other than the first, and every fixture was `n_heads 1, n_kv_heads 1`, which
makes `h * dim` and `kv * dim` identically zero in the only check that reached them.

`unroll_s` in `src/autograd_test.zig` is nine tokens at `n_heads 4, n_kv_heads 2, head_dim 12`, and
it is the only fixture that executes those paths. The proof that it is load-bearing is a manual experiment rather than a committed
mutation, and there is no `tools/mutation` entry for it: no mutation in `mutate.zig` targets an
unrolled lane. Replacing one unrolled lane write with lane zero's value fails that test and
**only** it, 192 of 193 still passing, because every other gradcheck in the file never reaches that
code at all. A reviewer read the test names back and observed that `train_test.zig` does reach the
`s`-splits at `ctx 32`; whether its other assertions would catch a bad `d q` lane is not checkable
from the tree, so the "only it" is what was measured, not what is proven.

The scalar tail that finishes an odd width was covered by none of the fixtures either, because every fixture lands on a multiple of
eight by accident. `unroll_tail` in `src/autograd_test.zig` is the fixture that is not, at
`d_model` 12.

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
| `model.Name` | 18 values | Which intermediate a `Sink.put` call is about. |
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

`model.Sink` has two callers, `parity.run` and `autograd.Cache`, and changes no arithmetic: the
callback is the only thing the pass does that the arithmetic does not already do, so a null sink walks
the same statements in the same order, which `model_test.zig` asserts bit for bit. It is a function
pointer rather than a set of tensor pointers because the intermediates live in buffers the pass frees
before it returns, and a pointer recorded during the pass would dangle by the time a reader got to it.
Of the tensors the design names, all of them are gated, and the last one to close was the attention
probabilities. `attention.forwardWith` materialises them as `[T·n_heads, T]` — row `h·T + t` is the
softmax over the causal prefix of query `t` in head `h`, with the upper triangle left at the zero
`Tensor.init` writes, which is what a causal query's out-of-prefix keys are — and `attention.forward`
still does not, because it is that function with a null sink. Note which callers
that does and does not spare: the autograd pass and the training run reach attention through
`model.forwardWith` with a live sink and so do materialise it, while `attn_bench.zig` calls
`attention.forward` with no sink at all -- its own comment at line 98 says so -- so **the bench's ratio
is a per-call figure on the null-sink path and not on this one**; the null path is for a caller
that wants no intermediate at all. That is the only intermediate quadratic in the context, and what a
sink costs is 1 MiB at the shipped T=256 over four heads, freed before `forwardWith` returns — one
layer's worth of peak against a measured whole-model peak of about 47 MiB, and a step time that did
not move. `autograd.Cache` takes the name and stores nothing, which is true rather than convenient —
`attentionBackward` is handed `q_pos` and `k_pos` and rebuilds the forward's softmax row in f64 from
them. `mlp_gate`, `mlp_up` and `mlp_hidden` are exported into `data.bin` at `[T, ffn]` and compared
against external, which hands all three over at module boundaries: the two halves as `gate_proj` and
`up_proj` outputs, the product as `down_proj`'s input. `parity.run` writes raw little-endian f32 with a
text index rather than safetensors, because the oracle never calls `from_pretrained` and there is one
dtype and no mmap on either side. `tools/README.md` says what the harness checks and what it does not.

## Gradients

| Symbol | Signature | Purpose |
|---|---|---|
| `autograd.LayerGrads` | 9 tensors | Mirrors `model.Layer`, one gradient each. |
| `autograd.Grads` | `{ tok_embed, layers, final_norm }` | Mirrors `model.Params`. Accumulates. |
| `autograd.zeroGrads` | `zeroGrads(allocator, like: Params) !Grads` | Exact zeros, shaped like the parameters. |
| `autograd.dLossDLogits` | `dLossDLogits(allocator, logits, targets) !Tensor` | `(softmax - onehot) / T`. |
| `autograd.backward` | `backward(allocator, p, g, cfg, tokens, dlogits) !void` | Accumulates into `g`. Never zeroes it. Runs the forward pass itself. |
| `autograd.backwardFrom` | `backwardFrom(allocator, p, g, cfg, tokens, dlogits, c: *const Cache) !void` | The same walk, over a forward pass that has already run. |
| `autograd.Cache` | `{ init(allocator, p, cfg, tokens), deinit }`, holding a `model.Sink` | The eleven tensors per block the backward reads, plus the block input, copied out as the forward produces them. |
| `gradcheck.Mismatch` | `{ layer: ?usize, field, index, analytic, numeric, diff, budget }` | One failing parameter element. |
| `gradcheck.Report` | `{ allocator, groups, floor, headroom, mismatch }`, with `deinit` | The whole sweep, as data. `headroom` is `1 / max(diff / budget)`: the factor the budget could be multiplied by before the check would begin to fail. |
| `gradcheck.compare` | `compare(allocator, cfg, p, tokens, targets, g) !Report` | Central differences, no printing, no tolerance argument. |
| `gradcheck.line` | `line(allocator, m) ![]u8` | One diagnostic line for one mismatch. |
| `gradcheck.report` | `report(w: *std.Io.Writer, r: Report) std.Io.Writer.Error!void` | Writes the table to the writer it is given. |
| `gradcheck.checkAll` | `checkAll(allocator, cfg, p, tokens, targets) !void` | Runs one backward pass, then prints and fails on mismatch. |

Gradients are hand-written per module, not produced by a tape. A tape would need every tensor to
become a graph node, which rewrites every numerics module and changes every signature they expose.
Hand-written backward leaves those untouched and makes each gradient independently checkable. The
cost is one backward function per forward function, and that cost is smaller than the rewrite.

What a hand-written backward has instead of a tape is the question of where the forward's
intermediates come from. They used to be rebuilt: `backward` replayed the whole forward a second time,
layer by layer, which made a training step two forward passes and the larger of the two the one nobody
counted. The sink was already there and already handed every intermediate over as it was produced, so
`Cache` keeps them instead of recomputing them: `train.run` collects during the forward it was going to
run anyway, and `backwardFrom` reads what it collected. The walk that is gone is one forward pass,
which is a quarter of what the step runs: `zig build scale-profile` counts a layer's step as
`3 * (attn_core + attn_proj + mlp)`, the forward once and each of the two backward products once
more, so a backward that replays the forward runs four of those passes and the cached one runs three.
`backward` is the same walk with the forward inlined, which is the shape a standalone gradient
check wants, and the two are held to bit-identical gradients by a test — a tolerance would let a
tensor one of them never filled read as agreeing.

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

where `budget` comes from `floorOf`:

    budget = sqrt(2) * k * f32_epsilon * logit_scale / (2 * step * sqrt(T)),   k = 12

The loss is `(1 / T) * sum_t (logsumexp z[t] - z[t][target])`, so every `f32` logit carries a
representation error of at most `f32_epsilon` (2^-24) of itself, and the largest logit of the three
evaluations that element is differenced over bounds all of them. Cauchy-Schwarz over those independent
per-logit errors bounds the loss error at `eps * scale * sqrt(sum ((p_v - 1[v=tgt]) / T)^2) / T`, and
the sum under that root is `T * vocab * E_v[(p - onehot)^2]`. For a near-uniform softmax `p_v ~
1 / vocab`, so the per-token sum is `1 / vocab + (1 - 1 / vocab)^2 * (vocab - 1)`, which is
independent of `vocab`, and the whole thing collapses to `eps * scale / sqrt(T)`. The two losses a
difference is made of are separate evaluations, so their roundoffs add in quadrature: the `sqrt(2)`.

**There is no `sqrt(vocab)`.** An earlier version of this section had one, and it was wrong in both
directions at once. The derivative `dL/dz_v = (p_v - 1[v=tgt]) / T` already carries the
`1 / sqrt(vocab)` a near-uniform softmax imposes, so multiplying by `sqrt(vocab)` out here counts it
twice. That single term made the budget roughly a thousand times looser than the discrepancy it was
describing at `vocab 16` and 5.6% too tight at `vocab 256`, where a correct gradient on a
grouped-query fixture was reported as a mismatch. A check that cries wolf on a correct gradient is
worse than no check, because a maintainer learns to ignore the only instrument there is.

`k` is where the derivation ends and measurement begins, and it is measured rather than chosen. Over
550 sweeps (110 configs × 5 seeds, 2.2M elements) the largest per-element discrepancy ran 3.8 to 8.4
sigma, and at one fixed config sigma itself swung 500 to 1000 across seeds. No config-only formula can
be tight against a spread that depends on the weights, so `k` absorbs all of it. At `k = 12` the worst
of those 550 sweeps sat at **0.651 of budget**, which bounds the observed worst case with 35% to
spare; the same formula under the old one measured 2.02, failing 62 of the 550 sweeps outright.

Those 550 sweeps are an offline analysis, and this is the one number in this file that no command in
the repository reproduces: the sweep harness was not committed, and the 110-config grid it walked
does not exist here. What is committed is the constant it selected, `k = 12` in `src/gradcheck.zig`,
and the test that asserts against it. So the number above is the evidence for the constant rather
than a measurement of the tree, and it is quoted as one on purpose. Re-deriving `k` means
rebuilding the sweep, not re-running a command.

Three terms the derivation omits, and why:

| Omitted | Why |
|---|---|
| Truncation, `h^2 / 6 * f'''` | Richardson on the real `f32` forward measures it near `1e-6` at `h = 1e-3`, three orders below the roundoff term. If a future shape makes the two comparable, add a term for truncation rather than widening `k`; `k` measures weight spread and widening it to cover a different error hides both. |
| Gradient-path accumulation depth | Needs no term. At `d_model 32`, sweeping kv-group 1 to 4 over five draws, per-element sigma was flat with no trend. |
| FMA contraction | Was carried until measured, and is not real. An earlier claim that Debug and Release differ by about one ulp from contraction was wrong: `ReleaseFast` and `ReleaseSafe` are bit-identical, so Zig is not contracting these loops and there is nothing to carry. The `Debug` vs release difference is real and its mechanism remains unestablished; see [Reproducibility](#reproducibility). |

The budget is read per element, off the largest of the three evaluations that element is differenced
over, because a base point whose logits vanish still has a gradient and reading the scale off the base
point alone would report a budget of zero there, which is not a resolution but a claim that `f32`
resolved something it did not.

Two limits are worth stating plainly. An element whose true gradient is far below the budget cannot
be checked elementwise in `f32` at all; no derived budget can see it, and a chosen one that could
would be a chosen one. And a budget has to be bounded in both directions, which is what
`Report.headroom` (`1 / max(diff / budget)`, the factor the whole budget could be multiplied by
before the check would begin to fail) exists to make assertable: a wrong constant shows up as a
`headroom` near 1, and a budget inflated to absurdity shows up as a `headroom` in the hundreds while
`floor < gmax * 0.001` keeps failing. The suite pins both bounds rather than a tolerance, on a
grouped-query fixture where the grouped path sums four `dk` terms per kv head, so a one-percent
gradient error is named rather than absorbed.

The `headroom` is asserted as a **band** around a measured value, `2.62 ± 0.35`, and the band is the
fix rather than a retuned threshold. The assertion was a one-sided floor, `headroom > 1.5`, and a floor
cannot catch a budget that has been loosened: weakening multiplies the budget, which divides
`diff / budget`, which **raises** the headroom. The sweep at `k = 12` measures 2.6184, so a 1.5x
weakening reports 3.93 and passes a 1.5 floor with room to spare — while raising the floor to 3.93 to
catch it would report every correct gradient as a failure. The gate was pointed the wrong way, and
could only ever have caught a budget that had been *tightened*.

One note on the figures in this section, because they are the kind that get quoted onward: `2.6184`,
`2.6442`, `0.46 to 10.2`, `0.88` and `0.62` are **not** reproduced by any command in this repository.
No test prints them — `gradcheck_test.zig` asserts `2.62 +/- 0.35` and stops there — and the
sweep harness that produced them was never committed, the same as the 550-sweep analysis below. They
are recorded here with their method (a pinned-seed finite-difference sweep, `liveParams` drawing from
`0xbeef`, median over the grid), as are the figures derived from them — `3.93` at `f = 1.5` and
`26` at `f = 10` — and they are evidence for the committed constant `k = 12`, not measurements of
the current tree. What the tree does reproduce is the assertion.

Both sides are now asserted against a measurement rather than a preference. The fixture is pinned —
`liveParams` draws from the literal `0xbeef` — so the sweep is the same sweep every run, reading 2.6184
in ReleaseFast and 2.6442 in Debug. The 0.35 tolerance is about 13%: an order of magnitude above the 1%
spread between the two configurations, and 3.8x below the 50% a 1.5x weakening moves it by. A budget multiplied by `f` reports `2.6184 * f`, so the band catches any change beyond roughly
0.87x to 1.13x in either direction.

The 4000-draw study of the headroom under **redrawn** seeds, which runs 0.46 to 10.2, still governs
`k` and is unaffected: `checkAll` runs on trained weights, which are arbitrary draws, and the 0.46 end
is a correct gradient reported as a mismatch, so `k` is set from the worst draw rather than the median.
What that distribution must not do is size an *assertion* about a pinned draw, because a bound loose
enough to cover draws the test never takes is a bound that can no longer see `k` being changed at all.
Deriving the budget so this spread does not reach a false mismatch would be a real improvement to
`floorOf`; it is not what this is, and the two questions are now kept apart.

The broken `sqrt(vocab / T)` budget that the original assertion was written against still fails: 0.88
at this vocab and 0.62 at `vocab 16`, both outside the band.

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
| `scale.Projection` | 18 fields | FLOPs per term, element counts for parameters, gradients, AdamW state and activations, and the four byte counts. |
| `scale.coreOverMlp` | `coreOverMlp(Projection) f64` | The attention core against one layer's MLP. The number the fused-kernel deferral rests on. |
| `scale.crossoverT` | `crossoverT(h, bar) f64` | The `T` at which `core/mlp` reaches `bar`: `3 * bar * h - 1`. |
| `scale.print` | `print(w: *std.Io.Writer) !void` | The whole report to the writer it is given, never to a stream. |

Those seven are the whole public surface, because `main.zig` calls one of them. Everything else
`scale.zig` needs is private: the bar a deferred item has to clear (`attention_bar`, `0.25`, a
choice and named as one), the 32 GiB a `fits` verdict is a statement about (`host_bytes`), one
layer's whole step (`layerStep`), the tied head's share of a step (`tiedOverStep`), the core's
share of a step (`coreOverStep`), the tied head's vocabulary crossover (`tiedCrossoverVocab`) and
the dense score matrix's context ceiling (`denseScoreCtx`).

### What it says about the deferred work

Three items were deferred on arithmetic asserted in prose. This is the arithmetic, and one of the
three claims does not survive it.

The three tables below are quoted from `zig build scale-profile`, not written here, and the block
between the two markers is checked against the tool's own output by `zig build verify`. The check is
what makes the claim worth reading: a table that says it came from a tool and can drift from it is
worse than no table, because the drift is invisible until a reader believes it. Re-copy the block
from the tool when a `model.Config`, a tensor shape or the cache changes, and the gate fails until
you do.

<!-- scale-profile:begin -->
```
ARITHMETIC, PROJECTED, GFLOP
name              attn_core   attn_proj         mlp    tied/stp weight_grad  core/mlp   core%   tied%
shipped                0.02        0.03        0.10        0.20        0.13     0.167    3.9%   10.5%
ctx-1k                 0.27        0.10        0.40        0.81        0.50     0.667   11.6%    8.0%
ctx-4k                 4.30        0.40        1.61        3.22        2.01     2.667   22.7%    4.1%
d4096-ctx4k          137.47      343.60     1649.27    12910.67     1992.86     0.083    2.2%    5.9%
llama3-8b            549.82      687.19     3298.53    25821.34     3985.73     0.167    4.0%    5.6%
llama3-8b-32k       8796.36     2748.78    13194.14   103285.37    15942.92     0.667   11.9%    4.2%
at-parity-12d      19791.61     4123.17    19791.21   154928.06    23914.38     1.000   15.1%    3.6%

BYTES, PROJECTED, GiB
name               params     grads      adam       act      peak    scores  tied_GiB  kv_cache
shipped             0.004     0.004     0.008     0.014     0.031     0.004     0.125     0.000
ctx-1k              0.004     0.004     0.008     0.058     0.075     0.063     0.500     0.002
ctx-4k              0.004     0.004     0.008     0.232     0.248     1.000     2.000     0.008
d4096-ctx4k        30.958    30.958    61.916    42.524   166.356    64.000  8016.000     1.000
llama3-8b          30.958    30.958    61.916    85.047   208.879   256.000 16032.000     2.000
llama3-8b-32k      30.958    30.958    61.916   340.188   464.020  4096.000 64128.000     8.000
at-parity-12d      30.958    30.958    61.916   510.282   634.114  9216.000 96192.000    12.000

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
<!-- scale-profile:end -->

**The `T > 6 * d` claim is wrong by a factor of two, and it is the deferral's own arithmetic.**
`attention.forward` walks `0..t + 1`, so it computes half of a dense score
matrix, and the core is `2 * d * T * (T + 1)` rather than `4 * T^2 * d`. The MLP half of the
comparison is correct and is `mlp`'s own shape: three matmuls of a `[T, d]` input against
`[d, h]`, `[d, h]` and `[h, d]` is `6 * T * d * h`, which is `24 * T * d^2` at
`ffn_mult = 4`. So `core / mlp = (T + 1) / (3 * h)`, parity is `T = 12 * d` and not `6 * d`, and
the 6% the note names is `T = 0.71 * d` rather than whatever the dense count gives. The direction of
the deferral survives, and it was the right call anyway: the core is 3.9% of a layer's step at the
shipped shape and 4.0% at Llama-3 8B. The threshold was wrong.

The four deferred items and their thresholds. `zig build scale-profile` prints these as its
`CROSSOVERS` block, and this table is transcribed from it rather than gated by it: the automated
check in `build.zig` compares the `scale-profile:begin`/`end` block above against the tool's output
and would reject prose in between. So the transcription is a place this file can drift, and the
command to re-check it is the one named in `build.zig`. The `T >= 3 * d` column is the tool's
`no below T = 383 ... yes at or above` restated for this model's `d = 128` head width: `crossoverT`
is `3 * bar * h - 1` with `h = 4d`, so for `bar = 0.25` it is `3d - 1 = 383`, and the tool's first
affirmative `T` is 383. The table rounds that to `3 * d`, which is the same threshold stated for a
head width that cannot be fractional and is off by the one the `-1` leaves behind.

| Deferred item | Worth doing at | Deciding number |
|---|---|---|
| Fused IO-aware attention | `T >= 3 * d` | **Superseded.** `core/mlp >= 0.25` gives no at the shipped shape (0.167) and no at 8B (0.167), yes at 32k (0.667). The kernel exists and ran 98.7x at the shipped shape, between 103x and 160x across the other shapes, the floor withdrawn pending re-measurement. `core/mlp` is a share of arithmetic and not of time. See "Attention: the floor, and then the kernel" below. |
| KV cache | `T >= 3 * d` | The same term, and the same flaw: it is a share of arithmetic and not of time. The attention row above is the cautionary tale for this one. `src/kv_cache.zig` has landed and carries five tests, but nothing decodes through it -- there is no generation loop and the forward kernel's `q_offset` is exercised only at 0, the training shape, so a cached key is not yet attended to by a single-token query -- and the row is still unmeasured on time. It also costs 2 GiB at 8B and 12 GiB at the parity row. |
| Tied-head restructure | `vocab >= 3 * d` | Landed, see below. `tied / (3 * mlp) >= 0.25` at every row here, including the shipped one at 10.5% of the step. |
| `weightGrad` loop-order swap | never | Not a ratio. It reorders a fixed multiply-add count, so no shape improves it, and `matmul`, `weightGrad` and `inputGrad` already stream contiguous rows. |

The tied head was the one that should not have waited, and it is the one that did not. It is 10.5% of a
step at the shipped shape and 5.6% at 8B, and the reason is arithmetic intensity rather than an
arithmetic count: `tiedHead` does `2 * T * vocab * d` operations over `4 * T * vocab * d` bytes,
which is 0.5 flop per byte with no reuse in it, where one MLP layer re-reads its weights once for
all `T` rows and gets `T / 2`, which is `128` at the shipped `T = 256`. That is a 256x gap, and it is arithmetic a reader can
do off `model.tiedHead` and `model.initLayer`; no timing is claimed for it here, because no command in
this repository measures one and a wall time would outlive the shape it was taken at.

Low intensity cannot be fixed by reducing traffic when there is no reuse to recover, so the lever
is the other one: how fast the adds issue. One f64 accumulator per logit is a serial chain of `d`
adds that cannot retire faster than the adder's latency. The vocab loop now runs eight rows at a
time into eight accumulators. Eight is a choice and named as one: twelve and sixteen lanes were no
better on this host, and the benchmark that says so is not one this repository ships, so the width
is recorded as the plateau rather than as a measurement. Tiling the vocab loop and blocking over
`d` are both absent, and the reason recorded at the time was the size of the table rather than its
access pattern: at `vocab = 1024` the embedding table is 512 KiB.

The output is unchanged, which is the whole reason this is a refactor and not a numeric change.
Each accumulator sums a disjoint set of vocab rows and, within a row, still walks `i` ascending,
so every logit is the same f32 narrowing of the same f64 sum. `outputs/loss.csv` still hashes to
`f1dd5444...`. Reassociating the sum would be a different project with a different answer, and the
f64 accumulator is not negotiable here: it is the tied head that the softmax in `loss.forward` then
exponentiates. The tied-head *backward* in `autograd` is a separate loop, and was left alone for
the same reason this section's first paragraph gives for the shape: it is intensity, not count.

## Attention: the floor, and then the kernel

`zig build attn-bench` measures one call of `attention.forward` at five context lengths and prints it
beside the PCIe floor a GPU implementation would have to clear.

```
     ctx       calls  cpu_us_min   pcie_MiB    floor_us   ratio  verdict
  --------  ---------  ------------  ---------  ----------  --------  -------------------------------------------
       256         82       6084.07      0.375       64.46       94x  floor is below the cpu: the kernel's own cost decides
       512         20      25313.41      0.750      128.92      196x  floor is below the cpu: the kernel's own cost decides
      1024          5     103435.41      1.500      257.85      401x  floor is below the cpu: the kernel's own cost decides
      2048          3     422220.31      3.000      515.69      819x  floor is below the cpu: the kernel's own cost decides
      4096          3    1693164.72      6.000     1031.39     1642x  floor is below the cpu: the kernel's own cost decides
```

On the 32-core Linux host. `cpu_us_min` is the **minimum** per-call time over `calls` calls, because
contention and frequency scaling only make a call slower. `floor_us` is q, k and v in plus the result
out at 6.1 GB/s. **No run in this repository measures a PCIe rate.** `src/attn_bench.zig` derives that constant
from the norm table's own `gpu_e2e` minus `gpu`, and says so at `src/attn_bench.zig:36`; `src/attn_bench.zig` names where that constant
comes from, and it is the weakest number in the table.

**The ratio column is not a figure to quote to two significant digits.** Three sweeps of this table on
that host put the shipped row at 6084, 8344 and 10311 us, a 69% spread in host load alone. What is
stable across all three is the order of magnitude: two at the shipped window, three at 4096.

**That row is also not reproducible across sessions.** The `6084.07 us` over 82 calls printed above and
the `10707.91 us` over 47 calls in `src/cuda/README.md` are the same tool, the same shape, the same
host, 1.76x apart, from a later session. Neither number is deleted here: the pair is the record, and
what it establishes is that a single `zig build attn-bench` cell is a sample of a noisy quantity and
not a property of the code.

### What the kernel turned out to be: `sh src/cuda/run-attn.sh`

`src/cuda/attn_kernels.cu` is that kernel: one block per query and head, walking the causal prefix with
the running softmax in registers and no score matrix ever written to global memory. Its two backward
kernels are in that same file and are graded against `attentionBackward` by the same script, at three
separate gates, one each for dq, dk and dv; neither half is wired into `zig build train`. The file
beside it, `src/cuda/attn.cu`, is the benchmark harness and not a library: it `#include`s the kernels
and defines `main`, so it cannot be linked as one. The full table, the three-run reproducibility
measurement, the attack on its own gate and its known limitation are in `src/cuda/README.md`; the two
facts that belong here are these.

**The `gate` column is `ATTN_TOL` 1e-4 from `src/cuda/attn.cu`, and it belongs to none of the eighteen
per-tensor gates in `tools/removed/oracle.txt`** -- those run 2e-6 to 2e-4 against a Llama
reference, while the backward's `ATTN_BWD_TOL` 1e-5 grades this directory's own `attentionBackward`.
Three tolerance systems, no number in common, and a row passing here says nothing about the
block-parity claim, which rests entirely on `oracle.txt`.

```
shape          ctx     cpu_us   kernel_us   max_abs     gate   used    ratio  parity  argmax
ctx256          256   10884.32     103.442  5.960e-08     1e-04  0.060%    105.2x  ok   argmax 517
ctx4096        4096 3080129.43   18625.433  5.960e-08     1e-04  0.060%    165.4x  ok   argmax 517
llama3-T512     512 1551863.40   11817.677  7.451e-08     1e-04  0.075%    131.3x  ok   argmax 76157
attn: OK
attn: broken variant 1 was caught, as it must be
attn: broken variant 2 was caught, as it must be
attn: broken variant 3 was caught, as it must be
attn: broken variant 4 was caught, as it must be
```

**That transcript is abridged, three rows out of seven and with two tables' worth of output removed.**
The full run also prints the backward parity table with its three worst-case index columns, the
`ATTN_GROUP_Q=` line, the `configurations used:` block, the four backward variants each with the
`[dq/dk/dv ...]` signature it produced, and the line asserting that those four signatures differ.
`src/cuda/README.md` carries the seven-row forward sweep; neither file quotes the run whole.

At Llama-3's own geometry -- 32 heads over 8 kv heads at `head_dim` 128 -- the parity is `7.451e-08`,
0.075% of the gate. That row is there because a kernel only ever run at `head_dim` 32 has not been
shown to run at the width this project is about.

`max_abs` is `5.960e-08` and `7.451e-08`, which is 0.060% and 0.075% of the `1e-4` gate. An earlier
version of this file called that "one `f32` ulp at a value of 1.0". That was wrong: the ulp at 1.0 is
`1.19e-07`, and these outputs are weighted averages of V values in `[-0.5, 0.5]` so `|out|` sits well
below 1. The defensible statement is one or a few units in the last place at the magnitudes involved,
at every context length and both geometries. Note also that the comparison is **f32 against f32** --
`attention.zig` narrows to `f32` before the reference reaches disk -- so the CPU's `f64` accumulator
improves the reference rather than widening what is compared.

### The speedup, and why it is not the floor's number

The kernel ran **98.7x** faster at the shipped window and between 103x and 160x across the other six, a floor that
is withdrawn rather than restated until the table is re-measured: the ratio's denominator is a CPU call that has
read 6450.78 to 10770.23 us across sessions on one host, a spread of 1.67x, so no floor here was ever a property
of the kernel. It previously read
106x and 165x. Three runs put the `kernel_us` spread at 0.5% to 2.6% and the `cpu_us` spread at 0.1% to 1.4%
-- except at `ctx256`, where one `forward` call read 6089.81, 8572.32 and 10884.32 us, a 78.8% spread.
The published run is the third of three and the only one taken while the GPU read 0%, so its ratio is
three.

Comparing both ratios from that same run rather than from two: the floor promises 169x at the shipped
window and the kernel delivers 62% of it; at 4096 the floor promises 2986x on the twin's own CPU figure -- against the
1642x printed four lines above from `attn-bench`'s, which is the same disagreement the next paragraph
records -- and the kernel delivers 5.5%. The reason is the known limitation in `src/cuda/README.md`: every block re-reads the whole
causal prefix of K and V, so at 4096 the kernel moves 8.6 GB against 6.3 MB of compulsory traffic, a
1366x amplification. An earlier version of that figure said 17.2 GB and 2731x, which overcounted by
`n_kv_heads`: a block reads one kv head, not all of them. It is arithmetic and L2 bound, not bandwidth bound, which is why it is nowhere
near the bus.

One discrepancy is recorded rather than smoothed over. `attn-bench` measured one `forward` call at
4096 as 1.69e6 us in one session, and this table's twin measured 3.08e6 to 3.09e6 us in another -- a
factor of 1.8, both minimum-of-three, both correct about their own method.

**That was first written up as two tools disagreeing, and that framing was wrong.** The obvious
suspect was the input data: `attn-bench` fills q, k and v from one seed and the twin from three, so
the softmax sees a different distribution and `exp` sees a different argument range. A controlled
test varying only that -- same host, same method, same three-call minimum, three fill patterns --
refuted it:

```
same seed (attn-bench)          3104936 us
distinct seeds (twin)           3102682 us
distinct, k/v swapped           3100309 us
```

A 0.15% spread, and **3.10e6 us from both tools.** So the data is not the cause, the two tools do not
disagree, and what the factor of 1.8 actually records is that *one measurement of this function on
this host has read 1.69e6 and 3.10e6 at different times with no methodological difference identified.*
That is a worse thing to have found than a bug in a tool, because it means the number is not
reproducible across host states. Until it is explained, nothing above should be read to better than
an order of magnitude.

### It reverses the `scale-profile` verdict, and the projection was the weaker of the two

`scale-profile` answers `fused_attn` with `core/mlp >= 0.25`, which at the shipped shape is `0.167`, so
it says no, and the CUDA kernel wins over `attention.forward` by 98.7x at the shipped window, the floor being withdrawn pending re-measurement. Both are computed correctly, because they
measure different things: `core/mlp` is the share of an MLP layer's *arithmetic* that attention
contributes and it says nothing about how long either takes. `src/scale.zig` calls the bar "a choice,
and named as one".

The arithmetic and the time disagreed because the CPU path was a naive scalar `f64` loop. One call at
the shipped window is exactly `4 * 32 * (256 * 257 / 2) = 4.21 M` multiply-adds for QK and the same
again for PV, `8.42 M` in all. `scale-profile` prints the same quantity as `attn_core 0.02 GFLOP`,
which is those same 8.42 M multiply-adds counted as two operations each. Nothing was wrong with the
tool; using a share of arithmetic as if it were a share of time was the error.

The KV cache row above is gated on the same `core/mlp` term and this section does not speak to it. It
has landed -- `src/kv_cache.zig`, five tests -- and its reasoning still has the flaw the attention row
had: the threshold is not yet backed by a time, because nothing decodes through the cache. There is
no generation loop, and the forward kernel's `q_offset` is exercised only at 0, so a cached key cannot yet be attended to by
a single-token query.

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
`tests.zig` holds 3 tests of its own, the third being the guard that closes that gap:
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
`1.14e-13` for `f64` and `6.09e-5` for `f32`, with the measured divergence of `1.9e-7` between them.
That divergence is 1.7 million times the `f64` gate and 0.3% of the `f32` one, so an `f32`
accumulator is comfortably outside the first and comfortably inside the second, and the test can
tell the two apart in both directions. The reference is narrowed to `f32` before it is compared,
because that is what `matmul` returns: comparing an `f32` against an unrounded `f64` sum leaves the
`f32` store rounding on one side only, and that term alone is `9.3e-9` against a `1.14e-13` `f64`
gate, `8.2e4` times it. **Nearly five orders above**, which is what makes it the thing the test has
to be blind to. The conclusion never depended on the magnitude; this file said "two orders" until
an audit recomputed it, which was wrong by three.

### Two survivors that are not holes

`tools/mutation` leaves two survivors after that test. Neither is a coverage hole, and
neither is going to become one, so they are recorded here rather than left for the next person to
spend a day on. `clip-ge` is the `>` / `>=` question settled above: bit-identical, uncaughtable.
`norm-reassociate` is `v / rms * w[i]` written as `v * w[i] / rms`, and the measured answer is that
it is below the resolution of any tolerance that could be written down.

The reassociation touches no reduction, so its error is the rounding of one element's three
operations and cannot grow with `d`: measured at 1.1 to 1.6 `f32` ulp for `d` from 4 to 4096, flat.
That pair, and the `0.0` below, are hand-computed figures with no committed command behind them —
hand arithmetic on three rows, which is why they are quoted as observations rather than as
measurements of the tree.
The gate in `norm_test.zig` is `1e-6`, which is 8.4 ulp at a value of one, so catching it would mean
a sub-ulp tolerance. There is a second and stronger reason, and it is the reason no such test should
be written: every weight on the rows this argument rests on is a power of two — `0.5`, `1`, `2`,
`0.25` — and a power of two commutes exactly with a single `f32` division. One test in this file does
use the weight `3`; its input is all zero, so it agrees for an unrelated reason and says nothing about
this one. Measured on all three hand-computed rows, the
two spellings differ by `0.0`: they are bit-identical, so they are identical at a tolerance of zero
and not merely at `1e-6`. Catching this needs a non-dyadic weight, and then the divergence is one
ulp, which is a statement about `f32` and not about this function. The `tol` is left where it is.

Reproducibility is per build configuration and per host, and the distinction is measured rather
than assumed. Two runs at one seed in one build produce byte-identical output. Across optimization
levels it does not hold: Debug differs from Release by about one f32 ulp per step. One ulp is
harmless over a hundred steps and unbounded over ten thousand, so the training criterion is stated
per build configuration.

The mechanism was attributed to fused multiply-add in the element-wise accumulation loops, and that
attribution was wrong on a second count too: `ReleaseFast` and `ReleaseSafe` are bit-identical, so
Zig is not contracting `a * b + c` in these loops at all, and there is no contraction term for the
gradcheck budget to carry. Each of those loops was extracted and compiled both ways at a size that
shows the difference: `matmul`, `weightGrad` and `inputGrad` are bit-identical between Debug and
ReleaseFast. At full model scale the only tensors that move at all are `w_gate` and `w_down` in
layer 0; the forward pass, `tok_embed`, `wq`, `wk`, `attn_norm` and `final_norm` are all
bit-identical. The phenomenon is real and reproducible, the explanation for it is not established,
and it is recorded here as an observation rather than as a mechanism nobody has verified. `@exp`,
`@sqrt` and `@cos` additionally resolve to the platform libm, so output is not comparable across
libm versions either.
