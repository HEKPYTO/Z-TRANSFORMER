# src

Ten modules. 1,228 lines of implementation, 2,516 of tests.

Every numerics module takes a `std.mem.Allocator` first and returns an error union. No module holds
global state or spawns a thread. Every reduction runs in a fixed order, so two runs in one build
configuration are bit-identical.

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
| `tensor.matmul` | `matmul(a, b) !Tensor` | `[m,k] @ [k,n] -> [m,n]`. Errors on shape mismatch. |
| `norm.forward` | `forward(allocator, x, weight) !Tensor` | RMSNorm, eps 1e-5, per row, weight fused. |
| `rope.forward` | `forward(allocator, x, pos, theta) !Tensor` | Rotary embedding, Llama-3 half-split. Row `r` sits at `pos + r`. |
| `mlp.forward` | `forward(allocator, x, w_gate, w_up, w_down) !Tensor` | SwiGLU feed-forward. |
| `mlp.silu` | `silu(z: f32) f32` | `z * sigmoid(z)`, safe in the negative tail. |
| `attention.Config` | `{ n_heads, n_kv_heads, head_dim }` | Grouped-query shape. |
| `attention.forward` | `forward(allocator, q, k, v, cfg) !Tensor` | Causal GQA, single-pass softmax. |
| `loss.forward` | `forward(logits, targets) !f64` | Mean cross-entropy, `logsumexp` form. |

`at` and `set` panic on an out-of-range index. They are typed `f32` and `void`, so an error return
is not available, and the guard is explicit rather than a bare slice index because `at(0, 3)` on a
three-column tensor would otherwise read `data[3]`, a valid element of a different row. The panic is
a precondition violation and stays live in `ReleaseFast`. This is the one place in the codebase
where an out-of-range access panics rather than returning an error, and it is deliberate: shapes come
from `Config` and from internal arithmetic, not from data. The one place a value from data reaches
an index, `loss.forward` checking a target id against the vocabulary, does return an error.

## Training

| Symbol | Signature | Purpose |
|---|---|---|
| `data.Batch` | `{ inputs, targets, allocator }` | One window of `ctx` inputs and `ctx` targets. |
| `data.Corpus` | `{ train, val }` | Subslices of a caller's token array. No `deinit`, it owns nothing. |
| `data.split` | `split(tokens, val_fraction) !Corpus` | Positional split, before any shuffling. |
| `data.Batcher` | `init(allocator, tokens, ctx, seed)` | Yields shuffled batches; `next` returns null when spent. |
| `optim.AdamW` | `init(allocator, like)`, `step(p, g, lr, weight_decay)` | Decoupled weight decay, bias-corrected moments. |
| `optim.cosineLR` | `cosineLR(step, total, warmup, base_lr) f32` | Linear warmup then cosine to zero. Never negative. |
| `optim.clipByNorm` | `clipByNorm(grads, max_norm) f32` | Scales in place by the GLOBAL norm. Returns the norm. |

`data.Batcher` shuffles batch order, never the tokens inside a batch. Shuffling inside a batch would
destroy the sequence and train on noise. `Corpus` deliberately has no `deinit`: `split` returns
subslices, and a deinit that frees nothing misstates ownership.

## Model

| Symbol | Signature | Purpose |
|---|---|---|
| `model.Config` | `{ n_layers, n_heads, n_kv_heads, head_dim, n_ctx, vocab_size, ffn_mult }` | Model shape. |
| `model.dModel` | `dModel(cfg) usize` | `n_heads * head_dim`. |
| `model.ffnDim` | `ffnDim(cfg) usize` | `ffn_mult * dModel`. |
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
| `gradcheck.Mismatch` | `{ path, index, analytic, numeric, diff, budget }` | One failing parameter element. |
| `gradcheck.Report` | `{ rows, mismatch }` | The whole sweep, as data. |
| `gradcheck.compare` | `compare(allocator, cfg, p, tokens, targets, tol, g) !Report` | Central differences, no printing. |
| `gradcheck.line` | `line(allocator, m) ![]u8` | One diagnostic line for one mismatch. |
| `gradcheck.report` | `report(Report) void` | Prints the table. Only on demand. |
| `gradcheck.checkAll` | `checkAll(allocator, cfg, p, tokens, targets, tol) !void` | `compare`, then print and fail on mismatch. |

Gradients are hand-written per module, not produced by a tape. A tape would need every tensor to
become a graph node, which rewrites all five numerics modules and changes every signature they
expose. Hand-written backward leaves those untouched and makes each gradient independently
checkable. The cost is one backward function per forward function, and that cost is smaller than
the rewrite.

`checkAll` is silent when every parameter matches, and prints the full table plus a diagnostic line
naming the parameter, index, analytic value, numeric value, difference and budget when one does not.
The budget is not a chosen tolerance: it is the most an `f32` central difference of that loss can
resolve, derived from the loss scale and the step. A real gradient bug lands two orders of magnitude
above it, so the check discriminates instead of merely passing.

`compare` returns the table as data and takes the gradient tensor as a parameter. That split is what
lets a test corrupt one element and assert on the returned `Mismatch` rather than scraping stderr.
`checkAll` is a thin wrapper over it, and every existing call site is unchanged.

## Tokenizer

| Symbol | Signature | Purpose |
|---|---|---|
| `tokenizer.byte_vocab_size` | `256` | Every byte is a token, so encoding is total. |
| `tokenizer.Merge` | `{ pair, id }` | One learned merge. |
| `tokenizer.Tokenizer` | `init`, `deinit`, `train`, `encode`, `decode`, `save`, `load` | Byte-level BPE. |

There is no pre-tokenizer, so merges may cross any byte boundary and the merge list is a property of
the corpus alone. That costs some compression against a GPT-2-style split. There is no `ranks` array
either: the rank is the merge index, merge `r` yields id `256 + r`, and a parallel array is state
that can go stale.

## Tests

One test file per module, collected by a `comptime` block in `tests.zig`. Zig has no test globbing,
so a new test file is inert until it is named there. `tests.zig` is 26 lines and holds no module
tests of its own.

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
