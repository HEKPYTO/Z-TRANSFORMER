# src

Ten modules. 1,228 lines of implementation, 2,516 of tests.

Every numerics module takes a `std.mem.Allocator` first and returns an error union. No module holds
global state or spawns a thread. Every reduction runs in a fixed order, so two runs on one host are
bit-identical across optimization levels.

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

Reproducibility is per-host. `@exp`, `@sqrt` and `@cos` resolve to the platform libm, so
bit-identical output holds on one machine across optimization levels, not across libm versions.
