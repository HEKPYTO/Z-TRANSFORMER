# Z-TRANSFORMER

A transformer built from scratch in Zig: RMSNorm, RoPE, grouped-query attention, SwiGLU and tied
embeddings, with hand-derived gradients checked against central differences. Everything runs on the
CPU in f32. The block is checked tensor by tensor against a Llama reference, and a
GPT-mini trains end to end. Both work today; the CUDA kernels do not exist yet.

On "Llama-shaped", precisely: the block has RMSNorm, RoPE at Llama-3's theta, grouped-query
attention, SwiGLU and no biases, and it matches the reference op by op. It is not a Llama-3 you
could scale up, and three things say so. The feed-forward width is `ffn_mult * d_model` with
`ffn_mult` an integer, so it is 4x where Llama-3 is 3.5x rounded to a multiple of 256 — the rule
cannot express Llama-3's at all. There is no RoPE scaling, so long context is out. The embeddings
are tied, where Llama-3 8B's are not. Everything is f32, not bf16, and the model has no KV cache,
so a position is always 0. At the 8B shape this does not run: one layer of the tied head alone
takes about two minutes at T=64, and the logits tensor at T=8192 is 4.2 GB.

Requires Zig 0.16.0, enforced by `build.zig` rather than by hope. There is nothing to install and
no package manager step. Tested on macOS on Apple Silicon; the CPU path is portable, but the CUDA
row below needs a Linux host with docker and a GPU. A fresh clone has no git hook, because
`core.hooksPath` is per-clone local config:

    git config core.hooksPath .githooks

| Command | What it does | Leaves behind |
|---|---|---|
| `zig build` | Builds the binary. | `zig-out/bin/ztransformer` |
| `zig build run` | Runs it. With no argument it prints the version banner; `zig build run -- train` reaches training. | nothing |
| `zig build train` | Trains one pass over a 64 KiB prefix of the corpus. | `outputs/loss.csv` |
| `zig build test` | Runs the test suite, and writes nothing at all when it passes. | nothing |
| `zig build verify` | The whole gate CI runs: `zig fmt --check`, the test suite in Debug and in ReleaseFast, the version banner, and a sha256 check that the corpus still hashes to the digest `data/README.md` documents. Silent when it passes. | nothing |
| `sh tools/removed/check.sh` | Compares the block against a Llama reference, tensor by tensor. Needs the pinned oracle in a repo-local virtualenv; see `tools/README.md`. | `tools/removed/report.csv` |
| `sh src/cuda/run-probe.sh` | Compiles and runs one CUDA kernel on an NVIDIA GPU, in a container, and checks its integer sum against a closed form. Needs a Linux host with docker and a GPU. See `src/cuda/README.md`. | nothing |

## Status

The numerics run on CPU f32. The five block ops and the gradients are real and tested, and the block
is checked against a Llama reference. The CUDA toolchain is proven end to end on a real
GPU, but no CUDA source implements a transformer operation yet, there is no fused attention kernel,
and no benchmark is published. No checkpoint is written: a run leaves a loss curve and no model. The
table above is the whole interface.

`zig build train` links its own ReleaseFast binary whatever `-Doptimize` says. One step is a dense
f32 forward and backward over a 256-token window, so the default Debug build turns a run of minutes
into one of hours. Its defaults are a smoke run rather than a recipe: 200 BPE merges, a 64 KiB corpus
prefix, 123 windows, one epoch, seed 7. The whole 1.1 MB corpus is 2211 windows, eighteen times the
work, so a full epoch belongs in a scheduled run.

`zig build train` takes about 90 to 100 seconds end to end on an Apple M-series Mac once the binary is
built, and about 120 seconds the first time, because that run also compiles it. That figure covers
the whole invocation: the build check, BPE training, tokenization, and 123 training steps. It is
stated as a range on purpose. A per-step rate would be more elegant and is not published, because the
fixed startup work does not divide into a step cost, and a number derived from it would be wrong in a
way that is invisible until someone checks. The full-corpus epoch is not timed here for the same
reason; at eighteen times the windows it is a scheduled run, not a figure in a README.

Wall time is not a reproducibility guarantee, and the loss curve is. This repo states its criterion
as one seed, one build configuration, one host, and `src/README.md` documents where that boundary
sits. Two runs at seed 7 in ReleaseFast on this host produce a byte-identical `outputs/loss.csv`.

Source lives in `src/`. Everything the repository generates lands in `outputs/`. Apache-2.0 licensed.

One measured training result is committed, and it is the whole of that kind. The default `zig build train` ends at a
train loss of 5.593455 and a validation loss of 5.154903; both are the last row of the committed
`outputs/loss.csv`, and the validation number is measured on held-out tokens the training batcher
never touches. That run is a 64 KiB single-epoch smoke run over 123 windows. It shows the loop runs
to completion and the loss falls, and it is not a benchmark, not a throughput figure, and not a
claim about model quality. There is no inference benchmark.

## Parity with Llama

`sh tools/removed/check.sh` runs the same block through `reference-library` 4.57.3 and compares the two
side by side, tensor by tensor. It is the check that says the arithmetic here is the arithmetic there,
rather than a claim that it is.

| | |
|---|---|
| Reference | `reference-library==4.57.3`, `torch` 2.14.0, CPU, float32, attention `eager` |
| Shape | d_model 64, 2 layers, 4 heads over 2 kv heads of 16, ffn 256, vocab 256, ctx 512, batch 1 |
| Sweep | sequence lengths 1, 8 and 257, at seeds 7 and 8. Six runs, 156 tensor rows, 532 argmax comparisons |
| Result | 14 tensor kinds inside their gates, worst case 2.1e-06 against a 2.0e-05 gate |
| Argmax | 532 of 532 rows pick the same token. Smallest reference top1-top2 margin 8.5e-04 |
| Record | `tools/removed/report.csv`, one row per tensor per run |

All fourteen gates and the argmax have to hold. The two are not ranked, and the argmax is not the
sharper of the two: on this sweep the smallest reference top1-top2 margin is 8.5e-04 while the
widest gate on the logits is 2e-4, so any run that passes the gates cannot have flipped a token. The
gates are what catch a real difference: perturbing one weight element by 1e-3 fails 51 of them, and
scaling a whole projection by a tenth of a percent still fails 16. The argmax earns its place by saying *why* a row failed: on any row where the
two disagree, the reference's own top1-top2 margin is printed, which turns "it failed" into "it
failed on a near-tie" or "it failed outright".

What it claims: that this forward pass and the reference's agree, on the sweep above, at a stated
set of pinned weights. What it does not claim: anything about initialisation, because the harness
pins the weights on both sides and never compares how they were drawn; anything about batched
forward, because there is no batch axis in this model and the harness runs batch 1; anything about a
KV cache, quantized weights, or CUDA; and the attention probabilities and the SwiGLU hidden state,
which `attention.forward` and `mlp.forward` reduce internally and never hand out. It also does not
claim sensitivity: this shape is small, and an error of about a tenth of a percent in a projection
passes. `tools/README.md` says what was measured. The oracle is
Python and lives outside `src/`; the export it reads is written by Zig alone, so no committed number
depends on it.
