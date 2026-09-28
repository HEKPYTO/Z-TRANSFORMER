# Z-TRANSFORMER

A transformer built from scratch in Zig: RMSNorm, RoPE, grouped-query attention, SwiGLU and tied
embeddings, with hand-derived gradients checked against central differences. Everything runs on the
CPU in f32. The target is a Llama-3-class block with a measured max-abs-diff against a external
reference, and a GPT-mini that trains end to end. Training works today; the parity table and the
CUDA kernels do not exist yet.

Requires Zig 0.16.0. There is nothing to install and no package manager step.

| Command | What it does | Leaves behind |
|---|---|---|
| `zig build` | Builds the binary. | `zig-out/bin/ztransformer` |
| `zig build run` | Runs it. With no argument it prints the version banner; `zig build run -- train` reaches training. | nothing |
| `zig build train` | Trains one pass over a 64 KiB prefix of the corpus. | `outputs/loss.csv` |
| `zig build test` | Runs the test suite, and writes nothing at all when it passes. | nothing |
| `zig build verify` | The whole gate CI runs: `zig fmt --check`, the test suite in Debug and in ReleaseFast, the version banner, and a sha256 check that the corpus still hashes to the digest `data/README.md` documents. Silent when it passes. | nothing |

## Status

CPU f32 only. There is no CUDA source in the tree, no fused attention kernel, no external parity
report, and no saved checkpoint: a run leaves a loss curve and no model. The five block ops and the
gradients are real and tested; the CUDA and parity halves of the goal are targets. The table above is
the whole interface.

`zig build train` links its own ReleaseFast binary whatever `-Doptimize` says. One step is a dense
f32 forward and backward over a 256-token window, so the default Debug build turns a run of minutes
into one of hours. Its defaults are a smoke run rather than a recipe: 200 BPE merges, a 64 KiB corpus
prefix, 123 windows, one epoch, seed 7. The whole 1.1 MB corpus is 2211 windows, eighteen times the
work, so a full epoch belongs in a scheduled run.

`zig build train` takes about 95 to 120 seconds end to end on an Apple M-series Mac, measured across
several runs. That figure covers the whole invocation: the build check, BPE training, tokenization,
and 123 training steps. It is stated as a range on purpose. A per-step rate would be more elegant and
is not published, because the fixed startup work does not divide into a step cost, and a number
derived from it would be wrong in a way that is invisible until someone checks. The full-corpus epoch
is not timed here for the same reason; at eighteen times the windows it is a scheduled run, not a
figure in a README.

Wall time is not a reproducibility guarantee, and the loss curve is. This repo states its criterion
as one seed, one build configuration, one host, and `src/README.md` documents where that boundary
sits. Two runs at seed 7 in ReleaseFast on this host produce a byte-identical `outputs/loss.csv`.

Source lives in `src/`. Everything the repository generates lands in `outputs/`. Apache-2.0 licensed.

One measured result is committed, and it is the whole of it. The default `zig build train` ends at a
train loss of 5.593455 and a validation loss of 5.154903; both are the last row of the committed
`outputs/loss.csv`, and the validation number is measured on held-out tokens the training batcher
never touches. That run is a 64 KiB single-epoch smoke run over 123 windows. It shows the loop runs
to completion and the loss falls, and it is not a benchmark, not a throughput figure, and not a
claim about model quality. There is no parity check against a external reference and no
inference benchmark.
