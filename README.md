# Z-TRANSFORMER

A transformer built from scratch in Zig: RMSNorm, RoPE, grouped-query attention, SwiGLU and tied
embeddings, with hand-derived gradients checked against central differences. Everything runs on the
CPU in f32. The block is checked tensor by tensor against a Llama reference, and a
GPT-mini trains end to end. One CUDA kernel exists and is benchmarked against its own CPU twin; the
rest of the plan does not, and `src/cuda/README.md` says exactly which parts those are and why.

On "Llama-shaped", precisely: the block has RMSNorm, RoPE at Llama-3's theta, grouped-query
attention, SwiGLU and no biases, and it matches the reference op by op. It is not a Llama-3 you
could scale up, and three things say so. The feed-forward width is `ffn_mult * d_model` with
`ffn_mult` an integer, so it is 4x where Llama-3 is 3.5x rounded to a multiple of 256 — the rule
cannot express Llama-3's at all. There is no RoPE scaling, so long context is out. The embeddings
are tied, where Llama-3 8B's are not. Everything is f32, not bf16, and the model has no KV cache,
so a position is always 0. At the 8B shape this does not run: the tied head is 5.6% of a step there,
and the logits tensor alone is `T * vocab` f32, which is 4.2 GB at T=8192 and `vocab = 128256`, of
which a backward pass holds two. `zig build scale-profile` prints what the formulas behind those
statements say at every shape from the shipped one to 32k context, every number labelled a
projection and none of them measured.

Requires Zig 0.16.0, enforced by `build.zig` rather than by hope. There is nothing to install and
no package manager step. Tested on macOS on Apple Silicon; the CPU path is portable, but the CUDA
row below needs a Linux host with docker and a GPU. A fresh clone has no git hook, because
`core.hooksPath` is per-clone local config:

    git config core.hooksPath .githooks

| Command | What it does | Leaves behind |
|---|---|---|
| `zig build` | Builds the binary. | `zig-out/bin/ztransformer` |
| `zig build run` | Runs it. With no argument it prints the version banner; `zig build run -- train` reaches training. | nothing |
| `zig build train` | Trains one pass over a 64 KiB prefix of the corpus. Writes `outputs/loss.pending.csv` and promotes it to `outputs/loss.csv` only if it matches the committed digest, so a run never overwrites the evidence it failed to reproduce. | `outputs/loss.csv` |
| `zig build scale-profile` | Projects the cost of shapes this model cannot be run at, from the code's own formulas. Every number is labelled a projection, and two runs are byte-identical. See `src/README.md`. | nothing |
| `zig build test` | Runs the test suite, and writes nothing at all when it passes. | nothing |
| `zig build verify` | The whole gate CI runs, silent when it passes: `zig fmt --check`; the test suite in Debug and in ReleaseFast; the version banner; a sha256 on each committed input and output, the corpus against the digest `data/README.md` documents and `outputs/loss.csv` against the digest this file documents; the scale tables quoted in `src/README.md` against `zig build scale-profile`'s own output; `tools/symbols.sh` against the symbol table in that same file; `tools/removed/report.csv` against carrying no failing row; and the training run's peak resident set size against a budget in `build.zig`, read through `/usr/bin/time` on both Darwin and Linux. | nothing |
| `zig build bench` | Reports user CPU seconds per training step, median of `-Dbench-runs` runs (default 3). It reports and never gates: a time in seconds is a property of this program and this machine, and a threshold on one fires on somebody else's load average. The figure includes fixed startup work and says so in its own output. | nothing |
| `zig build peak-rss` | The memory gate on its own, so the number can be read rather than inferred. Measures on Darwin and Linux, and **fails** on Linux rather than skipping when `/usr/bin/time` is absent, because CI runs there. Budget: 128 MiB, chosen to sit between two measured populations. Also runs as part of `verify`. | nothing |
| `sh tools/removed/check.sh` | Compares the block against a Llama reference, tensor by tensor, then checks the report it just wrote against the committed record: the platform-independent projection on every host, and the exact bytes too where the oracle's versions match the committed ones. Needs the pinned oracle in a repo-local virtualenv; see `tools/README.md`. | `tools/removed/report.csv` |
| `sh tools/removed/sensitivity.sh` | Proves those gates can fail: perturbs the exported weights on one side only and requires the check to catch it. Same venv requirement. | nothing |
| `sh src/cuda/run-norm.sh` | Runs the RMSNorm kernel against its CPU twin across eighteen shapes: the parity table and the benchmark table, including where the GPU stops winning. Needs a Linux host with docker and a GPU. | nothing |
| `sh src/cuda/run-probe.sh` | Compiles and runs one CUDA kernel on an NVIDIA GPU, in a container, and checks its integer sum against a closed form. Needs a Linux host with docker and a GPU. See `src/cuda/README.md`. | nothing |

## Status

The numerics run on CPU f32. The five block ops and the gradients are real and tested, and the block
is checked against a Llama reference. The CUDA toolchain is proven end to end on a real
GPU, and one CUDA source implements a transformer operation: RMSNorm, matching its CPU twin to 1e-5
across eighteen shapes and benchmarked against it, 9.5x faster in isolation at the shipped model
shape. `sh src/cuda/run-norm.sh` prints the ratio and names the shape it applies to. No
end-to-end speedup is claimed from it: the model has no device-resident tensor, so every other part
of a step would have to cross PCIe to use the GPU as well, and there is no fused attention kernel.
No checkpoint is written: a run leaves a loss curve and no model. The table above is the whole
interface a reader needs, and `build.zig` declares two steps beyond it that no reader needs:
`zig build dbg-train`, a Debug `train` binary for reproducing a checked-build failure, and
`zig build removed-digest`, the report gate that `sh tools/removed/check.sh` runs over the report it
has just written. It is one command doing two checks, and which of them ran is printed on every
invocation: a projection of the report that any host can check, and the exact bytes as well where
the oracle's versions are the committed ones.

`zig build train` links its own ReleaseFast binary whatever `-Doptimize` says, because one step is a
dense f32 forward and backward over a 256-token window and Debug leaves both loops unoptimised.
Its defaults are a smoke run rather than a recipe: 200 BPE merges, a 64 KiB corpus
prefix, 123 windows, one epoch, seed 7. The whole 1.1 MB corpus is 2211 windows, eighteen times the
work, so a full epoch belongs in a scheduled run. That 2211 is an offline figure: it is the full
corpus's token count over `ctx + 1`, and no command here produces it, because the binary reads no
flags and `corpus_bytes` is fixed at 65536, so `zig build train` only ever prints the arithmetic
for the prefix it runs.

No **wall** time is published for `zig build train`, and that is the decision rather than an
omission. A run of it on the host that committed the curve, on a machine already carrying unrelated
load, spent less CPU time than it did wall time, by more than a factor of two — the shape of that
gap is the reason a wall-clock figure is not a property of this program.

CPU time is a different quantity and `zig build bench` reports it, because the load argument above
applies to wall clock specifically. What `bench` is for is comparing one code change against
another, where the fixed startup work — build check, BPE training, tokenization — cancels because
it is the same on both sides. It is **not** a cost model, and it says so: it divides the whole run
including that startup by the steps the run reports, and prints the caveat in its own output. A
per-step rate quoted as what the model costs would be exactly the number the caveat is there to
disown. What the run itself publishes is its own configuration, on its first line, read back from
`src/main.zig`, and the curve it writes, which is checked. The full-corpus epoch is not timed for the
same reason: at eighteen times the windows it is a scheduled run, not a figure in a README.

Wall time is not a reproducibility guarantee, and the loss curve is. This repo states its criterion
as one seed, one build configuration, one host, and `src/README.md` documents where that boundary
sits. Two runs at seed 7 in ReleaseFast on this host produce a byte-identical `outputs/loss.csv`.

Both halves of that are commands rather than sentences. `zig build verify` checks the committed
curve against its digest, so `f1dd5444 5064810c 28002dca caf23b4b c82bb1e6 ecfa28f5 ed91e7fa 4518f792`
is what is in the tree. `zig build train` writes `outputs/loss.pending.csv` and renames it over
`outputs/loss.csv` only when those bytes match, so a run on a host with a different libm reports the
difference, leaves the committed curve alone, and still exits 0. It is a different libm, not a broken
build, and the run says so in those words.

Source lives in `src/`. Everything the repository generates lands in `outputs/`. Apache-2.0 licensed.

One measured training result is committed, and it is the whole of that kind. `zig build verify` fails
if its bytes are not the ones below, so these two numbers are checked rather than quoted. The default
`zig build train` ends at a
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
| Record | `tools/removed/report.csv`, one row per tensor per run. `check.sh` checks the report it just wrote against the digests held in `build.zig`: the byte digest `c91abe2e 924883b5 19ecfa53 7f015de1 cf4d7e03 4fcbfd92 828e21d1 39c8d30e`, and a digest over the same file with `max_abs_delta` and the six environment columns dropped. The projection is checked on every host, so the row set, the gates, the verdicts and the argmax count above are the table a run reproduces rather than one that was true once. The bytes are checked only where the oracle reports the same six version columns this file records, because a different host's BLAS rounds `max_abs_delta` its own way — `l0.v` reads `5.96e-08` on macOS and `1.043e-07` on glibc, both far inside the `1e-05` gate. `zig build removed-digest` prints which of the two it ran. |

All fourteen gates and the argmax have to hold. The two are not ranked, and the argmax is not the
sharper of the two: on this sweep the smallest reference top1-top2 margin is 8.5e-04 while the
widest gate on the logits is 2e-4, so any run that passes the gates cannot have flipped a token. The
gates are what catch a real difference, and `sh tools/removed/sensitivity.sh` is the command that
proves it: perturbing one element of `wq` by 1e-3 fails 21 of them, and scaling the whole
projection by a tenth of a percent still fails 27. The argmax earns its place by saying *why* a row failed: on any row where the
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
