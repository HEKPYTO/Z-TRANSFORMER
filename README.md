# Z-TRANSFORMER

A transformer built from scratch in Zig: RMSNorm, RoPE, grouped-query attention, SwiGLU and tied
embeddings, with hand-derived gradients checked against central differences. Everything runs on the
CPU in f32. The block is checked tensor by tensor against a Llama reference, and a
GPT-mini trains end to end. Four CUDA kernels implement transformer operations, and each is timed
against its own CPU twin, though only the attention forward carries a published speedup ratio; the
rest of the plan is not measured, and `src/cuda/README.md` says exactly which parts those are and why.

On "Llama-shaped", precisely: the block has RMSNorm, RoPE at Llama-3's theta, grouped-query
attention, SwiGLU and no biases, and it matches the reference op by op. It is not a Llama-3 you
could scale up, and the differences are these. The feed-forward width is `ffn_mult * d_model` with
`ffn_mult` an integer, so it is 4x where Llama-3 is 3.5x rounded to a multiple of 256 — the rule
cannot express Llama-3's at all. There is no RoPE scaling, so long context is out. The embeddings
are tied, where Llama-3 8B's are not. Everything is f32, not bf16. A KV cache exists
(`src/kv_cache.zig`) and a greedy generation loop decodes through it (`src/decode.zig`), on the CPU
and on the GPU: `decode.cudaAttnStep` derives `q_offset = pos`, `n_keys = pos + 1` and `t = 1` from the
cache itself, so the forward kernel's offset is exercised at every real decode position rather than only at
`q_offset` 0 across the whole `T`, and `zig build cuda-attn-check` grades that path against the CPU
`attnStep` over a cache filling one position at a time. A training batch is
still correct — `rope.forward` derives row `r`'s angle from `pos + r`, so positions run `0..T-1` —
and that derivation is the one `decode.zig` reaches for at a nonzero `pos`, which is what a cache needs.
At the 8B shape this
does not run: the tied head is 5.6% of a step there,
and the logits tensor alone is `T * vocab` f32, which is 4.2 GB at T=8192 and `vocab = 128256`, of
which a backward pass holds two. `zig build scale-profile` prints what the formulas behind those
statements say at every shape from the shipped one to 32k context, every number labelled a
projection and none of them measured.

Requires Zig 0.16.0, enforced by `build.zig` rather than by hope. There is nothing to install and
no package manager step. Tested on macOS on Apple Silicon; the CPU path is portable, but the four rows
that need one require a Linux host with docker and a GPU. A fresh clone has no git hook, because
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
| `zig build peak-rss` | The memory gate on its own, so the number can be read rather than inferred. Measures on Darwin and Linux, and **fails** rather than skipping anywhere else: on Linux when `/usr/bin/time` is absent, because CI runs there, and on a platform that is neither Darwin nor Linux, because a gate that cannot run is a claim. Budget: 128 MiB, chosen to sit between two measured populations. Also runs as part of `verify`. | nothing |
| `sh tools/removed/check.sh` | Compares the block against a Llama reference, tensor by tensor, then checks the report it just wrote against the committed record: the platform-independent projection on every host, and the exact bytes too where the oracle's versions match the committed ones. Needs the pinned oracle in a repo-local virtualenv; see `tools/README.md`. | `tools/removed/report.csv` |
| `sh tools/removed/sensitivity.sh` | Proves those gates can fail: perturbs the exported weights on one side only and requires the check to catch it. Same venv requirement. | nothing |
| `sh src/cuda/run-norm.sh` | Runs the RMSNorm kernel against its CPU twin across eighteen shapes: the parity table and the benchmark table, including where the GPU stops winning. Needs a Linux host with docker and a GPU. | nothing |
| `zig build attn-bench` | Measures one CPU attention call at five context lengths and prints it beside the PCIe floor a GPU kernel would have to clear. ReleaseFast, because the number is the point. See `src/README.md`. | nothing |
| `zig build cuda-check` | Compiles the four CUDA sources with the pinned toolchain the two shell scripts measure with, as three compile units: `norm`, `probe` and `attn`, with the shipped `attn_kernels.cu` reached through the include in `attn.cu`. So a syntax or type error in the CUDA sources is caught by the build system. Deliberately outside `verify`, because a GitHub runner has no CUDA toolchain and a gate that is permanently red for a reason unrelated to the code is worse than no gate. Fails loudly rather than skipping when there is no toolchain or no GPU. | nothing |
| `sh src/cuda/run-attn.sh` | Grades the fused causal attention kernels -- forward and backward -- against the CPU implementations they replace, across eleven shapes at five head widths from 32 to 256: a parity table and a speedup for the forward, three parity gates for the backward (one each for dq, dk and dv), and a proof that all eight deliberately broken variants are caught, four per direction, with the backward's four required to produce four *distinct* signatures. Needs a Linux host with docker and a GPU. Neither half is wired into `zig build train`. See `src/cuda/README.md`. | nothing |
| `sh src/cuda/run-probe.sh` | Compiles and runs `src/cuda/probe.cu` on an NVIDIA GPU, in a container, and checks its integer sum against a closed form. Needs a Linux host with docker and a GPU. See `src/cuda/README.md`. | nothing |

## Status

**The block is real and the CUDA path is real.** The five block ops and every gradient run on CPU
f32 and are checked against a Llama reference, tensor by tensor, across all eighteen
intermediates the forward pass hands over.

Four CUDA kernels implement transformer operations, and each is graded against this repository's
own CPU twin rather than against a hand-written expectation:

| kernel | graded against | gate | measured |
|---|---|---|---|
| `rmsNormKernel` | `norm.forward` | 1e-5, 18 shapes | worst `2.861e-06`, 28.6% of gate |
| `fusedAttnForward` | `attention.forward` | 1e-4, 11 shapes | `5.960e-08` at `head_dim` 32, `7.451e-08` at 128, `1.043e-07` at 192 |
| `attnDqKernel` | `attentionBackward` | 1e-5 | worst `1.118e-08` |
| `attnDkDvKernel` | `attentionBackward` | 1e-5 | worst `5.960e-07` |

**Attention now runs on the GPU inside a real training step.** One flag in `src/model.zig`
(`cuda_attn`) routes the forward and backward through `attn_kernels.cu`, and `zig build
cuda-attn-check` grades that path against `attention.forward` on a real step's tensors. It is a
relative gate, `max|a-b| <= 1e-4 * max|b|`, because a real step's gradients are orders of magnitude
smaller than the synthetic inputs the benchmark uses; the observed ratio is `3.79e-08`, so the gate
sits about 2600x above the noise. That gate has been watched fail, by the same pattern
`run-attn.sh` uses for its four broken variants: `ATTN_CORRUPT_KEY=0.05 zig build cuda-attn-check`
perturbs one uploaded key on the device side only and must exit 1, printing
`differs by 2.366975e-4 against a reference of at most 7.8568566e-1, which is 3.0126233e-4 times
the relative gate of 1e-4`. With the variable unset, or set to something unparseable, the upload is
the plain one and the gate passes — a typo must not read as corruption.

**It is not Llama-3-shaped.** `head_dim` must be even and in `[32, 256]`; nothing requires a power
of two, and eleven benchmark shapes cover `head_dim` 32, 96, 128 (Llama-3), 192 and 256. 96, 192
and 256 are also the head widths Phi-3, DeepSeek-V2/V3 and Gemma-2 ship, and what runs at them is
this repository's plain attention -- grouped-query, or MHA where the group is 1 -- with none of the
rest of those models: no sliding window, no LongRoPE, no GeGLU, no logit softcapping, no latent
attention. At 256 the shared-memory request does not fit, so the kernel narrows its own tile and
runs; before that it killed the process outright on a width its own predicate accepted.

### What is not claimed

**The measured ratio: three invocations of `sh src/cuda/run-attn.sh` on the CUDA host, GPU idle, measured `2026-10-03`.** Minimum first, because that is this repository's rule for a shared host.

| shape | min of 3 | max of 3 | spread |
|---|---|---|---|
| `ctx256` | 57.9x | 105.6x | **1.82x** |
| `ctx512` | 133.6x | 137.5x | 1.03x |
| `ctx1024` | 162.0x | 164.4x | 1.01x |
| `ctx2048` | 162.2x | 166.2x | 1.02x |
| `ctx4096` | 168.1x | 168.8x | 1.00x |
| `llama3-T256` | 106.2x | 107.2x | 1.01x |
| `llama3-T512` | 127.3x | 131.0x | 1.03x |
| `gqa-96-T256` | 100.8x | 103.6x | 1.03x |
| `gqa-96-T512` | 109.8x | 110.2x | 1.00x |
| `mha-192-T256` | 104.4x | 105.3x | 1.01x |
| `gqa-256-T256` | 99.6x | 100.8x | 1.01x |

**That is the whole claim: 57x to 168x per call, and no floor.** Exactly one shape swings across the
three runs: `ctx256`, at 1.82x. The other ten hold between 1.00x and 1.03x.

That is not a rounding difference and it is not noise, and the committed sweep says why.
`outputs/bench/ctx256-sweep.csv` is ten runs of the CPU twin at that one shape: three land at
6070.021 to 6088.290 us and seven at 10583.812 to 10875.440 us, each cluster internally tight, with
**nothing at all between them** and a **1.7384x** gap. **70% of single runs land in the higher mode**,
so one sample here is a 1.74x coin toss. The minimum is the only safe statistic at this shape, and
even the minimum of three catches the low mode only 65.7% of the time by that distribution. What
flips it -- core placement, thread affinity, a competing process -- is not identified. **The
bimodality is this shape's, not the benchmark's**, which is the one thing the ten-run sweep settles
that three invocations could not.

**The backward publishes no speedup ratio at all.** `attn_twin.zig` times the CPU backward and
writes it to the manifest, but the benchmark prints only the kernel's own `bwd_us`, and the CPU
backward at ctx4096 has read 1.69e6 us and 3.10e6 us in different sessions. A ratio built on that is
not a figure.

**None of this is a step speedup.** Attention is 6.3% to 7.3% of a training step at this model's
shape, so moving it to the GPU is worth 1.065x to 1.077x on a step. The other 92.7% is matmul,
norms, RoPE, SwiGLU, the tied head and AdamW, all still on the CPU. **The model has no
device-resident tensor**, so every other part of a step would still cross PCIe to use the GPU.

**The CUDA gates and the block-parity gates share no number.** `ATTN_TOL` and `ATTN_BWD_TOL` grade a
CUDA kernel against this repository's own CPU twin. The eighteen per-tensor gates in
`tools/removed/oracle.txt` grade this block against external. A row passing the first says nothing
about the second, and the Llama-3 block-parity claim rests entirely on the `oracle.txt` gates.

No checkpoint is written: a run leaves a loss curve and no model. The table above is the whole
interface a reader needs, and `build.zig` declares five steps beyond it, none of which a reader needs to run:
`zig build dbg-train`, a Debug `train` binary for reproducing a checked-build failure;
`zig build removed-digest`, the report gate that `sh tools/removed/check.sh` runs over the report it
has just written; `zig build table-block-check`, which breaks the gate that holds the scale tables in
`src/README.md` four ways on purpose and fails unless all four are caught, and prints what each one
produced; and the two CUDA steps `zig build cuda-attn-check` and `zig build cuda-train`,
which are the ones that link `attn_kernels.o`. Both of the CUDA pair **exit 1 while `cuda_attn` is
false**, naming the line to edit, rather than skipping: a green that checked nothing is worse than
a refusal. `removed-digest` is one command doing two checks, and which of them ran is printed on every
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
curve against its digest, so `7d7bcbd8 f7a2e67a d9b29c34 4d60a3d9 5e37bc14 e85ba379 f52adb4f 72cb6160`
is what is in the tree. `zig build train` writes `outputs/loss.pending.csv` and renames it over
`outputs/loss.csv` only when those bytes match, so a run on a host with a different libm reports the
difference, leaves the committed curve alone, and **exits 1** -- the env var named in the next sentence is what turns that into an exit 0 without promoting anything. An earlier version of this line said it still exited 0, which is what the env var does and not what the default does, and it is the line a reader on a second machine follows. It is a different libm, not a broken
build, and the run says so in those words.

Worth being plain about what `verify` does *not* do, because it is the obvious thing to assume. It
hashes the committed file; it does not compare that file against a curve freshly produced at the
shipped shape. `zig build determinism` narrows that gap without closing it — it runs two short passes
and requires them to agree with each other, so it catches a build whose output is no longer a
function of its input. It cannot catch a change that is deterministic and merely different: editing
`optim.AdamW.beta1` leaves every run agreeing perfectly on a curve that is no longer the committed
one. The gate for that remains `zig build train`, and reproducibility stays a host-relative pair —
baseline against change on the same machine — rather than a digest any host can be checked against.
Its peak-memory check does run a full 123-step pass, but with the accepting env var set, so `verify`
stays green on a host where `zig build train` refuses.

Source lives in `src/`. Everything the repository generates lands in `outputs/`. Apache-2.0 licensed.

One measured training result is committed, and it is the whole of that kind. Three different checks
cover it and they are not the same check, which an earlier version of this paragraph ran together.
`zig build verify` fails if the committed `outputs/loss.csv` is not byte-for-byte the file these two
numbers were read from, so the curve cannot be edited out from under the paragraph. Beside it,
`zig build determinism` runs the training twice at 30 steps and requires byte-identical curves, which
catches an output that is no longer a function of its input — an uninitialised read, an iteration
order, a race. Neither of those **not** checks that the code still produces that file: both compare
against something stable while the curve itself is never freshly derived at the shipped shape, so
changing `optim.AdamW.beta1` leaves both green. The gate for *that* is `zig build train`, which re-runs
the training and refuses to promote a curve that differs — and it is a manual, single-host command,
because a different libm legitimately produces different bytes and making it a CI gate would leave
the job permanently red for a reason unrelated to the code. The default
`zig build train` ends at a
train loss of 5.596625 and a validation loss of 5.155396; both are the last row of the committed
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
| Sweep | sequence lengths 1, 8 and 257, at seeds 7 and 8. Six runs, 204 tensor rows, 532 argmax comparisons |
| Result | 18 tensor kinds, every row inside its own gate. The worst case is 2.056e-06 on `l0.k_rope`, against that kind's 2.0e-05 gate. The tightest gate in the set is 2.0e-06, on `attn_norm_out` — a different tensor, so the two numbers are not to be compared |
| Argmax | 532 of 532 rows pick the same token. Smallest reference top1-top2 margin 8.5e-04 |
| Record | `tools/removed/report.csv`, one row per tensor per run. `check.sh` checks the report it just wrote against the digests held in `build.zig`: the byte digest `d0d501f0 dcc5170b 7b3bb8f3 24ed14ba dae9a228 2b7f1424 5e80bdd8 525586b1`, and a digest over the same file with `max_abs_delta` and the six environment columns dropped. The projection is checked on every host, so the row set, the gates, the verdicts and the argmax count above are the table a run reproduces rather than one that was true once. The bytes are checked only where the oracle reports the same six version columns this file records, and in
practice that is macOS alone: the Linux torch wheel reports itself as `2.14.0+cu130` where this file
records `2.14.0`, so on Fedora and on `ubuntu-latest` alike the guard fires and the bytes are skipped.
Measured on Fedora: the projection matched exactly — 18 kinds, 206 rows, every gate and verdict
identical — and the bytes were skipped for the version string, not for a float. Nothing here therefore
shows that two hosts round `max_abs_delta` the same way, and no cross-host pair of deltas has been kept.
The projection is the check that runs wherever a comparison actually runs. `zig build removed-digest`
prints which of the two it ran. |

All eighteen gates and the argmax have to hold. The two are not ranked, and the argmax is not the
sharper of the two: on this sweep the smallest reference top1-top2 margin is 8.5e-04 while the
widest gate on the logits is 2e-4, so any run that passes the gates cannot have flipped a token. The
gates are what catch a real difference, and `sh tools/removed/sensitivity.sh` is the command that
proves it: perturbing one element of `wq` by 1e-3 fails 20 of the 204 compared rows, and scaling the
whole projection by a tenth of a percent still fails 18. The argmax earns its place by saying *why* a row failed: on any row where the
two disagree, the reference's own top1-top2 margin is printed, which turns "it failed" into "it
failed on a near-tie" or "it failed outright".

What it claims: that this forward pass and the reference's agree, on the sweep above, at a stated
set of pinned weights. What it does not claim: anything about initialisation, because the harness
pins the weights on both sides and never compares how they were drawn; anything about batched
forward, because there is no batch axis in this model and the harness runs batch 1; anything about a
KV cache, quantized weights, or CUDA; and the four tensors `forward` reduces internally
and never hands out. `forwardWith` does hand them out, through the sink, and that is the
path the harness uses, so all four are gated. It also does not
claim sensitivity: this shape is small, and an error of about a tenth of a percent in a projection
passes. `tools/README.md` says what was measured. The oracle is
Python and lives outside `src/`; the export it reads is written by Zig alone, so no committed number
depends on it.
