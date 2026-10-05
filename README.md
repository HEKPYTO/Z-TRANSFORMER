# Z-TRANSFORMER

A transformer built from scratch in Zig: RMSNorm, RoPE, grouped-query attention, SwiGLU and tied
embeddings, with hand-derived gradients checked against central differences. Everything runs on the
CPU in f32. The block's gradients are checked against central differences and the CUDA kernels
against their own CPU twins, and a
GPT-mini trains end to end. Four CUDA kernels implement transformer operations, and each is timed
against its own CPU twin, though only the attention forward carries a published speedup ratio; the
rest of the plan is not measured, and `src/cuda/README.md` says exactly which parts those are and why.

On "Llama-shaped", precisely: the block has RMSNorm, RoPE at Llama-3's theta, grouped-query
attention, SwiGLU and no biases. It is not a Llama-3 you
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
| `zig build verify` | The whole gate CI runs, silent when it passes: `zig fmt --check`; the test suite in Debug and in ReleaseFast; the version banner; a sha256 on the committed corpus and on the committed loss curve, read with `sha256sum` or `shasum -a 256` on either platform; the attention benchmark's CPU half compiles; the scale tables in `src/README.md`; the symbol table in `src/README.md`; and the training run's peak resident memory, which is the one sub-check that needs `/usr/bin/time`. | nothing |
| `zig build bench` | Reports user CPU seconds per training step, median of `-Dbench-runs` runs (default 3). It reports and never gates: a time in seconds is a property of this program and this machine, and a threshold on one fires on somebody else's load average. The figure includes fixed startup work and says so in its own output. | nothing |
| `zig build peak-rss` | The memory gate on its own, so the number can be read rather than inferred. Measures on Darwin and Linux, and **fails** rather than skipping anywhere else: on Linux when `/usr/bin/time` is absent, because CI runs there, and on a platform that is neither Darwin nor Linux, because a gate that cannot run is a claim. Budget: 128 MiB, chosen to sit between two measured populations. Also runs as part of `verify`. | nothing |
| `sh src/cuda/run-norm.sh` | Runs the RMSNorm kernel against its CPU twin across eighteen shapes: the parity table and the benchmark table, including where the GPU stops winning. Needs a Linux host with docker and a GPU. | nothing |
| `zig build attn-bench` | Measures one CPU attention call at five context lengths and prints it beside the PCIe floor a GPU kernel would have to clear. ReleaseFast, because the number is the point. **Its `ratio` column is a diagnostic and not a speedup**: it divides by a bandwidth constant the tool itself calls "a lower bound and not a prediction", so it reads roughly ten times the real kernel comparison and every row's verdict says the kernel's own cost decides. The published ratio is in `src/cuda/README.md`, measured on a GPU. See `src/README.md`. | nothing |
| `zig build cuda-check` | Compiles the four CUDA sources with the pinned toolchain the two shell scripts measure with, as three compile units: `norm`, `probe` and `attn`, with the shipped `attn_kernels.cu` reached through the include in `attn.cu`. So a syntax or type error in the CUDA sources is caught by the build system. Deliberately outside `verify`, because a GitHub runner has no CUDA toolchain and a gate that is permanently red for a reason unrelated to the code is worse than no gate. Fails loudly rather than skipping when there is no toolchain or no GPU. | nothing |
| `sh src/cuda/run-attn.sh` | Grades the fused causal attention kernels -- forward and backward -- against the CPU implementations they replace, across eleven shapes at five head widths from 32 to 256: a parity table and a speedup for the forward, three parity gates for the backward (one each for dq, dk and dv), and a proof that all eight deliberately broken variants are caught, four per direction, with the backward's four required to produce four *distinct* signatures. Needs a Linux host with docker and a GPU. Neither kernel is on the DEFAULT training path -- `src/model.zig`'s `cuda_attn` is false as shipped. Switched on, it routes a step's FORWARD through the kernel and `zig build cuda-attn-check` grades it; the backward has no training path at all, nothing in the Zig tree calling `device.Attn.backward`, and is graded only here. See `src/cuda/README.md`. | nothing |
| `sh src/cuda/run-probe.sh` | Compiles and runs `src/cuda/probe.cu` on an NVIDIA GPU, in a container, and checks its integer sum against a closed form. Needs a Linux host with docker and a GPU. See `src/cuda/README.md`. | nothing |

## Results, and what they are a result *of*

**The specific task is long-context causal attention behind a KV cache.** That is the one thing
this project is built to beat the obvious implementation at, and it is where every number below
comes from. The obvious implementation materialises the `[T x T]` score matrix; this one never
writes it, and every row below is the CUDA kernel against this repository's own CPU twin —
measured, gated, and named with the command that re-derives it.

**Why the kernel can be fast is a shape claim, and it is checkable from the source.** The
running softmax maximum and the unnormalised accumulator are per-thread **registers**
(`attn_kernels.cu:253`), the only shared memory is one K/V tile per block (`:213`), and the
kernel allocates nothing itself — Q, K, V and O are the caller's buffers, each sized
`T * n_heads * dim`. So the footprint is `O(T * n_heads * dim)`. A dense attention pass has to
hold the score matrix to finish its softmax, which is `O(T^2 * n_heads)`. The ratio is

```
dense / fused  =  (T^2 * n_heads) / (T * n_heads * dim)  =  T / dim
```

At `T = 4096`, `head_dim = 128`, `n_heads = 8` that is 134,217,728 f32 against 4,194,304 —
**32x**, and the `n_heads` cancels. It is 32x at 4096 tokens, 128x at 16384, 512x at 65536: the factor is the context length
over the head width, so **lengthening the context widens the gap**. The fused form is not
constant — it grows linearly in `T` as the four buffers must — it is the `T^2` term that is
avoided, and `T^2` is what outruns a linear budget. That is arithmetic on the shapes, not a
measurement, and it is stated as such.

**Speed — same work, same answer** (`sh src/cuda/run-attn.sh`, minimum of three per shape):

```
llama3-T256  341188 us  ->  3161 us    108x   |###########################################               |
llama3-T512 1667537 us  -> 11801 us    141x   |##########################################################    |
```

`ctx256` is deliberately absent. Its CPU column is bimodal on this host -- three runs of the
sweep drew the high mode and none the low, a 1.74x coin toss -- so a ratio at that shape is a
reading of which mode the host happened to be in, not a measurement of the kernel. The kernel
column stays steady across the same runs (1.06x). `src/cuda/README.md` carries the full sweep
including that shape, with the spread stated.

**Accuracy — the kernel is not trading correctness for either** (same run, `ATTN_TOL` 1e-4):

| shape | max_abs | fraction of gate used |
|---|---|---|
| `llama3-T256` | `7.451e-08` | **0.075%** |
| `llama3-T512` | `7.451e-08` | **0.075%** |

The backward's three gradients (`ATTN_BWD_TOL` 1e-5) sit at 2.4% to 7.2% of their gate at the
same shapes.

**What this claim is not.** Every number compares this CUDA path against **this repository's own
CPU implementation** — `attention.forward`, `attentionBackward`, `src/norm.zig`. Nothing here is
a comparison against another project, and no such measurement was ever taken. The ratio is a
statement about a fused kernel against a naive one on the same machine, which is the claim this
project can actually support. Both directions are gated: `zig build cuda-attn-check` proves the
GPU forward runs inside a real training step, and its `ATTN_CORRUPT_KEY=0.05` control exits 1,
so a green row means the gate would notice a fault.

**Training end to end, and the curve is a committed artifact** (`outputs/loss.csv`, sha256
`7d7bcbd8`, reproduced byte for byte by `zig build train`):

```
6.5 |*                                                                  6.489 step 24
6.0 |    *                                                            6.062 step 49
5.8 |         *                                                      5.814 step 74
5.7 |              *                                               5.687 step 99
5.6 |                   *                                          5.597 step 122
    +--------------------------------------------------------------------
      0        25        50        75       100       125   step
```

## Status

**The block is real and the CUDA path is real.** The five block ops and every gradient run on CPU
f32. Every gradient is checked against central differences, and every CUDA kernel against its own
CPU twin — so the checks are internal, and this repository makes no claim about agreement with
another implementation of the same block.

Four CUDA kernels implement transformer operations, and each is graded against this repository's
own CPU twin rather than against a hand-written expectation:

| kernel | graded against | gate | measured |
|---|---|---|---|
| `rmsNormKernel` | `norm.forward` | 1e-5, 18 shapes | worst `2.861e-06`, 28.6% of gate |
| `fusedAttnForward` | `attention.forward` | 1e-4, 11 shapes | `5.960e-08` at `head_dim` 32 and 96, `7.451e-08` at 128 and 256, `1.043e-07` at 192 |
| `attnDqKernel` | `attentionBackward` | 1e-5 | worst `2.980e-08`, 0.298% of gate, at `head_dim` 128 (Llama-3) |
| `attnDkDvKernel` | `attentionBackward` | 1e-5 | worst `8.345e-07` at dv, 8.34% of gate, also at `head_dim` 128 (Llama-3, T 512) |

**The attention forward now runs on the GPU inside a real training step.** One flag in `src/model.zig`
(`cuda_attn`) routes the forward through `attn_kernels.cu`, and `zig build cuda-attn-check` grades that
path against `attention.forward` on a real step's tensors. **The backward kernel is not on the training path** — nothing in the Zig tree calls `device.Attn.backward`, so it is graded by the benchmark harness and nowhere else. It is a
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

**The measured ratio: three invocations of `sh src/cuda/run-attn.sh` on the Linux host of record, GPU idle, measured `2026-10-04`, on a host `sh tools/host-clean.sh` accepted with the GPU idle and load 0.16.** Minimum first, because that is this repository's rule for a shared host.

| shape | min of 3 |
|---|---|
| `ctx256` | 98.3x |
| `ctx512` | 134.7x |
| `ctx1024` | 128.2x |
| `ctx2048` | 140.3x |
| `ctx4096` | 165.4x |
| `llama3-T256` | 101.9x |
| `llama3-T512` | 118.5x |
| `gqa-96-T256` | 95.1x |
| `gqa-96-T512` | 102.8x |
| `mha-192-T256` | 97.9x |
| `gqa-256-T256` | 93.8x |

**That is the whole claim: 94x to 165x per call, and no floor.** The `max of 3` and `spread` columns
the previous table carried are not reproduced here: this run retained the minimum per shape and the
logs it was computed from were then removed, so a spread column would be reconstructed rather than
measured, and this file does not publish a number it did not take.

One reading of the sweep did not repeat. All three `ctx256` invocations drew the **high** CPU mode
(102.4x, 105.8x, 98.3x), which the committed ten-run sweep puts at **34.3%** -- `1 - 0.7^3` -- so this
table's `ctx256` is the minority outcome rather than the expected one, and a reader should expect the
number to be nearer the high cluster than to 98.3x.

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
shape, so moving it to the GPU is worth 1.016x to 1.027x on a step -- the forward-only swap the code can make today. The
other 92.7% to 93.7% is matmul, norms, RoPE, SwiGLU, the tied head and AdamW, all still on the CPU. **The model has no
device-resident tensor**, so every other part of a step would still cross PCIe to use the GPU.

**The CUDA gates grade CUDA against this repository's own CPU twin, and nothing else.**
`ATTN_TOL` and `ATTN_BWD_TOL` compare `src/cuda/attn.cu` against `attention.forward` and
`attentionBackward`. Two tolerance systems, no number in common with each other, and no
third-party reference anywhere in the loop: a green CUDA gate says the kernel and the Zig twin
agree, and that is the whole of it.

No checkpoint is written: a run leaves a loss curve and no model. The table above is the whole
interface a reader needs.

Two further steps exist because `verify` depends on them, so a reader never runs them by hand:
`zig build determinism`, which re-derives the loss curve host-relatively, and `zig build
attn-twin`, which compiles the benchmark's CPU half so that a rename there cannot stay green.
`attn-twin` is in the table above precisely because `verify` runs it.

The rest are ones a reader invokes deliberately. `zig build dbg-train`, a Debug `train` binary for
reproducing a checked-build failure; `zig build host-check`, which refuses a host whose GPU, VRAM,
load or process table would contaminate a timing; `zig build step-profile`, which times each op
kind in a step and requires the denominator to be elapsed time rather than the sum of its own
buckets, and the shares to move when the work does; `zig build table-block-check`, the negative
control for the gate that holds the scale tables in `src/README.md` -- **six** assertions, of which
one case is expected to PASS and three to fail, alongside a floor of zero and a check that a
missing-marker failure names the missing marker -- and it prints what each one produced; `zig build
cuda-train`, both of which link `attn_kernels.cu`. Those two **exit 1 while `cuda_attn` is false**,
naming the line to edit, rather than skipping: a green that checked nothing is worse than a refusal.

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
is the reproducibility claim itself: two runs of one seed on this host produce the same bytes. Its
negative control varied the corpus size, so it is shown to detect two runs that *differ*; it is not
shown to catch any particular cause of non-determinism, and none is named here. Neither check confirms
that the code still produces the committed file -- both compare against something stable while the
curve is never freshly derived at the shipped shape, so changing `optim.AdamW.beta1` leaves both green. The gate for *that* is `zig build train`, which re-runs
the training and refuses to promote a curve that differs — and it is a manual, single-host command,
because a different libm legitimately produces different bytes and making it a CI gate would leave
the job permanently red for a reason unrelated to the code. The default
`zig build train` ends at a
train loss of 5.596625 and a validation loss of 5.155396; both are the last row of the committed
`outputs/loss.csv`, and the validation number is measured on held-out tokens the training batcher
never touches. That run is a 64 KiB single-epoch smoke run over 123 windows. It shows the loop runs
to completion and the loss falls, and it is not a benchmark, not a throughput figure, and not a
claim about model quality. There is no inference benchmark.
