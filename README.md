# Z-TRANSFORMER

A transformer built from scratch in Zig and CUDA: RMSNorm, RoPE at Llama-3's theta, grouped-query
attention, SwiGLU and tied embeddings, with hand-derived gradients checked against central
differences and fused attention kernels checked against their own CPU twins. GPT-mini trains end to
end on the CPU in f32, and greedy text generation runs from a checkpoint.

On "Llama-shaped", precisely: RMSNorm, RoPE at Llama-3's theta, grouped-query attention, SwiGLU
and no biases, with four deliberate differences: the feed-forward width is an integer multiple of
`d_model` (4x; Llama-3's 3.5x rule cannot be expressed), no RoPE scaling so long context is out,
tied embeddings, and f32 throughout instead of bf16. A KV cache and a greedy decode loop exist, on
CPU and GPU.

Requires Zig 0.16.0, enforced by `build.zig` rather than by hope. Nothing to install, no package
manager step. Tested on macOS on Apple Silicon; the CPU path is portable, but CUDA work needs a
Linux host with docker and a GPU. Test hardware: Intel i9-13900K, NVIDIA RTX 3080 Ti
12 GB, CUDA 12.6.3 (driver 615.71.09), glibc 2.43, Zig 0.16.0. A fresh clone has no git hook, because `core.hooksPath` is
per-clone local config:

    git config core.hooksPath .githooks

| Command | What it does | Leaves behind |
|---|---|---|
| `zig build` | Builds the binary. | `zig-out/bin/ztransformer` |
| `zig build run` | Runs it. With no argument it prints the version banner; `zig build run -- train` reaches training. | nothing |
| `zig build train` | Trains one pass over a 64 KiB prefix of the corpus. Writes `outputs/loss.pending.csv` and promotes it to `outputs/loss.csv` only if it matches the committed digest, so a run never overwrites the evidence it failed to reproduce. Also writes `outputs/checkpoint.bin` (gitignored) on every run. | `outputs/loss.csv` |
| `zig build infer -- <prompt>` | Greedy continuation from `outputs/checkpoint.bin` (50 tokens, `ZTRANSFORMER_N_NEW` overrides). Retrains the same 200-merge tokenizer so the ids match with no vocab file. | nothing |
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

## Results

Per attention call, the fused kernel runs **94x to 165x** the CPU twin across eleven shapes
(minimum of three runs each, no floor) at a fraction of a percent of its parity gate. Training
writes a committed loss curve falling to 5.60 train / 5.16 val. Every number is graded against a
reference this repository implements: no third-party comparison was ever measured, and a per-call
ratio is not a step speedup (attention is about 7% of a step). One full benchmark run is
committed at `outputs/bench/run-attn.log`. Full tables, method, and every
caveat live beside the code they describe: `src/cuda/README.md` for the kernels,
`outputs/README.md` for the curve.

## Status

The block is real and the CUDA path is real: five CPU ops with gradients checked against central
differences, four kernels checked against CPU twins, the attention forward graded inside a real
training step. Not done: the backward kernel has no training path, there is no inference
benchmark, and long context is out (no RoPE scaling; 8B shapes do not run). `zig build train`
leaves a checkpoint beside the curve; `zig build infer` decodes from it, greedy only.

Source lives in `src/`. Everything the repository generates lands in `outputs/`.

## License

Apache-2.0. See LICENSE.
