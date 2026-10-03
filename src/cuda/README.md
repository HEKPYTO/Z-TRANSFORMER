# src/cuda

Eleven files. Four kernels -- one primitive, one attention forward, two attention backward -- across two
CPU twins and their runners, plus the shared container recipe, the older toolchain probe,
`attn_kernels.cu` (the kernels and their C entry points, separate from the benchmark that grades them)
and `device.zig` (the Zig side of that FFI).

| File | What it is |
|---|---|
| `norm.cu` | The RMSNorm kernel: the parity gate and the benchmark. All of it, no CPU reference. |
| `norm_twin.zig` | Generates inputs, runs `norm.forward`, writes the reference output and its own timings. |
| `run-norm.sh` | Compiles and runs both halves and exits non-zero on a failed check. |
| `attn_kernels.cu` | The three attention kernels -- `fusedAttnForward`, `attnDqKernel`, `attnDkDvKernel` -- and the six `extern "C"` entry points that call them. **This is the library object**: a caller links this file, not `attn.cu`. |
| `attn.cu` | The benchmark harness. It `#include`s `attn_kernels.cu` and holds `main`, so it cannot be linked as a library. Prints both tables, the four parity gates -- one forward at `ATTN_TOL`, three backward at `ATTN_BWD_TOL`, one each for dq, dk and dv -- drives the eight broken variants, four per direction, over two separate env vars, and then calls the library's own six entry points once per shape so the surface a training step would use is executed on every run rather than only compiled. |
| `attn_twin.zig` | Generates inputs at five head widths, runs `attention.forward`, writes the reference and its own timings. |
| `device.zig` | The Zig side of that FFI: six `extern fn` declarations, a checked size computation and an 11-buffer device holder. `src/model.zig` reaches it behind `pub const cuda_attn`: `cudaForward` calls `Attn.forward` when that flag is true; nothing calls `Attn.backward`, so the backward kernel has no training path. |
| `run-attn.sh` | Compiles and runs both halves, proves all eight broken variants are caught -- four per direction, and the backward's four must additionally produce four *distinct* signatures -- and exits non-zero if any check fails. |
| `run-probe.sh` | Compiles and runs the probe, the same container recipe, no parity. |
| `cuda.sh` | The container, image pin and nvcc flags, shared by all three runners. |
| `probe.cu` | Toolchain probe. It is the only file here that is not a transformer operation. |

## Run it

| Command | What it does |
|---|---|
| `sh src/cuda/run-norm.sh` | Pulls the image if absent, builds the twin, generates 18 shapes, checks parity, proves the gate can fail, benchmarks, removes everything. |
| `sh src/cuda/run-attn.sh` | Same three stages for the attention kernels: builds the twin, generates 11 shapes at five head widths with a forward reference and a backward reference each, checks all four parity gates, proves all eight broken variants are caught, calls the six C entry points once per shape, benchmarks, removes everything. |
| `ATTN_MAX_TILE=1 sh src/cuda/run-attn.sh` | The same, with the tile cap at its floor of 1 -- one key per tile iteration. Every row still passes its gate, which is what makes "a tile of 1 is correct" a measurement rather than an argument. |
| `sh src/cuda/run-probe.sh` | The toolchain probe. Still passes after `cuda.sh` was extracted. |
| `sh src/cuda/run-probe.sh --emit-object` | Compiles `probe.cu` to `.zig-cache/cuda/probe.o` and stops. |

Measured on an RTX 3080 Ti, CUDA 12.6.3, driver 615.71.09, against a Fedora host, on
`2026-09-29`. Both halves ran on the same machine, so neither side is measured against a different
processor.

```
$ sh src/cuda/run-norm.sh
norm: scratch <project>/.zig-cache/cuda/norm
norm: built the CPU twin
twin: one            1x1     gauss   iters 0     cpu_us     -1.000 alloc_us   -1.000
...
twin: ship         256x128   gauss   iters 4000  cpu_us    162.879 alloc_us   83.929
twin: big         4096x4096  gauss   iters 12    cpu_us  91628.051 alloc_us 48063.965
twin: peak RSS 192.6 MiB over 18 shapes
twin: OK
norm: wrote inputs and the reference output
norm: compiled norm.cu
device: NVIDIA GeForce RTX 3080 Ti, compute capability 8.6, 256 threads per block, 8 warps per row
parity  shape        size         kind       max_abs_diff  gate
...
parity: worst 2.861023e-06 over big (16777216 elements), gate 1.0e-05
...
norm: OK
```

Exit 0. A non-zero exit names the shape, the failing call and its line number.

## The kernel

One block per row, 256 threads, two passes over the row. Pass one accumulates the sum of squares and
reduces it across the block; pass two divides by the row's `rms` and applies the weight. `eps`,
the division order, the multiply, and the narrowing to `f32` before `rms` is used are all copied
from `src/norm.zig`.

### The reduction is a tree, and that is a correctness decision

`src/norm.zig` accumulates `sum_sq` in `f64`, serially. A GPU cannot: `f64` runs at a sixtieth of
`f32` rate on a GeForce part, and accumulating per element in `f64` would cost that on every one of
the row's elements.

So the GPU sums in `f32`, and that is only acceptable because the reduction is a **tree** rather
than a serial chain. A serial `f32` sum over `n` terms has error growing like `n`; a tree has error
growing like `log2(n)`. At 4096 columns a thread holds 16 partials and the block folds them through
8 warps, so the deepest chain of additions is about 24, not 4096.

This is not a hypothetical margin. `src/norm_test.zig` measures the serial case at 512 elements and
finds the `f32` result lands `1.667e-6` from the `f64` answer, past the `1e-6` that file holds the
CPU implementation to. That is the number a serial `f32` reduction would have carried into the
parity column. The measured column below peaks at `2.9e-6` at 4096 columns, and most of that is
ordinary `f32` storage rounding rather than accumulation.

### One `f64` per row, on purpose

After the tree reduction the total is widened to `f64` once per **row**, and the divide, the `eps`
add, the square root and the narrowing back to `f32` all happen in `f64`, exactly as the
`@sqrt(sum_sq / d + eps)` in `norm.forward` does. It is free: 4096 rows ask for 4096 double
divisions and square roots
against 16.7 million `f32` loads, and the kernel is bandwidth bound long before that.

Doing it this way means the only thing left that can differ between the two implementations is the
sum itself. Every per-element operation after it is `f32` on both sides and agrees to the bit.

### What is deliberately not optimised

`x / rms` is a real division, not a multiply by a reciprocal. A reciprocal would add a second
rounding for a few percent of a bandwidth-bound kernel, and the division is bit-exact against the
host's, so it was not worth the difference.

The row is read twice instead of being staged in shared memory. A shared-memory copy needs a
dynamic shared size that caps the row width, and the second read is an L2 hit. This is the first
thing to revisit if the kernel has to accept rows wider than a few thousand.

## Parity

The reference is `src/norm.zig` itself. `norm.cu` contains **no CPU implementation of RMSNorm**,
because AGENTS.md is explicit that a second implementation we wrote is not a reference and that
agreeing with it proves less than it appears to. The two sides meet on flat little-endian `f32`
blobs, and the shape table lives in `norm_twin.zig` and reaches the `.cu` as `manifest.tsv`, so
neither side can disagree about what was run.

Gate: worst absolute difference over every element of every shape must be at most `1e-5`.

```
parity  shape        size         kind       max_abs_diff  gate
parity  one          1x1          gauss      0.000000e+00  pass
parity  hand         1x4          gauss      0.000000e+00  pass
parity  ragged       3x7          gauss      2.384186e-07  pass
parity  model        256x128      gauss      9.536743e-07  pass
parity  model_flat   256x128      constant   0.000000e+00  pass
parity  ffn          64x512       gauss      9.536743e-07  pass
parity  ffn_flat     64x512       constant   0.000000e+00  pass
parity  wide         64x4096      gauss      1.430511e-06  pass
parity  wide_flat    64x4096      constant   7.152557e-07  pass
parity  micro        32x32        gauss      4.768372e-07  pass
parity  small        32x128       gauss      2.384186e-07  pass
parity  mid          128x128      gauss      4.768372e-07  pass
parity  ship         256x128      gauss      9.536743e-07  pass
parity  tall         4096x128     gauss      1.430511e-06  pass
parity  r1024c512    1024x512     gauss      1.907349e-06  pass
parity  r256c2048    256x2048     gauss      1.430511e-06  pass
parity  r1024c2048   1024x2048    gauss      1.907349e-06  pass
parity  big          4096x4096    gauss      2.861023e-06  pass
parity: worst 2.861023e-06 over big (16777216 elements), gate 1.0e-05
```

Worst is `2.86e-6` against a `1e-5` gate, a margin of 3.5x, and it is at the widest row measured.
The `constant` rows are the ones that matter for the accumulator argument: every element is the
same value, so the partial sums are all positive and identical rather than cancelling, which is the
worst case for a sum of squares. `wide_flat` at 4096 columns comes in at `7.2e-7`, and the shape it
would fail on is the serial reduction this kernel does not use.

Four rows come out bit-exact, which is not the print rounding. On a `constant` row every partial
sum is the same `f32` value, so at `cols = 128` and `cols = 512` the whole tree folds by powers of
two and the `f32` sum is exact, matching the `f64` sum to the bit; `rms` is then identical and so is
every element after it. At `cols = 4096` each thread holds 16 values and the last fold sums eight
partials of 512 values serially, which is no longer a power of two and does round — which is why
`wide_flat` reads `7.2e-7` instead of zero. The `f32` rounding of the square itself is common to
both sides at every width and is not what separates these rows.

## The gate is checked against two broken kernels

A gate that has never rejected anything is not known to work. Every run also launches two
deliberately broken variants of the same kernel over the same inputs and requires the gate to
reject both. A run where either slips past exits non-zero, which means the parity column above would
mean nothing.

| Fault | What it is | Worst difference seen | Rejected |
|---|---|---|---|
| `weight[0] += 1e-3` | One weight element out of 16.7 million moved by 1e-3 | `5.7e-4` to `3.9e-3` | every shape |
| `mean, not mean of squares` | The multiply by itself is dropped from the accumulator | `1.1` to `4.0e+03`, and `inf` | every shape |

The first is the important one: a `1e-3` error is caught by a `1e-5` gate with two orders of
magnitude to spare, which says the gate discriminates at its own scale and is not merely detecting
garbage.

The second caught a real bug in the gate itself, which is the reason it is worth running. On a row
whose mean is negative, that kernel takes `sqrt` of a negative and returns `NaN`, and the original
comparison was `if (d > worst)`, which is **false for every comparison against a `NaN`**. A kernel
returning `NaN` on every element therefore reported a worst difference of exactly `0.000000e+00`
and passed the gate on three shapes. The comparison now maps a `NaN` to infinity before comparing.
That is what the `inf` rows below are.

```
gate    shape        size         deliberate fault          max_abs_diff  caught?
gate    one          1x1          weight[0] += 1e-3         9.999871e-04  caught
gate    one          1x1          mean, not mean of squares           inf  caught
gate    hand         1x4          weight[0] += 1e-3         5.692840e-04  caught
gate    hand         1x4          mean, not mean of squares           inf  caught
gate    ragged       3x7          weight[0] += 1e-3         1.513243e-03  caught
gate    ragged       3x7          mean, not mean of squares           inf  caught
...
gate    big          4096x4096    weight[0] += 1e-3         3.232718e-03  caught
gate    big          4096x4096    mean, not mean of squares           inf  caught
```

## Benchmark

Both sides run on the same host in the same run. The CPU number is `src/norm.zig` itself, timed by
`norm_twin.zig` in `ReleaseFast`, because a `Debug` build would measure the build mode rather than
the operation. `cpu_core` is that number minus the output allocation, which is why the `win` column
uses it: the GPU path reuses buffers that already exist, and charging the CPU for an allocation the
GPU does not pay is not a comparison.

| Column | What it times |
|---|---|
| `cpu` | `norm.forward` as written, its output allocation included |
| `cpu_core` | that minus the allocation: what the arithmetic costs |
| `gpu` | this kernel, launches back to back, one synchronize at the end |
| `gpu_sync` | this kernel, launch and synchronize on every call |
| `gpu_e2e` | host to device, kernel, device to host, on every call |

Every number in this section is the output of `sh src/cuda/run-norm.sh`, on the RTX 3080 Ti and
host named above, quoted as it printed. Microseconds per call:

```
shape        size          elements        cpu   cpu_core        gpu   gpu_sync    gpu_e2e      win
micro        32x32             1024       3.43       2.37       3.68       7.51      16.49     0.6x
small        32x128            4096      14.13       9.98       3.16       6.91      18.95     3.2x
mid          128x128          16384      83.52      39.83       4.53       8.33      32.95     8.8x
ship         256x128          32768     162.03      78.64       7.31      11.06      50.45    10.8x
tall         4096x128        524288    2587.72    1295.46      76.56      80.35     541.32    16.9x
r1024c512    1024x512        524288    2632.53    1340.94      21.49      25.31     498.58    62.4x
r256c2048    256x2048        524288    2633.22    1341.79       8.60      12.43     478.94   156.0x
r1024c2048   1024x2048     2097152    10817.04    5428.86      28.79      32.56     1769.64   188.6x
big          4096x4096     16777216   89190.87   43027.31     232.11     235.96   19672.66   185.4x
```

This is the second of three consecutive runs on an idle GPU, quoted as it printed so every ratio
below is the script's own arithmetic on the row above it. Across the three runs the `gpu` column
moved by at most **0.87%** — four of the nine shapes repeated to the printed digit — and the `cpu_core`
column by under 1%. The `gpu_e2e` column is the noisy one, up to **8.7%** at the largest shape, because
it is two PCIe transfers and a synchronise wrapped around a kernel that takes 232 us. So the kernel
timings are the reproducible part of this table and the end-to-end column is a range, not a value.

The table this one replaces was measured on driver **13040**; this one is driver **615.71.09** on the
same card. That accounts for the whole difference between them and is the reason the earlier version of
this file described a 13% run-to-run move: the two tables were never the same configuration, and
comparing them across a driver change is not a measurement of variance.

### The two numbers that were asked for

**Shipped model shape, `256x128`**, which is `d_model 128` at a full context window of 256. The
kernel on its own is `7.31 us`; `norm.zig`'s arithmetic is `78.64 us`. On that measure the GPU is
`10.8x` faster, and that is the two cells in the `ship` row divided.

**Large shape, `4096x4096`.** The kernel is `232 us`; `norm.zig`'s arithmetic is `43.0 ms`. The GPU
is `185x` faster, and here the number is the **device memory** bandwidth it can actually reach --
device, not PCIe, and the distinction matters because the paragraph above is entirely about the bus:
`4096 * 4096 * 4` bytes in and the same out is 128 MiB, and 128 MiB in the kernel's own 232 us is
578 GB/s. This card's peak is
912 GB/s, which is arithmetic rather than a number quoted from a spec sheet: the 3080 Ti is a
384-bit GDDR6X part at 19 Gbps, and 384 / 8 * 19e9 is 912e9. So the kernel reaches 63% of it.
Quoted as a fraction of peak rather than as a microsecond count, because the
kernel column above is reproducible to under 1% on a fixed driver and this figure is not a
property of anything except this driver and this card.

### The crossover, and why the shipped-shape number is not the interesting one

The crossover is between `1024` elements, where the kernel loses (`3.68 us` against `2.37 us`), and
`4096` elements, where it wins (`3.16 us` against `9.98 us`). So it is a few thousand elements,
which is far below anything this model builds.

That floor is the whole story at the shipped size, and it is why the `10.8x` should not be read as
"this model would be faster on a GPU":

- The kernel takes `3.68 us` at 1024 elements, `3.16 us` at 4096, `4.53 us` at 16384 and `7.31 us` at
  32768. From 1024 to 32768 elements, thirty-two times the work, and the time only doubles. Almost
  all of it is launch and block scheduling. The actual work at `256x128` is 384 KiB of traffic -- the row is read twice and written once -- about
  `0.43 us` at the 912 GB/s this file derives from the norm table.
- The CPU side is slow for a reason that is specific to `norm.zig` rather than to RMSNorm. `78.64
  us` over `32768` elements is `2.40 ns` per element, and the reason is readable in the source: it is
  a serial `f64` accumulation, so each row is one dependency chain of 128 `f64` adds with no ILP to
  fill it. That is a correct and deliberate choice for the CPU implementation, and this kernel is
  not evidence that it was the wrong one.

The number that actually governs a decision is `gpu_e2e`, and at the shipped shape it is `50.45 us`
against a CPU call of `162.03 us`. This repository has no device-resident tensor type yet, so every
call copies 128 KiB in and 128 KiB out over PCIe, and that transfer is all but the `7.31 us` of
kernel: `50.45 - 7.31` is `43.14 us`, which is 55% of the `78.64 us` the CPU spends on the same
arithmetic. With 4 layers and 2 norms per layer, one forward pass spends `8 * 50.45`, about 404 us,
copying RMSNorm inputs before any arithmetic happens.

**So: no end-to-end speedup is claimed at the shipped model size.** The kernel is correct there, it
is faster than this CPU implementation there when the buffers are already on the device, and neither
of those is a reason to move this model onto a GPU today.

### Memory

At `4096x4096`:

| | |
|---|---|
| Device buffers: input, output, weight | 128.0 MiB |
| The same tensor in host memory | 128.0 MiB, and it starts there |
| Peak RSS of the CPU twin | 192.6 MiB |
| Peak RSS of the GPU process | 360.0 MiB |

The GPU path is the larger of the two, and the reason is not the device: it is that there is no
device-resident tensor, so the host keeps a full copy of the input and the output alongside the
device copies. RMSNorm is 2 reads and 1 write of the tensor in, 1 write out, so the device side
holds 2x the tensor while the host holds another 2x. A tensor type that kept its data on the device
would make the device side the only copy, and would make `gpu_e2e` the number that counts. Nothing
in this directory changes that yet.

### Known limitation

`tall` at `4096x128` is the same element count as `r256c2048` and takes `76.56 us` against
`8.60 us`, 8.9x worse, with no more memory traffic. The cause is the fixed block size: at
`cols = 128` only 128 of the 256 threads in a block have an element to read, so half the block idles,
and 4096 blocks each pay the full per-block cost for one element. Widening the row hides it and
narrowing it exposes it.

`ship` is `256x128` and costs `7.31 us`, so the shipped model is on the right side of this for its
row count, but a batch large enough to make 4096 rows out of 128-wide rows would land in the slow
case. The fix is to dispatch on a block size that fits the row, which needs the warp count to be a
template parameter rather than the `WARPS` constant it is now. Not done: it is an optimisation, not
a correctness fix, and the shapes this model actually builds are measured above.

## The width contract

`zt_attn_dim_ok` admits **`head_dim` even, in [32, 256]**. Two requirements, and they
are not the same kind of thing, so they are worth separating.

**Even is arithmetic.** Both K and V are staged with a row stride of `dim + 1`, padded
so a warp's store reaches 32 distinct banks rather than one, and the bank thread `c`
lands in is `(c * (dim + 1) + d) % 32`. That is a bijection across `c = 0..31` exactly
while `dim + 1` is **coprime with 32**, which is to say while `dim` is even. At an odd
`dim` the stride is even, the map collapses onto half the banks, and the conflict the
padding exists to remove comes back. It is also the width the layer above this one
refuses anyway: `rope.forward` returns `error.OddHeadDim` for an odd `head_dim`, so a
kernel that accepted one would be accepting a width nothing upstream can produce.

**The range is this directory's own choice and still a guard, not a hardware limit.**
`dim` is the block width, so below 32 the block is narrower than a warp and lanes sit
idle while the QK pass hands one whole key to each live thread; above 256 there is no
head this repository has been asked to grade. Relaxing either end wants a measurement,
and an unmeasured widening of a guard is how a correctness range becomes a performance
claim nobody made.

**A power of two was required until this section, and requiring one turned away three
shipping head widths.** The old guard tested `(dim & (dim - 1)) != 0` and so turned away
**96** (Phi-3's head width), **192** (DeepSeek-V2's and V3's MLA head width) and **80**, a width
several families ship, while the loops that would have run at those widths need nothing of the
kind: `i += dim` and `c = threadIdx.x % dim` are exact for any `dim`. The guard was left
over from when the tile was pinned to `dim` and the block had to divide a warp, and the
file's own comment said so while the code did the opposite. The predicate over the
boundaries, measured:

| `head_dim` | 0 | 1 | 30 | 31 | **32** | 33 | 63 | **64** | 65 | **80** | 95 | **96** | 97 | 127 | **128** | **160** | 191 | **192** | 255 | **256** | 257 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| admitted | 0 | 0 | 0 | 0 | **1** | 0 | 0 | **1** | 0 | **1** | 0 | **1** | 0 | 0 | **1** | **1** | 0 | **1** | 0 | **1** | 0 |

Every odd width is refused and every even width in the range is admitted. **80, 96, 160
and 192 are newly admitted**, and of those only 96 and 192 have a `max_abs` in this file.
32 and 128 are unchanged.

### The tile narrows, and the refusal is a return code

`zt_attn_forward` used to compute the tile as `min(head_dim, max_tile)`, hand the
resulting byte count to `cudaFuncSetAttribute`, and `exit(1)` when that failed. At
`head_dim` 256 with `max_tile` 64 the ask is
`256 + 64 + 2*64*257 = 33216` floats = **132864 bytes**, an sm_86 block will opt into
**101376**, and the answer is `cudaErrorInvalidValue`. Both halves of that were
measured rather than reasoned, by calling the previous `attn_kernels.cu` -- a file that
no longer exists -- at exactly that width:

```
probe: dim_ok(256)=1  shmem(256,64,1)=132864
attn: FAIL cudaFuncSetAttribute(...MaxDynamicSharedMemorySize, (int)shmem) at line 661: invalid argument
EXIT=1
```

`zt_attn_dim_ok(256)` returned **1**. The predicate admitted the width and the launcher
ended the process on it, which is the worst shape a failure can have: a caller sees
`exit(1)` from a call whose own gate said yes. The same probe against the current file:

```
probe: dim_ok(256)=1  shmem(256,64,1)=132864
probe: zt_attn_forward returned 0
EXIT=0
```

The fix is to spend the one variable that is free, the **tile**. It changes how many keys
an iteration stages and nothing about the arithmetic -- the QK pass hands each thread
whole keys, so a tile narrower than the block simply leaves the rest of the warp out of
that loop, and the online softmax's rescale is exact at any tile size. Both launchers
now scan down from the cap for the widest tile the device will grant, and `attn.cu` sizes
its own tile with the **same helper** against the **same ceiling**, because a benchmark
that sized its tile differently could not have caught this and one that sized it the
same way would have measured a configuration no caller could launch.

`attn.cu` prints what every entry point decided, once per manifest shape, so the decision
is a row rather than a claim. From the run above:

```
library entry points, one launch per manifest shape at T=8:
  ctx256       head_dim 32   tile 32/32   8704/8960   bytes  forward 0  backward 0
  llama3-T256  head_dim 128  tile 64/64  66816/67584  bytes  forward 0  backward 0
  gqa-96-T256  head_dim 96   tile 64/64  50304/50944  bytes  forward 0  backward 0
  gqa-96-T512  head_dim 96   tile 64/64  50304/50944  bytes  forward 0  backward 0
  mha-192-T256 head_dim 192  tile 64/64  99840/100864 bytes  forward 0  backward 0
  gqa-256-T256 head_dim 256  tile 48/48  99904/101120 bytes  forward 0  backward 0
```

Two rows are worth reading twice. `mha-192-T256` asks **99840** of the 101376 available,
so it runs at the cap on 1536 bytes of margin and is the shape most likely to need the
scan on a card with a smaller ceiling. `gqa-256-T256` is the only row where the scan moved
anything, and it moved it from 64 to 48.

At `head_dim` 256 the scan lands on **tile 48** -- 99904 bytes forward, 101120 for the
backward's kernel A -- inside the 101376 an sm_86 grants -- 1472 bytes of slack on the forward ask and 256 on the larger backward one. The
boundary, measured by handing the scan a stated limit:

| limit | 132864 | 101376 | 99904 | 99903 | 3084 | 3083 | 1 | 0 |
|---|---|---|---|---|---|---|---|---|
| tile chosen | 64 | **48** | 48 | 47 | **1** | 0 | 0 | 0 |
| bytes asked | 132864 | 99904 | 99904 | 97844 | 3084 | 3084 | 3084 | 3084 |

**The floor is a tile of 1, and it is correct rather than merely narrow.** At `n = 1` the
correction is `exp(mrun - max(mrun, score))`, which is 1 whenever the new score is not
the running max and `exp(negative)` otherwise -- the flash recurrence run one key at a
time. It is slow, which is a different problem from being wrong, and it is **graded
rather than argued**: `ATTN_MAX_TILE=1` runs the whole table with every key in its own
tile, and all 11 forward rows and all 33 backward gradients pass -- forward worst
`9.686e-08`, **0.097%** of the `1e-4` gate, and backward worst `7.153e-07`, **7.15%** of
the `1e-5` gate, both inside what the default cap already reads. (The timings from that
run are not recorded here for the same reason as the rest: one run per arm, on a host with
another process resident. It is worth knowing the direction -- a tile of 1 runs several
times slower than the shipped cap, which is 64x fewer keys staged per barrier -- and not
worth a row.)

**Below 1 there is nothing to try, so it is a refusal and not a clamp -- and on this card
that branch is unreachable.** Over every admitted `(head_dim, group_q)` the largest ask
at a tile of 1 is **6168 bytes** forward and **4112** for kernel A, against 101376
available, because `dim * group_q <= 1024` bounds the pair. So the code path that
returns 1 for want of shared memory is exercised only by handing the scan a limit
directly, as the table above does with 3083, 1 and 0. **On no configuration this file
admits can that refusal fire on an sm_86**, and a reader should treat the branch as
unproven against a real device rather than as covered.

## Fused causal attention, forward

Same card, same pinned container, same host as every table above: an RTX 3080 Ti, CUDA 12.6.3,
driver 615.71.09, against a Fedora host. Measured with the kernel in this directory rather than
transcribed.

One block owns one (query position, query head) pair and walks that query's causal prefix one tile at a
time. The running maximum, the running softmax denominator and one column of the output accumulator
live in registers for the whole walk, and the score matrix is never written to global memory. `dim` is
also the block width, one thread per output column, which is what lets a thread own one accumulator
element for the entire kernel and is why the QK pass needs no cross-thread reduction: thread `i`
computes the whole dot product for key `i` of the tile. The tile is capped below `dim` so that the K
and V tiles fit; at Llama-3's `head_dim` of 128 a full-width tile asks for 130 KiB of shared memory,
past what an sm_86 block will opt into, so the cap is what makes the kernel run at the head width the
goal actually names. At `head_dim` 256 the cap of 64 does not fit either, and the launcher **narrows
the tile until the ask does** rather than handing the number to the device and dying there -- see
[the width contract](#the-width-contract) above.

**What this run does not print, and what one block omits.** A full `sh src/cuda/run-attn.sh` prints,
beside the two parity tables: the `attn: running with ATTN_GROUP_Q=1` and `attn: ATTN_MAX_TILE=64`
lines; the `library entry points, one launch per manifest shape at T=8:` block, which calls
`zt_attn_forward` and `zt_attn_backward` at every width in the manifest and prints the tile each
chose and its two return codes; the `configurations used:` block listing every
`group_q`/block/tile combination the sweep reached; and the backward's four variant lines each
carrying its `[dq/dk/dv ...]` signature plus the `...and all four signatures differ` line. All of
that is omitted here and none of it is a number a reader would compare against.

The forward's four variant lines ARE kept, because "the parity gate can fail" is a claim this
directory should be able to show rather than assert.


One `sh src/cuda/run-attn.sh` on `2026-10-03`, after `zig build host-check` reported a clean host.
Both parity tables in full, and the forward's four broken-variant lines:

```
shape          ctx     cpu_us   kernel_us   max_abs     gate   used    ratio  parity  argmax
ctx256          256   10800.28     104.690  5.960e-08     1e-04  0.060%    103.2x  ok   argmax 517
ctx512          512   44694.46     336.031  5.960e-08     1e-04  0.060%    133.0x  ok   argmax 517
ctx1024        1024  188315.18    1160.164  5.960e-08     1e-04  0.060%    162.3x  ok   argmax 517
ctx2048        2048  761951.21    4576.780  5.960e-08     1e-04  0.060%    166.5x  ok   argmax 517
ctx4096        4096 3079732.55   18374.339  5.960e-08     1e-04  0.060%    167.6x  ok   argmax 517
llama3-T256     256  336746.01    3155.822  7.451e-08     1e-04  0.075%    106.7x  ok   argmax 76157
llama3-T512     512 1523552.59   11803.331  7.451e-08     1e-04  0.075%    129.1x  ok   argmax 76157
gqa-96-T256     256  255140.20    2468.086  5.960e-08     1e-04  0.060%    103.4x  ok   argmax 3792
gqa-96-T512     512 1022021.56    9295.332  5.960e-08     1e-04  0.060%    109.9x  ok   argmax 3792
mha-192-T256    256  251905.25    2411.664  1.043e-07     1e-04  0.104%    104.5x  ok   argmax 195211
gqa-256-T256    256  165910.12    1656.361  7.451e-08     1e-04  0.075%    100.2x  ok   argmax 43433

shape          ctx     bwd_us     max_abs_dq     max_abs_dk     max_abs_dv     gate   worst   parity      worst_dq      worst_dk      worst_dv
ctx256          256     588.80      7.451e-09      1.537e-08      2.384e-07     1e-05   2.38%  ok            170          119            5
ctx512          512    1474.56      7.451e-09      1.490e-08      2.980e-07     1e-05   2.98%  ok            170          184           39
ctx1024        1024    4727.10      7.451e-09      1.676e-08      3.576e-07     1e-05   3.58%  ok            170          108           39
ctx2048        2048   17725.34      7.451e-09      1.863e-08      2.384e-07     1e-05   2.38%  ok            170          108            5
ctx4096        4096   70319.87      7.451e-09      1.863e-08      2.980e-07     1e-05   2.98%  ok            170          108            5
llama3-T256     256   31140.86      2.980e-08      5.215e-08      7.153e-07     1e-05   7.15%  ok           7739          983         2043
llama3-T512     512  115709.92      2.980e-08      7.451e-08      8.345e-07     1e-05   8.34%  ok           7739         1017         2043
gqa-96-T256     256   18982.82      1.118e-08      4.098e-08      4.768e-07     1e-05   4.77%  ok           3370         2234           45
gqa-96-T512     512   69966.66      1.118e-08      4.843e-08      5.960e-07     1e-05   5.96%  ok           3370         2234          345
mha-192-T256    256   30832.64      2.235e-08      3.725e-08      2.384e-07     1e-05   2.38%  ok           3934         3993           73
gqa-256-T256    256   28313.25      2.095e-08      3.725e-08      4.768e-07     1e-05   4.77%  ok           5564         1246          248

attn: OK
attn: proving the parity gate can fail
attn: broken variant 1 was caught, as it must be
attn: broken variant 2 was caught, as it must be
attn: broken variant 3 was caught, as it must be
attn: broken variant 4 was caught, as it must be
```

**One run, so one sample per shape, and this is not the published table.** The root `README.md`
carries the minimum of three invocations, **57.9x to 168.1x** (min-of-3; the widest single run was 168.8x), because `ctx256`'s CPU column is
bimodal and a single sample there is a 1.74x coin toss -- `outputs/bench/ctx256-sweep.csv` is the ten
runs that establish it. The `10800.28 us` above is one draw from the high mode; the root table's
`57.9x` comes from a run that drew low. Both are correct and they differ by a factor, which is the
whole reason the minimum is the statistic and a single run is not one.

This block previously quoted `105.2x` at `ctx256` and a `10884.32 us` call from an earlier session,
and was withdrawn rather than re-pointed at numbers whose log was not kept. It is now a real run
whose log exists, which is the difference between the two states.

More than one shape's worth of geometry is in that table on purpose. The first five are this model's
own configuration; rows six and seven are Llama-3's, 32 heads over 8 kv heads at `head_dim` 128, because
a kernel only ever run at `head_dim` 32 has not been shown to run at the width the project is about.
Rows eight through eleven are the four widths the section below is about.

### The four widths the guard used to refuse

The table above carries all eleven rows because their timing columns are published and **the
`cpu_us` and `kernel_us` figures from the run that added these widths are deliberately
not recorded** -- the forward's CPU column at the shipped window is bimodal on this
host, and a timing taken alongside a parity change would be read as a speedup claim
that run cannot support. What that run *is* evidence for is parity, and parity is
reproduced here in full.

**The bimodality is measured, not asserted.** Ten runs of the CPU twin on this host,
each writing its own manifest, put `ctx256` in one of two clusters with nothing
between them. The ten samples are committed at `outputs/bench/ctx256-sweep.csv`:

| mode | samples | range (`cpu_us`) | internal spread |
|---|---|---|---|
| low | 3 of 10 | 6070.021 – 6088.290 | **0.30%** |
| high | 7 of 10 | 10583.812 – 10875.440 | **2.76%** |

The gap between them is **1.7384x**, and not one of the ten samples lands inside it.
Each mode is internally tight and the two are far apart, which is the whole point:
noise does not leave a gap. **70% of single runs land in the high mode**, so one
untimed sample at this shape is a 1.74x coin toss in either direction.

That is what makes the minimum the only safe statistic here rather than a
convention, and it also bounds what the minimum can promise. Taking the minimum of
three catches the low mode only when at least one of the three lands there, which
this distribution puts at 1 - 0.7^3 = **65.7%**: a third of min-of-3 tables at this
shape would still report the high mode. What flips it -- core placement, thread
affinity, a competing process -- is not identified, and nothing here claims it is.
The kernel column over the same runs is the steadier one, which is the opposite
assignment from what the ratio would suggest.

Measured `2026-10-02` on that same card, same pinned container, same host, one
`sh src/cuda/run-attn.sh` at the default tile cap of 64:

| shape | `head_dim` | heads/kv | ctx | tile | `max_abs` | gate | used | parity |
|---|---|---|---|---|---|---|---|---|
| `gqa-96-T256` | 96 | 32/8 | 256 | 64 | `5.960e-08` | `1e-4` | **0.060%** | ok |
| `gqa-96-T512` | 96 | 32/8 | 512 | 64 | `5.960e-08` | `1e-4` | **0.060%** | ok |
| `mha-192-T256` | 192 | 16/16 | 256 | 64 | `1.043e-07` | `1e-4` | **0.104%** | ok |
| `gqa-256-T256` | 256 | 8/4 | 256 | **48** | `7.451e-08` | `1e-4` | **0.075%** | ok |

And the backward, same run, same three separate gates:

| shape | `head_dim` | `max_abs_dq` | `max_abs_dk` | `max_abs_dv` | worst, % of `1e-5` |
|---|---|---|---|---|---|
| `gqa-96-T256` | 96 | `1.118e-08` | `4.098e-08` | `4.768e-07` | **4.77%** |
| `gqa-96-T512` | 96 | `1.118e-08` | `4.843e-08` | `5.960e-07` | **5.96%** |
| `mha-192-T256` | 192 | `2.235e-08` | `3.725e-08` | `2.384e-07` | **2.38%** |
| `gqa-256-T256` | 256 | `2.095e-08` | `3.725e-08` | `4.768e-07` | **4.77%** |

**The comparison that matters is with the seven rows above, and it is close.** The new
forward rows read 0.060% to 0.104% of the gate against 0.060% and 0.075% on the two old
widths: `gqa-96` at 96 is *exactly* as tight as `head_dim` 32, `gqa-256` at 256 exactly as
tight as Llama-3's 128, and `mha-192` at 192 is the one new row looser than anything
published, 1.4x the old worst and still **9.6x** inside the gate. The new backward rows
read 2.38% to 5.96% against a published spread of 2.38% to 8.34%, so all four are
**tighter than the worst row already in the table**. No gate was moved to make any of this
pass; `ATTN_TOL` and `ATTN_BWD_TOL` are the same two numbers as before the change.

Two details in that table are the change showing through rather than the arithmetic.
`mha-192-T256` runs 16 heads over 16 kv heads, so its group is **1** and there is no
grouped-query attention on it to collapse: it is the one row where the "collapse GQA to kv
head 0" variant is a different defect than at every other shape, because the correct
kernel already reads one kv head per query head there. And `gqa-256-T256` is the only row
that does not run at the cap: **tile 48**, 99904 bytes, inside the 101376 an sm_86 grants.
It is in the table because a width that runs is worth more than a width that refuses.

**The seven published rows did not move.** Same run, and every figure is the digit the
table above already published: `5.960e-08` at `argmax` 517 for the five `head_dim` 32 rows,
`7.451e-08` at `argmax` 76157 for the two at 128, and backward `7.451e-09` / `1.537e-08` /
`2.384e-07` at indices `170`, `119`, `5` for `ctx256`. Neither defect in this change has a
floating-point consequence -- the tile is a loop bound and the width test is a guard -- so
a difference here would have meant one of them had changed the arithmetic, and it did not.

### What changed, and what the change was worth

The kernel column is **1.71x to 2.10x faster than it was across the four rows of the gain table
below, 2.00x at the shipped geometry**, and the reason is one line in the shared-memory layout,
applied twice.

For a fixed inner index `d`, a warp of 32 threads writing row `i = threadIdx.x` addresses
`c * dim + d`. With `dim` a multiple of 32 -- 32 at the shipped geometry, 128 at Llama-3's -- every
one of those lands in bank `d`. **One distinct bank out of 32.** Padding the row stride to `dim + 1`
makes the bank `(c * (dim + 1) + d) % 32 = (c + d) % 32`, which is 32 distinct banks, a bijection on
the warp.

K was fixed first and the commit that did it called the conflict closed. It was not. **V's staging
store has the identical shape** -- `sv[i * dim + d]` with `i = threadIdx.x` -- and was left
conflicted. V's *reads* never had the problem, being `sv[i * dim + c]` with `c = threadIdx.x`, which
is stride-1 across the warp; the claim that V "was already conflict-free" was true of the reads and
false of the write, and it was the only place this defect survived the first fix. So both tiles are
padded, and V's reads stay conflict-free under the new stride too:
`(i * (dim + 1) + c) % 32 = (i + c) % 32` across `c = 0..31`.

Measured against the kernel as first published, and the `after` column is the table above rather than
an intermediate, because an intermediate is not a configuration this repository ships:

| Row | before | after | gain |
|---|---|---|---|
| `ctx256` | 206.8 us | 103.4 us | 2.00x |
| `ctx1024` | 2462.6 us | 1177.9 us | 2.09x |
| `ctx4096` | 39125.8 us | 18625.4 us | 2.10x |
| `llama3-T512` | 20210.1 us | 11817.7 us | 1.71x |

Two earlier versions of this table are worth naming because both were wrong in the direction that
makes the work look smaller or the numbers look more solid than they are. One stopped at the K-only
intermediate (144.2 / 1563.4 / 23743.9 / 13671.9 us, gains 1.43x to 1.65x) while the table above it
already carried the padded numbers, so the file contradicted itself twenty lines apart. The other
labelled its `before` column "the previous minimum", which was a relabelling of the previously
*published* row and therefore carried up to 2.6% of selection into the headline gain; no transcript
of those three runs is committed, so that selection cannot be recovered from this repository and is
not claimed.

Padding both tiles costs `2 * tile` floats per block: **256 bytes** at the shipped geometry and
**512 bytes** at Llama-3's. An earlier version of this paragraph said 128 bytes, which was the K-only
cost at `head_dim` 32 and understated the Llama-3 rows by half.

**Parity is bit-identical.** `max_abs` is `5.960e-08` and `7.451e-08` exactly as before, and so is
every `argmax`, because padding changes addresses and not the order of any floating-point addition.

**The ratio column moved further than the kernel did, and most of that is not the kernel.** At the
K-fix intermediate step the shipped window read 73.6x where the unpadded kernel had read 29.4x, a
factor of 2.5, while the kernel itself improved by 1.43. Neither figure is current: 73.6x belongs to
an intermediate this repository does not ship, and the table above publishes **103.2x** for the same
window. The distance between those ratios is the CPU column rather than the kernel -- 29.4x is
`6089.81 / 206.8` and 105.2x is `10884.32 / 103.442`, both quotient of a row quoted in this file --
and the shipped window's CPU reading is the one this host reproduces least. The kernel column is the
figure that moved because the code moved.

### What `max_abs` is and is not

`5.960e-08` at `head_dim` 32 and `7.451e-08` at 128, which is **0.060% and 0.075% of the `1e-4` gate**.
An earlier version of this file called that "one `f32` ulp at a value of 1.0" and that was wrong on
two counts: the ulp at 1.0 is `1.19e-07`, and these outputs are weighted averages of V values in
`[-0.5, 0.5]`, so `|out|` sits well below 1 and the exact binade cannot be read off this table. What
can be said is that the difference is one or a few units in the last place at the magnitudes
involved, at every one of the five context lengths and both geometries.

Also worth saying plainly: **the comparison is f32 against f32.** `attention.zig` narrows its output
to `f32` before it reaches the reference file, so the CPU's internal `f64` accumulator improves the
reference rather than widening what is compared, and "the closest two implementations can get is
zero" is true here in a way it would not be for an f64 comparison.

### Reproducibility, measured three ways

These are **observations, not reproducible outputs.** Three consecutive `sh src/cuda/run-attn.sh`
invocations on that host, and no transcript of them is committed, so a reader can check the `ctx256`
triple quoted below and nothing else.

| Column | Spread over three runs |
|---|---|
| `kernel_us` | 0.5% to 2.6% |
| `cpu_us` | 0.1% to 1.4%, except `ctx256` at **78.8%** |

So the GPU column is a figure this host can reproduce and the CPU column mostly is too. The exception
is the shipped window, where one `forward` call read 6089.81, 8572.32 and 10884.32 us, and that is
why the published run is the third of three, taken while the GPU read 0% -- the only one of the
three that can be. Its **kernel** column is the one that moved because the code moved; the CPU column
is noisy at the shipped window and is not what the paddings are measured against.

One discrepancy is **not** explained and is recorded rather than smoothed over. `zig build attn-bench`
measured one `attention.forward` call at `T = 4096` as 1.69e6 us in one session; this table's twin
measured 3.08e6 to 3.09e6 us in another. A factor of 1.8 on the one row where it matters most, both
minimum-of-three, both correct about their own method.

It was first written up as two tools disagreeing. That was wrong, and a controlled test says why. The
suspect was the input data -- `attn-bench` fills q, k and v from one seed and the twin from three --
and varying only that, on the same host by the same method, gave 3104936, 3102682 and 3100309 us for
three fill patterns: a 0.15% spread, with **both** tools reading 3.10e6. The data is not the cause
and the tools do not disagree. What the 1.8x records is that one measurement of this function on this
host has read 1.69e6 and 3.10e6 at different times, which is worse than a tool bug: it means the
number is not reproducible across host states. Until it is explained, no ratio in this file should be
read to better than an order of magnitude, and the speedup at the shipped window -- the row with the
widest CPU spread -- should be read as "at least one hundred times", not as a hundred and five.

### The gate, attacked before it is believed

`ATTN_BROKEN` selects a deliberately defective variant and the runner **fails** if any passes:
variant 1 drops the causal mask, variant 2 sends every query head to kv head 0 so GQA collapses to
MHA, variant 3 drops the running-max rescale so the online softmax never rescales what it already has,
variant 4 drops the `1/sqrt(head_dim)` scale. All four are caught, which is the only reason the clean
rows above mean anything. The comparison is also NaN-aware -- `NaN` counts as an infinite difference,
because every comparison against `NaN` is false and a kernel returning `NaN` everywhere would
otherwise report a worst difference of zero and pass.

### Known limitation: the prefix is re-read per block

Every block re-reads the whole causal prefix of K and V, because a block owns one query and a flash
style kernel keeps K and V resident across queries rather than re-deriving them per block. The
amplification over compulsory traffic, computed from the kernel's own loop:

| Shape | Re-read | Compulsory | Factor |
|---|---|---|---|
| `ctx256` | 0.034 GB | 0.39 MB | 86x |
| `ctx4096` | 8.59 GB | 6.29 MB | 1366x |
| `llama3-T512` | 4.30 GB | 20.97 MB | 205x |

A block reads **one** kv head -- `kvh = h / (n_heads / n_kv_heads)` -- so a query at `t` moves
`2 * (t + 1) * dim` floats, not `2 * (t + 1) * n_kv_heads * dim`. An earlier version of this table
used the latter and overstated every row by exactly `n_kv_heads`: 171x, 2731x and 1642x. The error is
the GQA collapse this file's own `ATTN_BROKEN=2` variant exists to detect, applied to the traffic
accounting instead of to the kernel.

That is the whole reason the kernel sits far above the `floor_us` in `zig build attn-bench`: at 4096 it
takes 18.6 ms against a 1.03 ms floor, which is `18.06x` it, and it is moving 8.59 GB to do it.

Several queries per block, so that one K/V load serves all of them, is the fix, and it is **written and
measured**: `group_q` in `attn.cu` implements it, and it wins by `1.15x` at 4096 and `1.34x` at
Llama-3's geometry while losing by `1.20x` at the shipped window. It ships at 1 because choosing per
shape needs a rule for when to switch, and that rule is not written. The measurement is in `attn.cu`,
not in the table above, because the table comes from the configuration that ships.

**What this file is not.** Not a training step *by default*. Both halves of attention run on the GPU,
but a training step reaches only one of them, and only when `src/model.zig`'s `cuda_attn` is true:
under that `comptime` switch `model.forwardWith` routes through `cudaForward`, and
`zig build cuda-attn-check` grades that path against `attention.forward` on a real step's tensors.
**Nothing in the Zig tree calls `device.Attn.backward`**, so the backward kernel has no training path
and is exercised by the harness above and by nothing else. With the switch false -- what ships -- no
binary that `train` and `verify` build links this file at all.

A KV cache exists (`src/kv_cache.zig`, six tests) and `src/decode.zig` decodes through it: a greedy
generation loop whose five tests run and pass, checking a cached decode step against a full forward
pass over the prompt and the second generated token against a full forward over the grown prompt.
`decode.cudaAttnStep` runs that step on the GPU and derives `q_offset = pos` and `n_keys = pos + 1`
from the cache itself, so the forward kernel's offset is exercised at every real decode position
rather than only at 0. **An earlier version of this paragraph said there was no generation loop and
no `q_offset` on a single-token query. Both were true when written and neither is now.**

The external parity comparison in `tools/removed/` runs entirely on the CPU and is untouched by
anything here, so the block-parity claim in `AGENTS.md` does not depend on this file existing.

The `gate` column in either table is `ATTN_TOL` / `ATTN_BWD_TOL` from this directory, and it is **not
one of the gates in `tools/removed/oracle.txt`.** Those eighteen are per-tensor tolerances against a
an external library reference, from 2e-6 on `attn_norm_out` to 2e-4 on `logits`; the two
systems share no number and grade different implementations of different things. The closest analogue,
`attn_ctx` at 2e-5, is tighter than the forward's 1e-4. A row passing here says nothing about that
one, and the block-parity claim rests entirely on the `oracle.txt` gates.

## Fused causal attention, backward

Two kernels, because the two gradients accumulate in opposite directions and causality makes that
asymmetric. `attnDqKernel` is query-outer: `dq[t]` sums over the prefix `s <= t` and `p_ds[s]` is a
different value for every `(t, s)` pair, so one block owns the whole chain in registers and writes
once. `attnDkDvKernel` is kv-outer: `dk[s]` and `dv[s]` sum over the suffix `t >= s` across every
query head sharing that kv head, so making `s` outer keeps the accumulation in one block and **needs no
atomics**. A single fused kernel would have to pick one outer loop and reach the other gradient through
`atomicAdd`, which makes the summation order -- and therefore `max_abs` -- differ between runs, and
this directory's published three-run reproducibility rests on that column being stable.

Kernel A also writes the three per-row scalars kernel B reads: the row max, the softmax denominator,
and `delta = sum_s probs * d_probs`, the softmax Jacobian's other half. The reference accumulates that
sum inline as `dot_pp`; it is the same real number as the rowsum-of-dout-times-out that a flash backend
precomputes, and precomputing it is what makes the two-kernel split possible at all.

```
shape          ctx     bwd_us     max_abs_dq     max_abs_dk     max_abs_dv     gate   worst   parity
ctx256          256     588.80      7.451e-09      1.537e-08      2.384e-07     1e-05   2.38%  ok
ctx512          512    1467.26      7.451e-09      1.490e-08      2.980e-07     1e-05   2.98%  ok
ctx1024        1024    4714.40      7.451e-09      1.676e-08      3.576e-07     1e-05   3.58%  ok
ctx2048        2048   17962.98      7.451e-09      1.863e-08      2.384e-07     1e-05   2.38%  ok
ctx4096        4096   70810.97      7.451e-09      1.863e-08      2.980e-07     1e-05   2.98%  ok
llama3-T256     256   31321.09      2.980e-08      5.215e-08      7.153e-07     1e-05   7.15%  ok
llama3-T512     512  115512.32      2.980e-08      7.451e-08      8.345e-07     1e-05   8.34%  ok
```

**There is no ratio column, and its absence is the point.** The CPU backward is `O(T^2 * H * dim)` in
f64 across three gradient loops; the forward's twin already reads milliseconds per call at `ctx4096`, so
a backward call is a multiple of that. This file does not time it, and a ratio with no measured
denominator is exactly the thing the rest of this README refuses to print. The `bwd_us` column is the
minimum of three timed launches of the kernel pair and nothing else -- no transfer, no host
synchronisation beyond the event pair, no allocator.

**Three gates, not one.** A single worst difference across `dq`, `dk` and `dv` cannot say *which*
gradient is wrong, and grading them together would let a broken `dk` hide behind a good `dq` at a length
where `dk` happens to be small.

**The gate is `1e-5`, set from the measurement.** The first working run read a worst of `8.345e-07`, so
`1e-5` leaves 12x over the worst row and 4x over the spread between shapes. It was `1e-4` before that,
which is 120x -- a gate that cannot detect anything this project has evidence to look for. The norm
kernel's own gate is `1e-5` at 28.6% used, and setting a gate from the measurement and recording why is
what that table did.

**The order of the three columns is the diagnostic.** `dq` is smallest because the reference accumulates
it in f64 and narrows once at the store, so its entire budget is a single f32 rounding of a
well-conditioned sum. `dv` is largest because the reference accumulates it as an f32 *read-modify-write*,
one narrowing per term, in a chain up to `group * (T - s)` long -- 2048 roundings at `llama3-T512`
against `dq`'s one, and `dv` reads 28x worse. **The ratio is the chain length.** So `dv` is the largest
error in the table because the reference is least accurate there, not because the kernel is: a kernel
accumulating `dk` and `dv` in f64 registers and narrowing once would be more accurate in absolute terms
and would disagree with this reference by about `1e-6`. That would be worse, not better, against the
thing being graded.

**Reproducibility.** Three runs at 0-2% GPU put `bwd_us` at `588.80 / 592.90 / 587.78` at
`ctx256`, a 0.9% spread, and every `max_abs` figure above is identical to the digit across all three.
The GPU state is recorded per run because it has to be: a further run on that host read 57% of the
GPU from another process, and it is excluded from every figure quoted here -- this `bwd_us` triple
and the published forward column alike.

**The gate, attacked before it is believed.** Four broken variants over a separate `ATTN_BWD_BROKEN`,
because they are different defects from the forward's four and one knob driving both would make
"variant 3 was caught" mean two things in a run. A variant failing is necessary but **not sufficient**:
while the correct kernel was itself broken, all four failed vacuously and the loop printed "caught" four
times while proving nothing. So the loop now also requires the four signatures to *differ*:

| variant | defect | dq | dk | dv |
|---|---|---|---|---|
| correct | -- | 7.451e-09 | 1.537e-08 | 2.384e-07 |
| 1 | kv-outer loop walks the prefix, not the suffix | 7.451e-09 | 2.444e-01 | 1.536e+00 |
| 2 | GQA collapses to kv head 0 | 3.835e-02 | 7.273e-02 | 4.992e-02 |
| 3 | the softmax Jacobian term is dropped | 5.224e-02 | 2.121e-01 | **2.384e-07** |
| 4 | the `1/sqrt(dim)` scale is dropped | 1.463e-01 | 2.449e-01 | 3.018e-01 |

Variant 3 is the one worth reading twice. It leaves `dv` **bit-identical to the correct kernel**, because
the Jacobian term lives in `p_ds`, which reaches `dq` and `dk` and never reaches `dv`. A gate that
watched only `dv` would see variant 3 pass -- which is the whole argument for three separate gates.

**One caveat that makes this gate narrower than it looks.** `attn_twin.zig` fills `dout` from its own
seed, uniform on `[-0.5, 0.5)`. The real `dout` at the attention boundary during training arrives from
`dLossDLogits`, where a non-target entry is about `(1/vocab)/T = 4e-6` and the target entry about
`4e-3`. So this table is a statement about the benchmark, on a `dout` two to three orders of magnitude
larger than a training step will supply, and applied unchanged to a real step `1e-5` is a much looser
*relative* gate than it sounds. A training step needs a scale-relative budget; `gradcheck.zig` already
has that machinery and it has not been applied here. Recorded as a known limit rather than left
implied.

## What the whole swap is worth, measured

Everything above measures an attention *call*. This section measures what moving attention off the CPU
is worth for a **training step**, which is the only question that decides whether the wiring work is
worth doing. It is here because an audit found that the figure it needs did not exist anywhere: the
forward's CPU cost had a published per-call number with a 78.8% run-to-run spread, the step time was
measured on a *different host*, and the backward's cost was the phrase "about four times the work".
Three numbers, none of them usable together.

All three below are from **one host in one session**, which is the whole point. Host is the Linux host of record,
32 cores, Linux 7.2.7, load 4.3 to 4.4 across the rounds.

| | measured | how |
|---|---|---|
| one training step | **1632.4 ms** | `zig build bench`, median of 3 runs of 123 steps, spread **0.1%** |
| CPU attention forward | **10707.91 us**/call | `zig build attn-bench`, minimum of 47 calls at ctx 256 |
| CPU attention backward | **19239.03 us**/call | `sh src/cuda/run-attn.sh`'s twin, minimum of 3 calls at ctx 256 |

The backward row is new: `attn_twin.zig` now times `attentionBackward` at every shape it generates, and
it is the only place in the repository where that cost is measured at all. Before this the only figure
available was an estimate.

**The forward row is the weak one and the table should not be read as though it were not.** Across
three sessions on this host the CPU forward at ctx 256 read **6450.78, 10707.91 and 10770.23 us** -- a spread
of 1.67x, which is the same 78.8%-class instability the forward's CPU column has always had and the
reason this section exists. The **backward** is stable where the forward is not: 19239.03, 19435.81 and
19450.54 us across the same sessions, a spread of **1.1%**. So the backward figure below is solid and the
forward figure is one sample of a noisy quantity, and the share and the speedup are quoted as a range for
that reason rather than as a point.

That asymmetry is itself worth a sentence. The backward reads `probs` and `d_probs` out of two f64 row
buffers it has already computed and walks `s` twice over short contiguous rows; the forward rebuilds a
`T x T` score row and its allocation and page-fault behaviour dominate at the shipped size. Neither
explanation was verified -- they are the obvious candidates and both are checkable -- but the measurement
is not ambiguous about which number is trustworthy.

**The backward is 1.80x the forward, not the ~4x that estimate suggested.** That single correction is
most of the difference between this section's answer and the 1.1x to 1.6x band the same audit guessed at.

```
CPU attention, 4 layers, per step
  forward    10707.91 us x 4 =  42.83 ms
  backward   19239.03 us x 4 =  76.96 ms
  total                       = 119.79 ms      =  7.34% of a 1632.4 ms step

GPU attention, per step, from the tables above
  forward   103.442 us x 4 =   413.8 us
  PCIe, 1.5 MiB round trip  =   257.8 us   [forward half of the 3.0 MiB derived below]
  launch/event overhead     =    30   us   [ASSUMED, measured by nothing here]
  total                     =   701.6 us     =  0.04% of a step

FORWARD ONLY -- the swap a step can actually make today:
  step after the swap: 1632.4 - 42.83 + 0.70 = 1590.3 ms
                                        speedup = 1.027x

IF THE BACKWARD WERE ALSO WIRED -- a projection, not a measurement:
  backward  588.80 us x 4 =  2355.2 us
  PCIe, 4 MiB round trip    =   687.6 us   [the round-up the derivation below explains]
  total                   =  3486.6 us     =  0.21% of a step
  step after the swap: 1632.4 - 119.79 + 3.49 = 1516.1 ms
                                        speedup = 1.077x
```

**This block is a projection, and one input to it cannot be reached by a step today.** `bwd_us` is
measured correctly, but only by this directory's harness: nothing in the Zig tree calls
`device.Attn.backward`, so a training step could not execute the `588.80 us x 4` line as written.
The number is right for the kernel; what is missing is the call site that would let a step use it.
Every other figure in the block is an input that IS reachable -- the forward through `cudaForward`,
the PCIe round trip, and the step total.

**The 4 MiB is a round-up, and here is the derivation it rounds.** At the shipped geometry
(`T 256`, 4 heads over 2 kv heads, `head_dim 32`) the forward moves `q`, `k` and `v` in and `out`
out: `(2*256*4*32 + 2*256*2*32) * 4` bytes, which is the `pcieBytes` formula in `src/attn_bench.zig`
and is 384 KiB per layer. The backward adds `dout` in and `dq`, `dk`, `dv` out, another 384 KiB per
layer, because `q`, `k` and `v` are already on the device from the forward and do not cross the bus
twice. That is 768 KiB per layer and **3.0 MiB** over four layers, which at the 6.1 GB/s rate is
`515.7 us`. The block above carries 4 MiB, which is the count that treats `q`, `k` and `v` as
re-sent for the backward, and rounds 687.6 us -- a `687.6 - 515.7 = 171.9 us` overcharge, 5% of the
total. It is carried rather than corrected because it moves the total by 0.01% of a step and every
number in the two blocks is a sum of parts that were each measured somewhere else.

**The `30 us` of launch and event overhead is an assumption, and no run in this repository measures
it.** It is there because a step pays it and omitting it would flatter the GPU side; there is no
transcript behind it and it is the one number in the block with no provenance. At 0.0018% of a step
it does not matter, but it is not a measurement and should not be read as one.

Using the fastest forward this host has read rather than the Stage 0 sample, because both are real
observations and the optimistic one is the one that survives scrutiny:

```
  forward  6450.78 us x 4 =  25.80 ms   (the low end of a 1.67x spread)
  backward 19239.03 us x 4 = 76.96 ms
  total                       = 102.76 ms  =  6.30% of a 1632.4 ms step
  step after the swap: 1632.4 - 102.76 + 3.49 = 1533.1 ms
                                        speedup = 1.065x   [projection]
```

**So: attention is 6.3% to 7.3% of a training step, and the reachable forward-only swap is worth 1.016x to 1.027x.** The range is the ctx256 CPU mode again -- at the low mode the CPU forward is 25.80 ms and the swap 1.016x, at the high mode 42.83 ms and 1.027x. Both ends
are quoted because the forward's spread is a factor of 1.67 and pretending to a third significant digit
across that spread would be the exact thing this README keeps refusing to do elsewhere.

**So the honest answer is 1.016x to 1.027x on a training step for the swap the code can make today, and attention is 6.3% to 7.3% of it. The wider 1.065x to 1.077x range earlier in this file assumed the backward was swapped too, which no step can do.** That
is worth stating plainly because the per-call numbers are 57.9x to 168.1x and they invite the conclusion that
the step will be many times faster. It will not be. The other 92.7% to 93.7% of a step is the matmuls,
the two norms, RoPE, SwiGLU, the tied head and AdamW, and **all of it is still on the CPU.** This
repository has one primitive kernel and three attention kernels; that is the entire device-side
inventory.

Two consequences, and they point in opposite directions:

- **Wiring attention in is worth doing to close the goal**, because the goal is that attention runs on
  CUDA, and a kernel nothing calls is a benchmark. 1.027x is a real improvement and it is not
  the reason.
- **If the objective is a faster step rather than a closed goal, attention is the wrong target.** The
  arithmetic points at `tensor.matmul`, which the CPU side already spends the majority of its time on and
  which has no kernel at all. That is a different and much larger piece of work, and it is named here
  rather than left implied.

**What is deliberately NOT claimed.** No number above is a measurement of a training step that calls the
GPU, because no such step exists. Every GPU figure is a kernel launch measured in isolation and then
*added up* with a PCIe estimate; the sum is arithmetic on measured parts, not a measurement of the whole.
The PCIe figure uses the 6.1 GB/s rate from `attn_bench.zig`, which that file already calls the weakest
number in it. A real end-to-end run would also move every layer's cached intermediates or not, and that
choice is worth 4.8x the attention-only transfer -- 3.3 ms, or 0.20% of a step -- depending on how far the
device residency goes. So that range is an estimate with a measured numerator, and it should be read as
the size of the prize, not as a result.

## The toolchain

The toolchain is the container, and `cuda.sh` is the whole recipe. That is a pinning decision
rather than a claim about what any host has installed: the table above was measured with this
image, and a repository that compiled with some newer host toolkit while measuring with this one
would carry two toolchains whose numbers described different builds. A host may install whatever it
likes; nothing here depends on it, and `zig build cuda-check` compiles through the same container so
the check cannot drift onto a second toolchain. Pull the image once — `docker images` reports
11.4 GB for this tag, not the 3 GB an earlier draft of this file claimed:

```sh
docker pull nvidia/cuda:12.6.3-devel-ubuntu22.04
```

Four things in `cuda.sh` each exist because they were got wrong once:

**The container writes root-owned files into the bind mount.** Without `--user`, every `.o` it
produces is owned by uid 0 and the next host-side build cannot overwrite it. `--user $(id -u):$(id -g)`
fixes it. The script pairs that with an `--entrypoint` bypass, because the image's own entrypoint
runs `ldconfig` and needs root to write `/etc/ld.so.cache`; it would fail before `nvcc` ever ran,
under exactly the `--user` this needs. `--gpus all` hands the container the driver library and device
nodes directly, so bypassing the entrypoint costs nothing.

**`-arch` comes from the device at run time**, not from a written-down constant. `sm_86` and `sm_89`
are one character apart and produce `cudaErrorNoKernelImageForDevice` at launch, which reads like a
broken GPU.

**`-Werror all-warnings` and `-Xcompiler -Werror`**, because the Zig build treats a warning as an
error and a CUDA file that is lax about it is the one nobody looks at until it is the only thing
broken. Verified to bite rather than merely to be present: an unused variable in a scratch `.cu`
comes back as `error #177-D` rather than as a warning.

**`-Xcompiler -fPIC`**, because the object is destined for a Zig-linked binary and the Zig build
links a position-independent executable by default on Linux.

The container bind-mounts the build root at the same absolute path it has on the host. A build that
compiles at one path and links at another bakes the container's path into the object and the host
then fails to resolve it.

### `addCudaFile` does not exist in Zig 0.16.0

`grep -rn addCudaFile` over `std/` returns nothing. The real API is `Module.addObjectFile(object:
LazyPath)` and `Module.linkSystemLibrary("cudart", .{})`, which takes three arguments.
`Step.Run.addOutputFileArg(basename) LazyPath` returns exactly the
`LazyPath` that `addObjectFile` wants, so an nvcc step and a link can be wired together without a
temporary path or a file read back off disk.

The FFI is two files and a training step reaches one of them: `src/model.zig`'s `cuda_attn` routes a real step's forward through `device.Attn`, and `zig build cuda-attn-check` grades that against the CPU one. The backward half has no caller in the Zig tree and so has no training path.
`attn_kernels.cu` is the C half: the three attention kernels plus six `extern "C"` entry points --
`zt_attn_forward`, `zt_attn_backward`, `zt_attn_dim_ok` and three shared-memory-size functions --
taking raw device pointers and returning 0 or 1 with the reason on stderr. None of them allocates;
the caller owns the memory. `device.zig` is the Zig half, described two sections below: six
`extern fn` declarations over the same six names, a checked size computation and an 11-buffer
device holder, described at the end of this section. Between them they are the closest this
repository comes to a device path, and it is still an unlinked object plus a module no call site
imports.

**`attn.cu` now calls the two launchers, once per manifest shape, at `T = 8`.** That is the
whole of what changed, and it is the thing that turned a claim into a measurement: before it,
the only program in the repository with a GPU launched the *kernels* through the `BROKEN`
templates and never executed the two functions a caller would use to do any work, which is
exactly how `zt_attn_forward` could carry a `cudaFuncSetAttribute` that ends the process at
`head_dim` 256 and still be described as untested rather than broken. The block prints the tile
the launcher chose and both return codes, a non-zero code fails the run, and it is why
[the width contract](#the-width-contract) can show `forward 0  backward 0` at every width from
32 to 256 rather than only asserting it. The device memory behind it is deliberately
uninitialised and nothing is compared -- a number nobody reads cannot be wrong -- so this is a
check that the entry point **accepts** a configuration, not a second parity gate. The other
three of the six, `zt_attn_dim_ok` and the shared-memory-size functions, are read by `attn.cu`
directly; `device.zig` is called by `src/model.zig` whenever `cuda_attn` is true.

It is `#include`d by `attn.cu` rather than linked beside it, deliberately: one translation unit means the
benchmark and a training step compile the SAME kernels and there is no second copy to drift. The proof
that the split changed nothing is that every `max_abs` and every `argmax` index in both tables came out
bit-identical across it -- `5.960e-08` and `7.451e-08` forward, `7.451e-09` / `1.537e-08` /
`2.384e-07` backward, and indices `170`, `119`, `5` and `7739`, `983`, `2043`. Only timings moved, and
only within their published spread.

**Which file a caller links is `attn_kernels.cu`, NOT `attn.cu`.** This cost an hour to work out, and
the wrong answer looks like a real blocker: linking a C caller against the object built from `attn.cu`
fails with `multiple definition of 'main'`, because that file still holds the benchmark's entry point.
The fix is not to split `main` out or to guard it behind a macro. `attn_kernels.cu` carries its own
`#include`s, holds no `main`, and *is* the library object -- it compiles to exactly the six entry points
and nothing else. `attn.cu` is the benchmark, which `#include`s the kernels file and adds the harness.
So:

```sh
# WRONG: this is the benchmark, and its main collides with the caller's
nvcc $FLAGS -c -o attn.o src/cuda/attn.cu

# RIGHT: this is the library. Six symbols, no main.
nvcc $FLAGS -c -o attn_kernels.o src/cuda/attn_kernels.cu
```

`nm -g --defined-only` on the second reports the six `zt_attn_*` and no `u`/`T main`, which is the check
to run before wiring a build edge, because a wrong file here fails at link time with a message about
`main` that says nothing about which of two files was meant.

**How this file is wired, stated so nobody reads the paragraph above as more than it is.**
`src/cuda/device.zig` holds the Zig-side `extern` declarations, a checked size computation and an
11-buffer device holder, with seven tests on the arithmetic. It is called whenever `cuda_attn` is
true, and `build.zig`'s `linkCudaAttn` links `attn_kernels.o` into three modules -- the CUDA test
binary, the decode test binary and the training executable -- so the FFI is reachable from an actual
caller rather than from C++ alone. Two things are still not in the graph, and neither is hidden.
`linkSystemLibrary("cudart")` NAMES the library; it does not supply it, so the pinned 12.6.3
runtime has to be copied out of the pinned image -- not taken from the host's own copy, which is a
different version and the wrong one -- onto BOTH `LIBRARY_PATH`, which the linker reads, and
`LD_LIBRARY_PATH`, which the loader reads and `LIBRARY_PATH` does not feed. The recipe two sections
below does that. And those build edges cannot be exercised on a Mac at all: the object is ELF and
the executable would be Mach-O, so the gate is NVIDIA-only by construction rather than by policy.

`zig build cuda-check` does reference this directory: it sources `cuda.sh` and calls `nvcc`
through `cuda()`, the same pinned container the two runners use, so the check compiles exactly what
gets measured. It does not take its input from `run-probe.sh --emit-object` -- it names the source
files itself, which is why the object the runners hand back is still unused by the build graph.

### `libcudart` is not on the host

Neither runner needs it, because both compile and run inside the container. A `zig build` edge that
links a device binary outside the container would, and it needs the runtime copied out.

**The host may already have a `libcudart`, and using it would be the wrong one.** The host of record carries a
CUDA 13.4 toolkit and `ldconfig -p` resolves `libcudart.so` to
`/usr/local/cuda/targets/x86_64-linux/lib/`, so a linker there finds `-lcudart` with no help at all.
That runtime is **13.4**, and this repository pins **12.6.3**: `AGENTS.md` records the same tension for
the compiler and refuses it there -- "one repository should not carry two toolchains whose numbers would
then describe different builds". Linking against the host's copy would reintroduce exactly that, one
level down and invisibly, because nothing would fail and every number would still print. So the recipe
below copies the pinned version out of the pinned image rather than accepting the one already present:

```sh
mkdir -p "$HOME/.local/lib"
# The glob MUST be expanded inside the container. Unquoted, the host shell expands it
# -- and on a host whose own toolkit lives at /usr/local/cuda, that is the 13.4 copy,
# so the recipe silently installs the wrong version and then fails much later for a
# reason that does not name the version. Two mistakes, both measured on this host.
docker run --rm -v "$HOME/.local/lib:/out" \
  --entrypoint /bin/sh nvidia/cuda:12.6.3-devel-ubuntu22.04 \
  -c 'cp -P /usr/local/cuda/lib64/libcudart.so* /out/'

# BOTH of these, or the link succeeds and the LOADER fails. This was the first version
# of this recipe and it was half a recipe: `LIBRARY_PATH` is read by the linker at
# link time, the loader never looks at it, and the result is
#   error while loading shared libraries: libcudart.so.12
# at run time on an object that linked cleanly a moment earlier.
export LIBRARY_PATH="$HOME/.local/lib:$LIBRARY_PATH"
export LD_LIBRARY_PATH="$HOME/.local/lib:$LD_LIBRARY_PATH"

# Confirm which one you got, before trusting any number built with it.
ldconfig -p | grep libcudart    # must NOT resolve under /usr/local/cuda
```

`-P` copies the `libcudart.so` -> `libcudart.so.12` -> `libcudart.so.12.6.85` symlink chain rather
than dereferencing it, because the linker looks for the unversioned name and the loader looks for the
versioned one. `LIBRARY_PATH` and `LD_LIBRARY_PATH` rather than a symlink into `/etc/ld.so.conf.d`,
because the host has no root and there is nothing to run `ldconfig` with.

## How the CPU side is invoked, and why

`norm_twin.zig` is not a build target and `build.zig` does not mention it. It runs as a root
module with a `src/` file brought in as a second module, because a module root in Zig 0.16 may not
import a file outside its own directory and `zig run src/cuda/norm_twin.zig` cannot reach
`../norm.zig` at all:

```sh
zig build-exe -OReleaseFast -OReleaseFast --dep ztransformer \
  -Mroot=src/cuda/norm_twin.zig -Mztransformer=src/norm.zig \
  --cache-dir "$SCRATCH/zig-cache" --global-cache-dir "$SCRATCH/zig-global" \
  -femit-bin="$SCRATCH/norm_twin"
```

Two details there are not incidental. The `Tensor` type is read off `forward`'s signature with
`@typeInfo` rather than imported, because `src/norm.zig` is the module root and a module exposes its
root's declarations but not the root's imports, and declaring `src/tensor.zig` as a second module
fails outright since a file may belong to only one module. And both caches are pointed into the
scratch directory so a run leaves nothing in `~/.cache` and cleanup is one `rm -rf` rather than a
judgement about what was there before.

`run-norm.sh` builds the twin with the host's zig and compiles and runs the `.cu` in the container.
They meet only on files, which is why neither side needs the other's language.

## Cleanup

`run-norm.sh` and `run-attn.sh` write everything under `.zig-cache/cuda/norm` and
`.zig-cache/cuda/attn` respectively, both gitignored, and both remove it on every exit path including
a failing one. Both zig caches are redirected there for the same reason. Neither leaves a container,
a cache or an object behind, neither writes outside the repository, and neither commits anything.
