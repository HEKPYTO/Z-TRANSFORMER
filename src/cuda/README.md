# src/cuda

Nine files. Two kernels, each with a CPU twin and a runner, plus the shared container recipe and the
older toolchain probe.

| File | What it is |
|---|---|
| `norm.cu` | The RMSNorm kernel: the parity gate and the benchmark. All of it, no CPU reference. |
| `norm_twin.zig` | Generates inputs, runs `norm.forward`, writes the reference output and its own timings. |
| `run-norm.sh` | Compiles and runs both halves and exits non-zero on a failed check. |
| `attn.cu` | The fused causal attention kernel, forward only: one transformer operation, the parity gate, the benchmark, and four deliberately broken variants. |
| `attn_twin.zig` | Generates inputs at two head geometries, runs `attention.forward`, writes the reference and its own timings. |
| `run-attn.sh` | Compiles and runs both halves, proves all four broken variants are caught, and exits non-zero if any check fails. |
| `run-probe.sh` | Compiles and runs the probe, the same container recipe, no parity. |
| `cuda.sh` | The container, image pin and nvcc flags, shared by all three runners. |
| `probe.cu` | Toolchain probe. It is the only file here that is not a transformer operation. |

## Run it

| Command | What it does |
|---|---|
| `sh src/cuda/run-norm.sh` | Pulls the image if absent, builds the twin, generates 18 shapes, checks parity, proves the gate can fail, benchmarks, removes everything. |
| `sh src/cuda/run-attn.sh` | Same three stages for the attention kernel: builds the twin, generates 7 shapes at two head geometries, checks parity, proves all four broken variants are caught, benchmarks, removes everything. |
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
  all of it is launch and block scheduling. The actual work at `256x128` is 256 KiB of traffic, about
  `0.3 us` at peak bandwidth.
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
goal actually names.

The first of the three runs below is the one published.

```
shape          ctx     cpu_us   kernel_us   max_abs     gate   used    ratio  parity  argmax
ctx256          256   10884.32     103.442  5.960e-08     1e-04  0.060%    105.2x  ok   argmax 517
ctx512          512   46742.28     328.468  5.960e-08     1e-04  0.060%    142.3x  ok   argmax 517
ctx1024        1024  189429.92    1177.853  5.960e-08     1e-04  0.060%    160.8x  ok   argmax 517
ctx2048        2048  761452.22    4653.296  5.960e-08     1e-04  0.060%    163.6x  ok   argmax 517
ctx4096        4096 3080129.43   18625.433  5.960e-08     1e-04  0.060%    165.4x  ok   argmax 517
llama3-T256     256  337659.88    3167.307  7.451e-08     1e-04  0.075%    106.6x  ok   argmax 76157
llama3-T512     512 1551863.40   11817.677  7.451e-08     1e-04  0.075%    131.3x  ok   argmax 76157
attn: OK
attn: proving the parity gate can fail
attn: broken variant 1 was caught, as it must be
attn: broken variant 2 was caught, as it must be
attn: broken variant 3 was caught, as it must be
attn: broken variant 4 was caught, as it must be
```

Two shapes' worth of geometry are in that table on purpose. The first five are this model's own
configuration; the last two are Llama-3's, 32 heads over 8 kv heads at `head_dim` 128, because a
kernel only ever run at `head_dim` 32 has not been shown to run at the width the project is about.

### What changed, and what the change was worth

The kernel column above is **1.43x to 1.65x faster than it was**, and the reason is one line in the
shared-memory layout. The QK phase read `sk[i * dim + d]` with `i = threadIdx.x`, so for a fixed `d`
all 32 threads of a warp addressed `c * dim + d`; with `dim` a multiple of 32 that is bank `d` for
every one of them. **One distinct bank out of 32, on every shared read of the hot loop.** Padding the
K row stride to `dim + 1` makes the bank `(c * (dim + 1) + d) % 32 = (c + d) % 32`, which is 32
distinct banks, and costs 128 bytes per block. V was already conflict-free -- it is read
`sv[i * dim + c]` with `c = threadIdx.x`, which is stride-1 -- so it was left alone.

Measured as the minimum of three runs against the previous minimum, because the CPU side of this
table is noisy and the kernel side is not:

| Row | before | after | gain |
|---|---|---|---|
| `ctx256` | 206.8 us | 144.2 us | 1.43x |
| `ctx1024` | 2462.6 us | 1563.4 us | 1.58x |
| `ctx4096` | 39125.8 us | 23743.9 us | 1.65x |
| `llama3-T512` | 20210.1 us | 13671.9 us | 1.48x |

**Parity is bit-identical.** `max_abs` is `5.960e-08` and `7.451e-08` exactly as before, and so is
every `argmax`, because padding changes addresses and not the order of any floating-point addition.

**The ratio column moved further than the kernel did, and most of that is not the kernel.** It went
from 29.4x to 73.6x at the shipped window, a factor of 2.5, while the kernel improved by 1.43. The
difference is the CPU column: this run read 10818.59 us where the earlier one read 6089.81 us, and
the ratio is a quotient of the two. The kernel column is the figure that moved because the code
moved.

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
is the shipped window, where one `forward` call read 6089.81, 8572.32 and 10890.25 us, and that is why
The published run is the third of three, taken while the GPU read 0% -- the only one of the three that
can be. Its **kernel** column is the one that moved because the code moved; the CPU column is noisy at
the shipped window and is not what the paddings are measured against.

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

**What this kernel is not.** Forward only. There is no backward, so nothing here is wired into
`zig build train` and this is not yet a step anyone can train through. The external parity
comparison in `tools/removed/` runs entirely on the CPU and is untouched by anything here, so the
block-parity claim in `AGENTS.md` does not depend on this file existing.

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

`zig build cuda-check` does reference this directory now: it sources `cuda.sh` and calls `nvcc`
through `cuda()`, the same pinned container the two runners use, so the check compiles exactly what
gets measured. It does not take its input from `run-probe.sh --emit-object` -- it names the source
files itself, which is why the object the runners hand back is still unused by the build graph.

### `libcudart` is not on the host

Neither runner needs it, because both compile and run inside the container. A future `zig build`
edge that links a device binary outside the container would, and it needs the runtime copied out:

```sh
mkdir -p "$HOME/.local/lib"
docker run --rm -v "$HOME/.local/lib:/out" nvidia/cuda:12.6.3-devel-ubuntu22.04 \
  cp -P /usr/local/cuda/lib64/libcudart.so* /out/
export LIBRARY_PATH="$HOME/.local/lib:$LIBRARY_PATH"
```

`-P` copies the `libcudart.so` -> `libcudart.so.12` -> `libcudart.so.12.6.85` symlink chain rather
than dereferencing it, because the linker looks for the unversioned name and the loader looks for the
versioned one. `LIBRARY_PATH` rather than a symlink into `/etc/ld.so.conf.d`, because the host has
no root and there is nothing to run `ldconfig` with.

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
