# src/cuda

Five files. `norm.cu` is the first kernel this project ships: RMSNorm on the GPU, checked against the
CPU twin in `src/norm.zig`. `norm_twin.zig` is the CPU half of that check. `run-norm.sh` runs both.
`probe.cu` is the older toolchain probe and `run-probe.sh` runs it.

| File | What it is |
|---|---|
| `norm.cu` | The kernel, the parity gate, and the benchmark. All of it, no CPU reference. |
| `norm_twin.zig` | Generates inputs, runs `norm.forward`, writes the reference output and its own timings. |
| `run-norm.sh` | Compiles and runs both halves and exits non-zero on a failed check. |
| `cuda.sh` | The container, image pin and nvcc flags, shared by the two runners. |
| `probe.cu` | Toolchain probe. Nothing here is a transformer operation except `norm.cu`. |

## Run it

| Command | What it does |
|---|---|
| `sh src/cuda/run-norm.sh` | Pulls the image if absent, builds the twin, generates 18 shapes, checks parity, proves the gate can fail, benchmarks, removes everything. |
| `sh src/cuda/run-probe.sh` | The toolchain probe. Still passes after `cuda.sh` was extracted. |
| `sh src/cuda/run-probe.sh --emit-object` | Compiles `probe.cu` to `.zig-cache/cuda/probe.o` and stops. |

Measured on an RTX 3080 Ti, CUDA 12.6.3, driver 13040, against a Fedora host, on
`2026-09-29`. Both halves ran on the same machine, so neither side is measured against a different
processor.

```
$ sh src/cuda/run-norm.sh
norm: scratch the project directory on that host/.zig-cache/cuda/norm
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
add, the square root and the narrowing back to `f32` all happen in `f64`, exactly as
`src/norm.zig:24` does. It is free: 4096 rows ask for 4096 double divisions and square roots
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
micro        32x32             1024       3.44       2.38       3.70       8.09      17.07     0.6x
small        32x128            4096      14.13       9.96       3.37       7.65      19.40     3.0x
mid          128x128          16384      83.79      39.56       4.73       9.34      35.91     8.4x
ship         256x128          32768     162.88      78.95       7.72      12.23      55.87    10.2x
tall         4096x128        524288    2600.90    1299.00      83.51      83.38     672.00    15.6x
r1024c512    1024x512        524288    2642.97    1336.26      21.63      25.57     528.03    61.8x
r256c2048    256x2048        524288    2727.09    1426.18       8.67      12.56     514.61   164.5x
r1024c2048   1024x2048      2097152   11272.11    5623.92      28.79      32.75    2158.18   195.4x
big          4096x4096     16777216   91628.05   43564.09     228.07     231.06   21826.60   191.0x
```

A rerun does not print these bytes again, and the reason is on the next line: run to run the small
shapes move by up to about 15 percent, which is the fixed launch cost moving around; the large shape
is repeatable to about 2 percent. So this table is a record of one run, the ratios are what to read,
and every figure quoted below is a subtraction or a ratio of two cells in it.

### The two numbers that were asked for

**Shipped model shape, `256x128`**, which is `d_model 128` at a full context window of 256. The
kernel on its own is `7.72 us`; `norm.zig`'s arithmetic is `78.95 us`. On that measure the GPU is
`10.2x` faster, and that is the two cells in the `ship` row divided.

**Large shape, `4096x4096`.** The kernel is `228 us`; `norm.zig`'s arithmetic is `43.6 ms`. The GPU
is `191x` faster, and here the number means what it looks like: `4096 * 4096 * 4` bytes in and the
same out is 128 MiB, and 128 MiB of traffic at roughly 900 GB/s is about 220 us, which is what the
kernel does.

### The crossover, and why the shipped-shape number is not the interesting one

The crossover is between `1024` elements, where the kernel loses (`3.70 us` against `2.38 us`), and
`4096` elements, where it wins (`3.37 us` against `9.96 us`). So it is a few thousand elements, which
is far below anything this model builds.

That floor is the whole story at the shipped size, and it is why the `10.2x` should not be read as
"this model would be faster on a GPU":

- The kernel takes `3.70 us` at 1024 elements, `3.37 us` at 4096, `4.73 us` at 16384 and `7.72 us` at
  32768. From 1024 to 32768 elements, thirty-two times the work, and the time only doubles. Almost
  all of it is launch and block scheduling. The actual work at `256x128` is 256 KiB of traffic, about
  `0.3 us` at peak bandwidth.
- The CPU side is slow for a reason that is specific to `norm.zig` rather than to RMSNorm. `78.95
  us` over `32768` elements is `2.4 ns` per element, and the reason is readable in the source: it is
  a serial `f64` accumulation, so each row is one dependency chain of 128 `f64` adds with no ILP to
  fill it. That is a correct and deliberate choice for the CPU implementation, and this kernel is
  not evidence that it was the wrong one.

The number that actually governs a decision is `gpu_e2e`, and at the shipped shape it is `55.87 us`
against a CPU call of `162.88 us`. This repository has no device-resident tensor type yet, so every
call copies 128 KiB in and 128 KiB out over PCIe, and that transfer is all but the `7.72 us` of
kernel: `55.87 - 7.72` is six tenths of the `78.95 us` the CPU spends on the same arithmetic. With
4 layers and 2 norms per layer, one forward pass spends `8 * 55.87`, about 450 us, copying RMSNorm
inputs before any arithmetic happens.

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

`tall` at `4096x128` is the same element count as `r256c2048` and takes `83.51 us` against
`8.67 us`, about 10x worse, with no more memory traffic. The cause is the fixed block size: at
`cols = 128` only 128 of the 256 threads in a block have an element to read, so half the block idles,
and 4096 blocks each pay the full per-block cost for one element. Widening the row hides it and
narrowing it exposes it.

`ship` is `256x128` and costs `7.72 us`, so the shipped model is on the right side of this for its
row count, but a batch large enough to make 4096 rows out of 128-wide rows would land in the slow
case. The fix is to dispatch on a block size that fits the row, which needs the warp count to be a
template parameter rather than the `WARPS` constant it is now. Not done: it is an optimisation, not
a correctness fix, and the shapes this model actually builds are measured above.

## The toolchain

`nvcc` is not installed on the host and cannot be. The distribution's NVIDIA repository ships a CUDA
newer than this repository targets, with a cuBLAS version-skewed against it, and there is no root.
The toolchain is the container, and `cuda.sh` is the whole recipe. Pull it once, about 3 GB:

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
LazyPath)` (`std/Build/Module.zig:460`) and `Module.linkSystemLibrary("cudart", .{})`
(`std/Build/Module.zig:363`), which takes three arguments.
`Step.Run.addOutputFileArg(basename) LazyPath` (`std/Build/Step/Run.zig:279`) returns exactly the
`LazyPath` that `addObjectFile` wants, so an nvcc step and a link can be wired together without a
temporary path or a file read back off disk.

Nothing here uses it. The shell script is the interface, `build.zig` does not reference this
directory, and `run-probe.sh --emit-object` is what a future `zig build` edge would take its input
from.

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

`norm_twin.zig` is not a build target and `build.zig` does not mention it. It runs the way
`tools/train_bpe.zig` does, as a root module with a `src/` file brought in as a second module,
because a module root in Zig 0.16 may not import a file outside its own directory and
`zig run src/cuda/norm_twin.zig` cannot reach `../norm.zig` at all:

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

`run-norm.sh` writes everything under `.zig-cache/cuda/norm`, which is gitignored, and removes it on
every exit path including a failing one. Both zig caches are redirected there for the same reason. It
leaves no container, no cache and no object behind, writes nothing outside the repository, and
commits nothing.
