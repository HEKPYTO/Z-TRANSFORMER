#!/bin/sh
# Check src/cuda/norm.cu against the CPU twin in src/norm.zig, on the local
# NVIDIA GPU, and benchmark both.
#
#   sh src/cuda/run-norm.sh          run it
#
# Two halves and no build target. The CPU half is src/cuda/norm_twin.zig, run by
# the host's zig as a root module with src/norm.zig brought in as a second
# module, because a module root in Zig 0.16 cannot import a file outside its own
# directory. The GPU half is norm.cu, compiled and run by nvcc inside the
# container cuda.sh describes. They meet on flat little-endian f32 blobs in the
# scratch directory, which is why the GPU process needs no Zig and the Zig
# process needs no CUDA: the shape table is in the twin and norm.cu reads it as a
# manifest rather than hardcoding shapes of its own.
#
# Everything lands under .zig-cache/cuda/norm, which is gitignored, and all of it
# is removed on every exit path including a failing one. Nothing is left on the
# host, nothing is written outside the repository, and no container, cache or
# object survives the run.

set -eu

. "$(dirname "$0")/cuda.sh"

ROOT=$CUDA_ROOT_DIR
SCRATCH=$ROOT/.zig-cache/cuda/norm
TWIN=$SCRATCH/norm_twin
OBJ=$SCRATCH/norm.o
BIN=$SCRATCH/norm

cleanup() {
    rm -rf "$SCRATCH"
}
trap cleanup EXIT

mkdir -p "$SCRATCH"

echo "norm: scratch $SCRATCH"

# ReleaseFast on both modules. A Debug build leaves the two inner loops
# unoptimised, and a benchmark run in Debug is a measurement of the build mode
# rather than of the operation, so it would flatter the GPU by an unknown factor
# on the CPU side and flatter nothing at all on the device side.
#
# The caches are pointed into the scratch directory rather than at the default
# under the home directory, so the run leaves nothing in ~/.cache and cleanup is
# a single rm -rf rather than a judgement about what was there before.
zig build-exe -OReleaseFast -OReleaseFast \
    --dep ztransformer \
    -Mroot=src/cuda/norm_twin.zig \
    -Mztransformer=src/norm.zig \
    --cache-dir "$SCRATCH/zig-cache" \
    --global-cache-dir "$SCRATCH/zig-global" \
    -femit-bin="$TWIN"
echo "norm: built the CPU twin"

# Run outside the container. src/norm.zig needs no CUDA, and putting it in the
# container would only buy a second copy of zig for no reason.
"$TWIN" "$SCRATCH"
echo "norm: wrote inputs and the reference output"

cuda_pull

NVCC_FLAGS=$(cuda_nvcc_flags)
cuda "nvcc $NVCC_FLAGS -c -o '$OBJ' src/cuda/norm.cu"
echo "norm: compiled norm.cu"
cuda "nvcc -arch=$CUDA_ARCH '$OBJ' -o '$BIN'"

# The exit code is this script's exit code, deliberately. norm.cu exits non-zero
# when any shape misses the parity gate or when either deliberately broken
# kernel slips past it, and a harness whose failure is visible only in its
# middle is a harness that gets piped to tail.
cuda "'$BIN' '$SCRATCH'"
