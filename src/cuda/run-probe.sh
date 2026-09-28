#!/bin/sh
# Compile src/cuda/probe.cu with nvcc and run it on the local NVIDIA GPU.
#
# nvcc is not installed on the host and cannot be: the distribution's NVIDIA
# repository ships a CUDA newer than this repository targets, with a cuBLAS that
# is version-skewed against it. The toolchain is therefore the container, and
# this script is the whole recipe. It needs no root and writes nothing outside
# the repository.
#
#   sh src/cuda/run-probe.sh                 compile, run, print the evidence, leave nothing
#   sh src/cuda/run-probe.sh --emit-object   compile only, leave .zig-cache/cuda/probe.o
#
# --emit-object exists for the future `zig build` edge, which links a .o built
# by nvcc into the Zig binary. It is the one mode that leaves a file behind, and
# what it leaves is inside .zig-cache/, which is already gitignored and already
# disposable.

set -eu

# The container invocation, the image pin and the nvcc flags are shared with
# run-norm.sh. They are here rather than written twice because they were got
# wrong once, in this file, and a second copy is a second chance at it.
. "$(dirname "$0")/cuda.sh"

ROOT=$CUDA_ROOT_DIR
OUT=$ROOT/.zig-cache/cuda
OBJ=$OUT/probe.o
BIN=$OUT/probe

mode=run
case "${1:-}" in
--emit-object) mode=object ;;
"") ;;
*)
    echo "usage: sh src/cuda/run-probe.sh [--emit-object]" >&2
    exit 2
    ;;
esac

# One container, one command, run as the person who typed this, plus the image
# pin and the nvcc flags: all four are in cuda.sh. run-norm.sh sources the same
# file, so there is one copy of the recipe rather than two that drift.
cuda_pull

mkdir -p "$OUT"

# The nvcc flags, and the reasoning behind each of them, live in cuda.sh and are
# shared with run-norm.sh rather than restated here.
NVCC_FLAGS=$(cuda_nvcc_flags)
cuda "nvcc $NVCC_FLAGS -c -o '$OBJ' src/cuda/probe.cu"

if [ "$mode" = object ]; then
    echo "run-probe: wrote $OBJ"
    exit 0
fi

# Removed on every exit path from here on, not only the successful one, so a
# failed link or a failed run leaves the host as clean as a passed one. It is
# also why the expected sum is compared inside probe.cu rather than here: this
# script cannot tell a passing probe from a skipped one, and a check that can be
# skipped is not a check. A reader who wants the binary to look at has
# --emit-object.
trap 'rm -f "$OBJ" "$BIN"' EXIT

# Link the object that was just built, rather than recompiling the source. That
# makes the link step a real check: a .o that cannot be linked proves less about
# the toolchain than one that can, and it is the same object the Zig edge takes.
cuda "nvcc -arch=$CUDA_ARCH '$OBJ' -o '$BIN'"
cuda "'$BIN'"
