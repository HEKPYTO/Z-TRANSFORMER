# Shared container plumbing for the scripts in this directory. Sourced, not run.
#
#   . src/cuda/cuda.sh
#
# Why a container at all: nvcc is not installed on the host and cannot be. The
# distribution's NVIDIA repository ships a CUDA newer than this repository
# targets, with a cuBLAS that is version-skewed against it, and there is no root.
# The image below is the toolchain, and it is pinned rather than "latest" because
# "latest" moves under a published command and a script whose toolchain silently
# changes is not evidence of anything.
#
# Every script here needs the same four things and got them wrong the same way
# once, so they live here instead of once per script.

# CUDA_IMAGE is the image tag. CUDA_ROOT_DIR is the repository root, resolved
# from the sourcing script's own location so a script can be run from anywhere.
#
# The container bind-mounts the same path at the same path. A build that compiles
# at one path and links at another bakes the container's path into the object and
# the host then fails to resolve it.
CUDA_IMAGE=nvidia/cuda:12.6.3-devel-ubuntu22.04
CUDA_ROOT_DIR=$(cd "$(dirname "$0")/../.." && pwd)
CUDA_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ' | tr -d '.' | sed 's/^/sm_/')

# One container, one command, run as the person who typed this.
#
# --user is the fix for the bind mount: without it every file the container
# writes is owned by uid 0, and the next host-side build cannot overwrite it.
# That is not hypothetical, it is what the first run of run-probe.sh did.
#
# --entrypoint bypasses NVIDIA's own entrypoint script, which runs `ldconfig`
# and needs root to write /etc/ld.so.cache. It would fail before nvcc ever ran,
# and it would fail under --user, which is exactly the combination these scripts
# need. The GPU is reachable regardless: --gpus all hands the container the
# driver library and the device nodes directly.
#
# HOME is set because --user leaves the process with no passwd entry, and nvcc
# wants somewhere writable for its own cache. /tmp is inside the container and
# dies with it.
cuda() {
    docker run --rm --gpus all \
        --user "$(id -u):$(id -g)" \
        --entrypoint /bin/sh \
        -e HOME=/tmp \
        -v "$CUDA_ROOT_DIR:$CUDA_ROOT_DIR" -w "$CUDA_ROOT_DIR" \
        "$CUDA_IMAGE" -c "$1"
}

# Pulls the image if it is absent, and says so, because the pull is 3 GB and a
# silent three-minute pause in a script is indistinguishable from a hang.
cuda_pull() {
    if ! docker image inspect "$CUDA_IMAGE" >/dev/null 2>&1; then
        echo "pulling $CUDA_IMAGE, about 3 GB, first run only"
        docker pull "$CUDA_IMAGE"
    fi
}

# The nvcc flags shared by every compile here.
#
# -Werror all-warnings and -Xcompiler -Werror, because the Zig build treats a
# warning as an error and a CUDA file that is lax about it is the one nobody
# looks at until it is the only thing broken. Verified to bite rather than
# merely to be present: an unused variable in a scratch .cu comes back as
# `error #177-D` rather than as a warning, and the image is pinned, so this
# cannot start failing on a toolchain nobody chose.
#
# -Xcompiler -fPIC because the object is destined for a Zig-linked binary and
# the Zig build links a position-independent executable by default on Linux.
# Without it the link fails with a relocation error against code compiled for a
# non-PIE address space, which names neither -fPIC nor the file at fault.
#
# -arch comes from the device rather than being written down, because the
# difference between sm_86 and sm_89 is one character and produces
# cudaErrorNoKernelImageForDevice at launch, which reads like a broken GPU.
cuda_nvcc_flags() {
    echo "-arch=$CUDA_ARCH -Werror all-warnings -Xcompiler -Werror -Xcompiler -fPIC"
}
