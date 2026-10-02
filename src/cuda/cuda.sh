# Shared container plumbing for the scripts in this directory. Sourced, not run.
#
#   . src/cuda/cuda.sh
#
# Why a container at all: because it is the toolchain the benchmark table in
# src/cuda/README.md was measured with, not because of anything this host has
# or lacks. An earlier version of this header said nvcc was not installed on
# the host and could not be, which was false -- the host does carry a working
# CUDA 13.4 -- and was true only of the machine it happened to be written on.
# The image below is the toolchain, and it is pinned rather than "latest" because
# "latest" moves under a published command and a script whose toolchain silently
# changes is not evidence of anything.
#
# Every script here needs the same four things and got them wrong the same way
# once, so they live here instead of once per script.

# CUDA_IMAGE is the image tag. CUDA_ROOT_DIR is the repository root.
#
# The container bind-mounts the same path at the same path. A build that compiles
# at one path and links at another bakes the container's path into the object and
# the host then fails to resolve it.
#
# CUDA_ROOT_DIR is derived from `$0`, and that is only a path in this repository
# when the caller is one of the two scripts here. A sourced file sees the
# CALLER's `$0`, so `zig build cuda-check` -- which sources this from `sh -c`,
# where `$0` is `sh` -- resolved the root to the parent of the repository and the
# container bind-mounted a directory that is not this one. cc1plus reported
# `No such file or directory` for a source that was plainly sitting there. So the
# derivation is a default rather than a fact, and a caller whose `$0` is not a
# path in the repository sets it.
CUDA_IMAGE=nvidia/cuda:12.6.3-devel-ubuntu22.04
CUDA_ROOT_DIR=${CUDA_ROOT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}

# The architecture comes from the device rather than being written down, because
# the difference between sm_86 and sm_89 is one character and produces
# cudaErrorNoKernelImageForDevice at launch, which reads like a broken GPU.
#
# It also has to fail here rather than downstream, and that is not free. This was
# one pipeline ending in `sed`, so its exit status was `sed`'s -- and `sed`
# succeeds on empty input. On a host with no `nvidia-smi`, or with one and no
# driver, the substitution turned nothing into the literal `sm_`, the script
# carried on, and `nvcc` failed much later with an arch complaint that named
# neither this line nor the real cause. `set -e` cannot catch it: a command
# substitution reports the status of its last pipeline element, not its own.
#
# So the emptiness is tested, and the two cases are told apart rather than
# merged. Absent `nvidia-smi` and present-but-failing are different problems and
# get different instructions.
#
# An exported CUDA_ARCH short-circuits the derivation, and that is the only way
# to reach the compiler on a host with a container runtime and no GPU: `-arch`
# is a compile-time argument, so a compile-only check has nothing to derive from
# and has to be told. `zig build cuda-check` is that caller.
if [ -n "${CUDA_ARCH:-}" ]; then
    case "$CUDA_ARCH" in
        sm_[0-9][0-9]*) ;;
        *)
            echo "cuda: CUDA_ARCH is set to '$CUDA_ARCH', which is not an sm_NN" >&2
            echo "  architecture. It looks like a compute capability such as 8.6," >&2
            echo "  which wants the sm_ prefix this variable carries." >&2
            return 1 2>/dev/null || exit 1
            ;;
    esac
elif ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "cuda: nvidia-smi is not on this host, so the device architecture cannot" >&2
    echo "  be derived and the CUDA sources cannot be compiled for this machine." >&2
    echo "  This script needs an NVIDIA GPU; there is nothing to check here." >&2
    echo "  To compile without one, set CUDA_ARCH, e.g. CUDA_ARCH=sm_86." >&2
    return 1 2>/dev/null || exit 1
else
    _cuda_cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '[:space:]')
    if [ -z "$_cuda_cap" ] || [ "$_cuda_cap" = "[N/A]" ]; then
        echo "cuda: nvidia-smi is present but reported no compute capability for any" >&2
        echo "  device, which is what an absent or mismatched driver looks like:" >&2
        echo "    $CUDA_IMAGE needs a host driver that exposes the device." >&2
        nvidia-smi 2>&1 | head -3 | sed 's/^/    /' >&2
        return 1 2>/dev/null || exit 1
    fi
    CUDA_ARCH=$(printf '%s' "$_cuda_cap" | tr -d '.' | sed 's/^/sm_/')
    case "$CUDA_ARCH" in
        sm_[0-9][0-9]*) ;;
        *)
            # `$_cuda_cap` is still set here on purpose. This branch exists to name a
            # capability it could not read, and both callers run under `set -u`, so
            # unsetting the variable above this point turned the diagnostic into an
            # unbound-variable abort -- the message written to explain the failure was
            # itself the failure. The unset moved below the case for that reason.
            echo "cuda: could not read a compute capability out of '$_cuda_cap'." >&2
            echo "  Expected something like 8.6." >&2
            return 1 2>/dev/null || exit 1
            ;;
    esac
    unset _cuda_cap
fi

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

# Pulls the image if it is absent, and says so, because the pull is large and a
# silent multi-minute pause in a script is indistinguishable from a hang. The
# figure is 11.4 GB, measured with `docker images` on the host this was run on.
#
# The image is left in place on purpose. `docker run --rm` removes the container,
# not the image, and removing it would make every run re-download 11.4 GB. It is
# a pinned toolchain, not build output: `docker image rm` it by hand if a
# pristine host matters more than the next run being fast.
cuda_pull() {
    if ! docker image inspect "$CUDA_IMAGE" >/dev/null 2>&1; then
        echo "pulling $CUDA_IMAGE, about 11.4 GB, first run only"
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
