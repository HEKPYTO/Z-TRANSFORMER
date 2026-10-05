#!/bin/sh
# Build the fused attention kernel and grade it against the CPU implementation it
# would replace. The three stages are deliberately in this order and each one can
# fail the script on its own:
#
#   1. attn_twin.zig runs on the host, outside the container, because
#      src/attention.zig needs no CUDA. It writes the inputs, the reference
#      output and a manifest into the scratch directory.
#   2. attn.cu is compiled inside the pinned container, so the kernel is built by
#      the same toolchain the RMSNorm benchmark was measured with.
#   3. the binary is run inside the same container, and its exit code is this
#      script's exit code. attn.cu exits non-zero when any context length misses
#      the parity gate, and a harness whose failure is visible only in its middle
#      is a harness that gets piped to tail and believed.
#
# Everything lands under .zig-cache/cuda/attn, which is gitignored, and all of it
# is removed on every exit path including a failing one. Nothing is written
# outside the repository. The toolchain it compiles with is the host's pinned
# container image, installed once and not touched by any of this; nothing is
# fetched beyond that image on its first run.

set -eu
. "$(dirname "$0")/cuda.sh"

ROOT=$CUDA_ROOT_DIR
SCRATCH=$ROOT/.zig-cache/cuda/attn
TWIN=$SCRATCH/attn_twin
OBJ=$SCRATCH/attn.o
BIN=$SCRATCH/attn

# This script prints a speedup table, and a speedup is a ratio whose denominator
# is one CPU call. So refuse to produce one on a host that would contaminate it,
# naming the condition rather than quietly recording the contention as a result.
#
# The escape hatch is for the half of this run that is not a measurement at all:
# parity against the CPU twin is exact arithmetic and does not care what else is
# running. Someone who wants only that should not have to wait for the card.
if ! sh "$ROOT/tools/host-clean.sh"; then
    if [ "${ATTN_ALLOW_DIRTY_HOST:-0}" = "1" ]; then
        echo "attn: ATTN_ALLOW_DIRTY_HOST=1, so continuing on a host this" >&2
        echo "      repository would refuse. Every timing in the output below is" >&2
        echo "      contaminated and the ratio beside it measures the contention." >&2
    else
        echo "attn: refusing to run. Re-run when the host is clean, or set" >&2
        echo "      ATTN_ALLOW_DIRTY_HOST=1 if you want the parity result only." >&2
        exit 1
    fi
fi

cleanup() {
    rm -rf "$SCRATCH"
}
trap cleanup EXIT

mkdir -p "$SCRATCH"

echo "attn: scratch $SCRATCH"

# ReleaseFast for the same reason run-norm.sh gives: a benchmark run in Debug is
# a measurement of the build mode rather than of the operation, and here it would
# flatter the GPU side by nothing at all while misrepresenting the CPU side.
#
# The caches are pointed into the scratch directory so the run leaves nothing
# under ~/.cache and cleanup stays a single rm -rf.
# `--dep` binds to the -M that FOLLOWS it, so the dependency is declared before
# -Mroot. Written the other way round it attaches to -Mautograd, the root imports
# nothing, and zig says "module 'autograd' declared but not used".
#
# One module only, and it is autograd rather than attention. The twin needs
# `attention.forward` for the forward reference and `attentionBackward` for the
# backward one; passing both files as separate modules compiles every symbol they
# share twice. autograd.zig owns model, tensor, norm, rope, mlp and attention
# through its own relative imports and re-exports `attention` and `model` for
# exactly this caller.
zig build-exe -OReleaseFast \
    --dep autograd \
    -Mroot=src/cuda/attn_twin.zig \
    -Mautograd=src/autograd.zig \
    --cache-dir "$SCRATCH/zig-cache" \
    --global-cache-dir "$SCRATCH/zig-global" \
    -femit-bin="$TWIN"
echo "attn: built the CPU twin"

# Host, no container: src/attention.zig is plain Zig.
"$TWIN" "$SCRATCH"
echo "attn: wrote inputs, the reference output and the manifest"

cuda_pull

NVCC_FLAGS=$(cuda_nvcc_flags)
cuda "nvcc $NVCC_FLAGS -c -o '$OBJ' src/cuda/attn.cu"
echo "attn: compiled attn.cu"
cuda "nvcc -arch=$CUDA_ARCH '$OBJ' -o '$BIN'"

# ATTN_GROUP_Q is set HERE, inside the container's own command string, and not
# inherited from the environment. `docker run` does not pass host environment
# variables into the container unless they are named with -e, so setting it on the
# host is a silent no-op: the binary reads its default and the A/B compares one
# configuration with itself. Six such runs looked like a plausible result, which is
# the whole danger -- ATTN_BROKEN below works precisely because it is set the same
# way, inside the string.
ATTN_GROUP_Q=${ATTN_GROUP_Q:-1}
echo "attn: running with ATTN_GROUP_Q=$ATTN_GROUP_Q"
cuda "ATTN_GROUP_Q=$ATTN_GROUP_Q '$BIN' '$SCRATCH'"

# ATTN_MAX_TILE is the tile cap, the one launch parameter that also decides whether
# a head WIDTH runs at all: at head_dim 256 a cap of 64 does not fit a block, and
# the launcher narrows the tile on its own. Setting it HERE, inside the container's
# own command string, for the reason ATTN_GROUP_Q's paragraph gives -- `docker run`
# drops host environment variables that are not named with -e, so exporting this
# one on the host and reading a table off the result is the same silent no-op that
# produced six runs of the default once. Its floor of 1 is the smallest workable
# tile and is the way to grade the claim that a tile of 1 is correct, which is why
# it is reachable rather than argued.
#
# The broken-variant loops below do NOT take it. They run at the default cap so
# their signatures stay comparable with the ones src/cuda/README.md publishes.
ATTN_MAX_TILE=${ATTN_MAX_TILE:-64}
echo "attn: running with ATTN_MAX_TILE=$ATTN_MAX_TILE"
cuda "ATTN_MAX_TILE=$ATTN_MAX_TILE '$BIN' '$SCRATCH'"

# A gate that has never been seen to fail is not known to be a gate. Each
# variant below is a real defect, and each must make the parity check exit
# non-zero. If one of them passes, the check is not looking at what it claims
# and this script fails rather than reporting a clean run.
echo "attn: proving the parity gate can fail"
for broken in 1 2 3 4; do
    if cuda "ATTN_BROKEN=$broken '$BIN' '$SCRATCH'" > "$SCRATCH/broken-$broken.log" 2>&1; then
        echo "attn: FAIL broken variant $broken passed the parity gate, so the gate" >&2
        echo "      is not exercising what it claims to check." >&2
        # Copied out before the EXIT trap removes the scratch directory, because a
        # failure that destroys its own evidence is a failure nobody can act on.
        cat "$SCRATCH/broken-$broken.log" >&2
        exit 1
    fi
    echo "attn: broken variant $broken was caught, as it must be"
done

# The backward's variants, and they are a SEPARATE loop over a SEPARATE variable
# because they are different defects: 1 walks the prefix instead of the suffix in
# the KV-outer kernel, 2 collapses GQA, 3 drops the softmax Jacobian (the one
# thing the backward does that the forward does not), 4 drops the scale. A single
# knob driving both would make "variant 3 was caught" mean two different things in
# one run, and the point of this loop is that each name means one defect.
echo "attn: proving the backward gate can fail too"
# A variant that fails is necessary but NOT sufficient. While the correct kernel
# was itself failing, all four of these failed too -- vacuously -- and the script
# printed "caught, as it must be" four times while proving nothing. So each
# variant's SIGNATURE is captured and the four are required to differ. Four
# distinct defects must leave four distinct traces, and a set of four identical
# readings means the loop is comparing one thing with itself.
#
# The signatures, at ctx256, are not arbitrary and they are the reason this check
# is worth having. Variant 3 drops the softmax Jacobian, which appears in p_ds and
# therefore in dq and dk but never in dv -- so dv reads BIT-IDENTICAL to the
# correct kernel at 2.384e-07 while dq and dk read 5.2e-02 and 2.1e-01. A gate
# that only watched dv would see variant 3 pass.
SIGS=""
for broken in 1 2 3 4; do
    if cuda "ATTN_BWD_BROKEN=$broken '$BIN' '$SCRATCH'" > "$SCRATCH/bwd-broken-$broken.log" 2>&1; then
        echo "attn: FAIL backward variant $broken passed the backward gate, so the gate" >&2
        echo "      is not exercising what it claims to check." >&2
        cat "$SCRATCH/bwd-broken-$broken.log" >&2
        exit 1
    fi
    sig=$(sed -n '/max_abs_dq/,+1p' "$SCRATCH/bwd-broken-$broken.log" | sed -n '2p' | awk '{print $4"/"$5"/"$6}')
    echo "attn: backward variant $broken was caught, as it must be   [dq/dk/dv $sig]"
    SIGS="$SIGS $sig"
done
distinct=$(printf '%s\n' $SIGS | sort -u | wc -l | tr -d ' ')
if [ "$distinct" -ne 4 ]; then
    echo "attn: FAIL the four backward variants produced $distinct distinct signatures," >&2
    echo "      expected 4. Identical readings mean the loop is not exercising four" >&2
    echo "      different defects." >&2
    exit 1
fi
echo "attn: ...and all four signatures differ, so they are four different defects"

echo "attn: the backward kernel exists and this harness grades it against"
echo "      attentionBackward, but it is NOT on the training path: src/model.zig's"
echo "      cuda_attn routes the FORWARD through this file, and nothing in the Zig"
echo "      tree calls device.Attn.backward, so the backward kernel is exercised"
echo "      only here. A KV cache exists (src/kv_cache.zig, six tests) and"
echo "      src/decode.zig decodes through it: decode.cudaAttnStep derives q_offset ="
echo "      pos from the cache itself, so the forward kernel's offset is exercised at"
echo "      every real decode position and not only at 0."