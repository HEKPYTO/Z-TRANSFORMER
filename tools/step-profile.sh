#!/bin/sh
# Time each op kind in a training step, and require two things of the result.
#
#   sh tools/step-profile.sh <ztransformer-train binary>
#
# A FILE rather than a `sh -c` blob in `build.zig`, for the reason
# `tools/host-clean.sh` and `tools/symbols.sh` are files: a gate has to be
# breakable on purpose, and a gate buried as a string inside a build script can
# only be run by running a build. This one is runnable on its own with `sh -x`.
#
# Two runs at different corpus sizes. Identical per-step arithmetic, three times
# the steps, so the fixed per-RUN cost -- tokenizer, `initParams` -- is amortised
# over three times as many steps and the per-step ops take a larger share.
#
# WHY NOT IN `verify`: these are times. A threshold on a time is a property of
# the machine and the hour. What is checked here is not a number but a SHAPE --
# that the denominator is elapsed time rather than the sum of the buckets, and
# that the shares move when the work does. Both hold on a loaded host as much as
# on an idle one, which is why they can gate and a magnitude cannot.
set -eu

bin=$1
work=$(mktemp -d)

# A trap rather than a line at the end: a failed run under `set -e` skips
# everything after it, and that is exactly when a stray
# `outputs/loss.pending.csv` would be left behind on disk.
cleanup() {
    rm -rf "$work"
    rm -f outputs/loss.pending.csv
}
trap cleanup EXIT

# stderr to a FILE, not into a pipe. Piping it into awk makes the pipeline's
# exit status awk's, so a run that crashed produced an empty table and a
# confusing control-1 failure while the real error sat inside the matcher where
# nothing could see it.
#
# The exit status is deliberately NOT a failure signal either. A short profiled
# run is EXPECTED to be refused: two rows cannot match the committed curve's
# five-row digest (123 is that curve's STEP count, not its row count), so
# `settleCsv` exits 1 with `error.CurveShape`. What is under
# test is whether the profiler printed a table, and that is the predicate.
profile() {
    ZTRANSFORMER_PROFILE=1 ZTRANSFORMER_CORPUS_BYTES="$1" "$bin" train \
        > /dev/null 2> "$2.raw" || true
    grep -q '^op ' "$2.raw" || {
        echo "step-profile: no profiler table at $1 bytes. The run said:" >&2
        tail -6 "$2.raw" >&2
        exit 1
    }
    awk '/^[a-z_]+ +[0-9]+ +[0-9]+ +[0-9.]+%$/ {gsub(/%/,"",$NF); print $1, $NF}' "$2.raw" > "$2.shares"
    awk '/^SUM[ ]+[0-9]+/ {print $2; exit}' "$2.raw" > "$2.sum"
    awk '/^SPAN[ ]+[0-9]+/ {print $2; exit}' "$2.raw" > "$2.span"
}

profile 16384 "$work/a"
profile 49152 "$work/b"

echo "=== step-profile, 30 windows ==="
awk '{printf "  %-10s %s%%\n", $1, $2}' "$work/a.shares"
echo "=== step-profile, 120 windows ==="
awk '{printf "  %-10s %s%%\n", $1, $2}' "$work/b.shares"

# ---------------------------------------------------------------------------
# CONTROL 1: SPAN > SUM, compared as the two integers the table prints.
#
# This is the entire claim -- the denominator is elapsed time and not the sum of
# the buckets -- and as integers it needs no tolerance and cannot pass on a
# knife edge. A profiler dividing by its own buckets prints SPAN == SUM exactly.
# An earlier float version of this check had to distinguish 99.996% from 100%
# and rested on a 0.004-point margin to do it, which broke whenever a refactor
# banked the last of the teardown.
# ---------------------------------------------------------------------------
sum_ns=$(cat "$work/a.sum")
span_ns=$(cat "$work/a.span")
echo "SUM ${sum_ns} ns of buckets, SPAN ${span_ns} ns elapsed"

if [ -z "$sum_ns" ] || [ -z "$span_ns" ]; then
    echo "step-profile: the table printed no SUM or SPAN row, so there is nothing" >&2
    echo "  to check and this run would have passed on an empty table." >&2
    exit 1
fi

# KNOWN WEAKNESS, recorded here because a gate's weakness that lives only in a
# conversation disappears with the conversation.
#
# This check is close to an arithmetic IDENTITY. The probes partition
# [first probe, finish], so `bucketSum <= span` holds for ANY placement, ANY
# count and ANY naming of the probes, and the only thing that can fail it is
# `denominatorNs` being changed to return `bucketSumNs`. **It cannot detect a
# mis-placed, mis-named or missing probe.**
#
# That is not hypothetical: the `.fetch` row once read 0.310% of a step while
# measuring the previous step's four `defer`s plus the row append, because
# Zig runs `defer` at the scope's closing brace, after the last op and before
# the next one. Every check above passed throughout. The rename to `loop` is
# what actually caught it, and it was a person reading the loop, not a gate.
#
# The check below is the one that bites: it pins the two rows whose relationship
# the defer-ordering bug inverted. `loop` must carry real time (it is the frees,
# the counters and the row append) and `fetch` must be small, because the batch
# fetch is a shuffle and nothing more.

if [ "$span_ns" -le "$sum_ns" ]; then
    echo "step-profile: SPAN (${span_ns}) is not greater than SUM (${sum_ns})." >&2
    echo "  That means the denominator is the sum of the buckets, so every share" >&2
    echo "  sums to 100% by construction and this check would be reporting" >&2
    echo "  completeness without having measured anything." >&2
    exit 1
fi

# Coverage, so an empty or degenerate table cannot satisfy the check above. Two
# ops carrying time, and no single op reaching the whole step on its own.
nonzero=$(awk '$2 > 0' "$work/a.shares" | wc -l | tr -d ' ')
top=$(awk 'BEGIN{m=0} {gsub(/%/,"",$2); if ($2+0 > m) m = $2+0} END{printf "%.3f", m}' "$work/a.shares")
if [ "$nonzero" -lt 2 ]; then
    echo "step-profile: only ${nonzero} op carried time; the probes are not in the loop." >&2
    exit 1
fi
if ! awk -v t="$top" 'BEGIN { exit !(t < 100.0) }'; then
    echo "step-profile: one op is ${top}% of the step, so the rest is noise." >&2
    exit 1
fi

# PLACEMENT CONTROL -- the check that actually catches a mis-placed probe, and the
# one whose absence let the defer-ordering bug survive every check above.
# `loop` is the frees, the counters and the row append, so it must carry real
# time; `fetch` is the batch shuffle, so it must be small. A probe that slides
# across the loop boundary swaps those two, and nothing else here notices.
loop_ns=$(awk '$1 == "loop" {print $2; exit}' "$work/a.raw")
fetch_ns=$(awk '$1 == "fetch" {print $2; exit}' "$work/a.raw")
if [ -z "$loop_ns" ] || [ -z "$fetch_ns" ]; then
    echo "step-profile: the table has no loop or fetch row, so probe placement" >&2
    echo "  cannot be checked at all." >&2
    exit 1
fi
if [ "$loop_ns" -le 0 ]; then
    echo "step-profile: the loop row carried no time (${loop_ns} ns)." >&2
    echo "  It is where the defers land, so a probe has moved." >&2
    exit 1
fi
if [ "$fetch_ns" -ge "$loop_ns" ]; then
    echo "step-profile: fetch (${fetch_ns} ns) is not smaller than loop (${loop_ns} ns)." >&2
    echo "  The batch fetch is a shuffle; anything that large there is the" >&2
    echo "  previous step's teardown landing under the wrong name." >&2
    exit 1
fi
echo "placement: loop ${loop_ns} ns, fetch ${fetch_ns} ns"

# ---------------------------------------------------------------------------
# CONTROL 2: the shares MOVE when the work does.
#
# `backward` moves about 77% to 82% and `forward` 19% to 15%. `eval` moves 0.01
# points and is NOT the mechanism here, whatever it resembles.
#
# 0.5 was chosen UNDER the ~5 points the arithmetic predicts, because run-to-run
# at a fixed size is tight (backward read 77.138 / 77.143 / 77.207 / 77.241
# across four runs) and a threshold set at the predicted signal would be decided
# by noise rather than by the check.
# ---------------------------------------------------------------------------
moved=$(awk 'NR == FNR { a[$1] = $2; next }
             { d = $2 - a[$1]; if (d < 0) d = -d; if (d > m) m = d }
             END { printf "%.4f", m }' "$work/a.shares" "$work/b.shares")
echo "largest share movement between the two: ${moved} points"

if ! awk -v m="$moved" 'BEGIN { exit !(m > 0.5) }'; then
    echo "step-profile: the shares barely moved (${moved} points) between a" >&2
    echo "  30-window and a 120-window run whose per-step arithmetic is" >&2
    echo "  identical. A table that does not respond to the work is not" >&2
    echo "  measuring the work." >&2
    exit 1
fi