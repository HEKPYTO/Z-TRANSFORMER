#!/bin/sh
# Fail when this host is in a state that would contaminate a timing.
#
#   sh tools/host-clean.sh
#
# A benchmark in this repository is a ratio, and the denominator of every one of
# them is a CPU call or a PCIe floor. Those are properties of the machine and
# the hour, not of the code, so a measurement taken while something else holds
# the machine is a measurement of the something else. This says so before a run
# rather than after, because the failure is otherwise invisible: a contaminated
# run produces a plausible table.
#
# Four conditions, each named on failure:
#
#   GPU utilisation   a compute process resident on the card
#   GPU memory        something large holding VRAM
#   load average      CPU contention
#   sibling build     another training or benchmark process on this host
#
# What this cannot do is explain the ctx256 bimodality in
# `outputs/bench/ctx256-sweep.csv`, which was measured on an idle host and is
# still a 1.7384x gap between two tight clusters with nothing between them. So
# a clean host here is necessary and is not sufficient, and the sweep rather than
# this script is what tells a reader which statistic to trust.
#
# Each threshold is overridable and the default is stated rather than guessed:
# a reader who disagrees with 5% GPU or load 8 has one number to change and a
# reason to say so.
#
# Exits 0 when the host is clean, 1 when a condition failed.
set -eu

max_gpu_util=${MAX_GPU_UTIL:-5}
# Above this host's resting desktop baseline, not at zero. A GUI session keeps
# the card open and its compositor moves between roughly 250 and 1050 MiB
# depending on what is drawn, so a ceiling below that reads a clean host as busy
# whenever a window repaints. The signal that matters is a compute process, and
# the utilisation check above is what catches that.
max_gpu_mem_mib=${MAX_GPU_MEM_MIB:-1536}
max_load=${MAX_LOAD:-8}

if ! command -v nvidia-smi > /dev/null 2>&1; then
    echo "host-clean: no nvidia-smi, so no card to be contaminated by."
    echo "  A CPU-only host is clean for this purpose; treating it otherwise would"
    echo "  make the check refuse on a machine that has nothing to wait for."
    exit 0
fi

failed=0

util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)

# An unreadable GPU is a REFUSAL, not a clean bill of health, and this is the
# hole that would make the whole script worse than nothing. If `nvidia-smi`
# fails or prints nothing then `util` is empty, `[ "" -gt 5 ]` is a comparison
# error, an error inside an `if` condition is simply false, and all three GPU
# conditions pass -- so a broken driver would report a host it never read as
# clean. A guard that reports "clean" without having looked is the defect class
# this repository keeps finding in its own gates, so the readback is checked
# before it is believed.
for pair in "utilisation:$util" "memory:$mem"; do
    hc_name=${pair%%:*}
    hc_value=${pair#*:}
    case "$hc_value" in
        '' | *[!0-9]*)
            echo "host-clean: nvidia-smi returned no usable ${hc_name} (got '${hc_value}')." >&2
            echo "  Refusing rather than reporting a host that was never read." >&2
            echo "  If the card is genuinely gone, this is the wrong host to measure on." >&2
            exit 1
            ;;
    esac
done

if [ "$util" -gt "$max_gpu_util" ]; then
    echo "host-clean: GPU is at ${util}% and the ceiling is ${max_gpu_util}%." >&2
    echo "  A resident compute process is using the card. Which one:" >&2
    nvidia-smi --query-compute-apps=pid,process_name,used_memory \
        --format=csv,noheader >&2 2>/dev/null || echo "  (nvidia-smi listed none)" >&2
    echo "  A ratio taken now divides by a contaminated CPU call and reads as a" >&2
    echo "  speedup the kernel did not produce." >&2
    failed=1
fi

if [ "$mem" -gt "$max_gpu_mem_mib" ]; then
    echo "host-clean: GPU memory is ${mem} MiB and the ceiling is ${max_gpu_mem_mib} MiB." >&2
    echo "  Something large is resident. VRAM outlives the process that took it," >&2
    echo "  so a card can read idle and still carry the residue of a prior run." >&2
    failed=1
fi

load=$(awk '{print $1}' /proc/loadavg)
load_whole=${load%%.*}
if [ "$load_whole" -gt "$max_load" ]; then
    echo "host-clean: load average is ${load} and the ceiling is ${max_load}." >&2
    echo "  Every cpu_us here is a minimum over calls, so contention inflates it" >&2
    echo "  without bound and taking the minimum does not hide that." >&2
    failed=1
fi

# `pgrep -c` prints 0 AND exits 1 when nothing matches, so `|| echo 0` appends a
# second 0 and the comparison below dies on "integer expected". Counting lines
# is the form that stays numeric under both outcomes.
#
# The pattern names the training and benchmark binaries rather than `zig build`,
# because this script is itself run from inside a build step and would otherwise
# match its own parent and refuse every clean host.
siblings=$(pgrep -f 'ztransformer-train|ztransformer-cuda-train|ztransformer-dbg-train|attn_twin|norm_twin' 2>/dev/null | wc -l | tr -d ' ')
if [ "$siblings" -gt 0 ]; then
    echo "host-clean: ${siblings} training or benchmark processes are resident." >&2
    echo "  This host runs one job at a time. A second one makes the first's CPU" >&2
    echo "  column a measurement of the second." >&2
    pgrep -af 'ztransformer-train|attn_twin|norm_twin' >&2 2>/dev/null || true
    failed=1
fi

if [ "$failed" -ne 0 ]; then
    echo "host-clean: the host is NOT clean, so a timing taken now is not a" >&2
    echo "  measurement of this repository. Refusing rather than recording it." >&2
    exit 1
fi

echo "host-clean: GPU ${util}% and ${mem} MiB, load ${load}, no sibling build."
echo "  Thresholds: GPU <= ${max_gpu_util}%, VRAM <= ${max_gpu_mem_mib} MiB, load <= ${max_load}."
echo "  Necessary and not sufficient: see outputs/bench/ctx256-sweep.csv."
exit 0
