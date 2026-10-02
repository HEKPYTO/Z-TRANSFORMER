#!/bin/sh
# Can tools/table-block.sh fail? Break it, and require that it notices and says so.
#
#   sh tools/table-block-check.sh
#
# A gate that cannot fail is indistinguishable from a gate that is not there, and
# the only way this repository finds out which of the two it has is by breaking it
# on purpose and watching. That is `sh tools/removed/sensitivity.sh` for the
# oracle, `sh src/cuda/run-attn.sh`'s four broken kernels for the CUDA attention
# gates, and this for the one gate that decides whether a published table can be
# hand-edited.
#
# Four cases. The first MATCHES, because a harness in which everything fails
# proves nothing -- without it, a check that rejected all three broken tables by
# accident would read the same as one that rejected them on purpose. The other
# three must not, and each is broken the way the defect arrives in practice
# rather than by a flag invented here:
#
#   a digit edited by hand   the drift the block exists to catch: someone fixes
#                            a number in the README and the tool still prints the
#                            old one
#   a marker deleted         the case most likely to be wrong, because an `awk`
#                            whose pattern matches nothing yields an EMPTY
#                            extraction rather than an error, and the floor below
#                            only guards the tool side
#   a tool that prints nothing   the vacuous-pass shape: BOTH sides extracted
#                            nothing and `cmp -s` exits 0 on two empty files, so
#                            the check passes having graded nothing at all. The
#                            README block is emptied as well as the tool on
#                            purpose, because that is the only way this case
#                            reaches the floor -- with the block intact the diff
#                            would fail the case on its own, the floor would never
#                            be exercised, and dropping the floor would change
#                            nothing this script could see.
#
# The fixture is eleven lines of table over three headers, and the floor is 6, so
# the floor is under the real thing and a truncated extraction is still under it.
#
# Every case prints what the check printed and what it exited with, on a passing
# run as well as a failing one, because "the gate cannot fail" is only a claim
# until the failure is in a transcript.
#
# Six assertions: four cases, one about a message rather than an exit code, and
# one about the floor itself.
#
# Exits 0 when all six behaved, 1 when one did not.
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd) || exit 2
cd "$root" || exit 2

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

block=table-block-check
floor=6

cat > "$work/tool.txt" <<'EOF'
a banner line that opens no table
TABLE ONE
alpha    1
beta     2

TABLE TWO
gamma    3
delta    4

TABLE THREE
epsilon  5
zeta     6
EOF

# An unquoted heredoc, because $block has to be in the markers, and so the
# backticks are escaped: they would otherwise be a command substitution.
cat > "$work/readme.md" <<EOF
# fixture

<!-- $block:begin -->
\`\`\`
TABLE ONE
alpha    1
beta     2

TABLE TWO
gamma    3
delta    4

TABLE THREE
epsilon  5
zeta     6
\`\`\`
<!-- $block:end -->
EOF

# A hand-edited digit: the tool still prints 5 on that row, the README says 7.
sed 's/^epsilon  5$/epsilon  7/' "$work/readme.md" > "$work/drifted.md"
# A marker removed: `awk` matches nothing and extraction comes back empty.
sed "/<!-- $block:end -->/d" "$work/readme.md" > "$work/nomarker.md"
# A tool that printed nothing at all.
: > "$work/silent.txt"

# The README half of the vacuous pass: the two markers adjacent, so the block
# between them is zero lines and the extraction really is empty. A blank line
# left inside would not do -- the extractor strips only the fence, so one blank
# line is one line to diff against, `cmp` fails on that difference, and the floor
# is never reached.
printf '# fixture\n\n<!-- %s:begin -->\n<!-- %s:end -->\n' "$block" "$block" \
    > "$work/blanked.md"

failures=0

check() {
    label=$1
    want=$2
    tool=$3
    readme=$4
    if sh tools/table-block.sh "$tool" "$readme" "$block" "$floor" \
        "TABLE ONE" "TABLE TWO" "TABLE THREE" \
        > "$work/out" 2>&1; then
        got=0
    else
        got=$?
    fi
    echo "== $label: exit $got, expected $want"
    sed 's/^/   | /' "$work/out"
    if [ "$got" -ne "$want" ]; then
        echo "   ^^ NOT AS EXPECTED: this case did not behave as the check claims"
        failures=$((failures + 1))
    fi
}

check "the tool and the block agree" 0 "$work/tool.txt" "$work/readme.md"
check "a digit edited by hand" 1 "$work/tool.txt" "$work/drifted.md"
check "a marker deleted" 1 "$work/tool.txt" "$work/nomarker.md"
check "the tool printed nothing" 1 "$work/silent.txt" "$work/blanked.md"

# And one for the floor, which is the only thing standing between a caller and
# the vacuous pass: `floor 0` makes every extraction "long enough", so the empty
# pair the fourth case is built to exercise would sail through. The script
# refuses it; this asserts the refusal, because a guard nothing here exercises
# is a guard that can be deleted without anything going red.
if sh tools/table-block.sh "$work/silent.txt" "$work/blanked.md" "$block" 0 \
    "TABLE ONE" "TABLE TWO" "TABLE THREE" > "$work/out" 2>&1; then
    got=0
else
    got=$?
fi
echo "== a floor of 0: exit $got, expected 2"
sed 's/^/   | /' "$work/out"
if [ "$got" -ne 2 ]; then
    echo "   ^^ NOT AS EXPECTED: floor 0 IS the vacuous gate and must be refused"
    failures=$((failures + 1))
fi

# And one assertion about the MESSAGE rather than the exit code, which is the
# only thing that tells the marker case from the digit case: with the tool's
# output non-empty, a deleted marker still fails the `cmp`, so an absent marker
# and a drifted digit both exit 1 and the exit code alone cannot say which
# happened. The fix has to be a copy-paste rather than an investigation, and
# this is what pins that.
if sh tools/table-block.sh "$work/tool.txt" "$work/nomarker.md" "$block" "$floor" \
    "TABLE ONE" "TABLE TWO" "TABLE THREE" 2>&1 \
    | grep -qF "no <!-- $block:end --> marker"; then
    echo "== the missing-marker failure names the missing marker: yes"
else
    echo "== the missing-marker failure names the missing marker: NO"
    failures=$((failures + 1))
fi

if [ "$failures" -ne 0 ]; then
    echo "table-block-check: $failures of 6 assertions did not behave as required."
    exit 1
fi
echo "table-block-check: 6 of 6 assertions behaved as required."