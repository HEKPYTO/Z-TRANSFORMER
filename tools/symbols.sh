#!/bin/sh
# Fails unless every `module.Symbol` row in a symbol table names a declaration
# that is `pub` in src/module.zig.
#
# The README arrives as an argument so a falsified copy can be checked without
# editing the real one, which is how `zig build table-block-check` proves its own
# assertions fire. The module name is the file stem, because the tables already say
# `tensor.Tensor` and the file is src/tensor.zig, so there is no map to keep in
# sync and no list of this script's own to go stale.
#
# One direction only. A `pub` nobody documented is an omission and costs a
# reader nothing. A row that says `pub` when the code says private is the
# README breaking the contract it is making, and that is the drift commit
# 6c2a9c5 had to hand-edit. Rows whose first cell names no module on disk
# (`init.arena`, a column of the allocator table) are not symbols, so a missing
# src/<mod>.zig is a skip rather than a failure.
#
# Silent on success: `verify` is silent by contract.
#
# `zig build verify` wires this in, and tracks the script and the README as inputs plus the library compile and the test run, so a declaration
# going private re-runs the gate from either direction. It still does not track
# a brand-new `src/<mod>.zig` that no README row names and no module imports,
# because then there is nothing to check; a new module that the README *does*
# name arrives through the README, which is tracked.
#
# One direction only, and deliberately. A `pub` nobody documented is an omission
# and costs a reader nothing. A row that says `pub` when the code says private
# is the README breaking the contract it is making, and that is the drift commit
# 6c2a9c5 had to hand-edit.
set -eu

readme=${1:-src/README.md}
root=${2:-.}

# The discovery grep below has to MATCH SOMETHING, and that is checked before
# anything else rather than inferred from the exit code.
#
# This is not hypothetical. The whole check is one pipeline: a `grep -o` feeds a
# `sed` feeds a `while` loop that accumulates into `$bad`, and the script exits 0
# when `$bad` is empty. Every stage of that pipeline succeeds on EMPTY input --
# `grep` finds nothing and exits 1, which `set -e` does not see because it is
# the left side of a pipe, and the loop body simply never runs. So a single extra
# space after the leading `|` in every table row, or the backticks dropped, takes
# 65 documented symbols down to 0 discovered ones and `verify` goes green over a
# table of 65 lies.
#
# Measured, on the real file: 65 rows today; one space of drift gives 0 rows and
# exit 0. And the failure mode is silence -- there is no output to notice,
# because a check that read nothing has nothing to report.
#
# The same reasoning is why `build.zig`'s report-rows gate asserts
# `grep -c . "$1" -lt 3` rather than trusting a diff, and why `src/tests.zig`
# asserts `named > 0` before walking. This gate had no such floor and an audit
# found it.
# `${rows:-0}` is load-bearing, not defensive noise. `grep -c` writes its count to
# stdout and its diagnostic to stderr, so an UNREADABLE $readme leaves $rows empty,
# `|| true` swallows the exit 2, and `[ "" -lt 20 ]` is an arithmetic error inside an
# `if` -- which is simply false, so the floor this block exists to enforce silently
# does not run. An audit found that this gate passed over zero symbols. A floor whose
# own operand can be empty is not a floor.
[ -r "$readme" ] || {
    echo "$readme: not readable, so the symbol gate has read nothing." >&2
    echo "  A gate that read nothing is not a clean bill of health." >&2
    exit 1
}
rows=$(grep -c '^| `[a-z_]*\.[A-Za-z_][A-Za-z_0-9]*`' "$readme" 2>/dev/null || true)
rows=${rows:-0}
if [ "$rows" -lt 20 ]; then
    echo "$readme: found $rows module.Symbol rows to check." >&2
    echo "  That is far below the 65 the table has, so the discovery grep is" >&2
    echo "  reading nothing and this check is vacuous. Either the table moved or" >&2
    echo "  the pattern is stale; both are failures, and both are silent today." >&2
    exit 1
fi

bad=$(
    grep -o '^| `[a-z_]*\.[A-Za-z_][A-Za-z_0-9]*`' "$readme" |
        sed 's/^| `//; s/`$//' |
        while IFS=. read -r mod sym; do
            f="$root/src/$mod.zig"
            # An ABSENT module was a silent skip, and that is the whole defect. A
            # blanket `|| continue` cannot tell "this first cell was never a module"
            # from "the module moved", so renaming src/gradcheck.zig took six
            # documented symbols out of the check with no line of output and verify
            # stayed green over a table of six lies. The allowlist below is measured,
            # not guessed: of the seventeen distinct first cells this table carries,
            # exactly one has no src/<cell>.zig, and it is `init`, whose rows are an
            # allocator table whose cells are names rather than module.Symbol pairs.
            # Every other absent module is now a failure, which is what a broken build
            # is. A grep rather than a `case`, for the reason in the comment below.
            if [ ! -f "$f" ]; then
                if printf '%s\n' "$mod" | grep -qx 'init'; then continue; fi
                echo "  $mod.$sym is documented, but src/$mod.zig does not exist"
                continue
            fi
            # `loss.csv` and `norm.cu` are filenames sitting in a first cell, not
            # symbols. `loss` and `norm` are both real modules, so the
            # missing-file skip above does not catch them and the row reads as a
            # symbol that was never declared. Extensions are a closed list, so
            # this is a list rather than a guess at which cells look like code.
            # A `case` is the obvious spelling and does not parse here: this is
            # the last stage of a pipeline inside a command substitution, and
            # macOS bash 3.2 rejects `case` there with a syntax error on the
            # pattern line. A grep is one line and has no such rule.
            if printf '%s\n' "$sym" |
                grep -qE '^(zig|cu|sh|py|csv|tsv|txt|json|bin|yml|md|out)$'
            then
                continue
            fi
            grep -qE "^pub (fn|const|var) $sym\b" "$f" ||
                echo "  $mod.$sym is documented as pub, but src/$mod.zig does not declare it pub"
        done || true
)

if [ -n "$bad" ]; then
    echo "$readme documents symbols that are not public:" >&2
    echo "$bad" >&2
    echo "make the declaration pub, or drop the row." >&2
    exit 1
fi
exit 0
