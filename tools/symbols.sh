#!/bin/sh
# Fails unless every `module.Symbol` row in a symbol table names a declaration
# that is `pub` in src/module.zig.
#
# The README arrives as an argument so a falsified copy can be checked without
# editing the real one, which is how tools/removed/sensitivity.sh proves its own
# gates fire. The module name is the file stem, because the tables already say
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
# ponytail: `zig build verify` wires this in, and tracks the script and the
# README as inputs plus the library compile and the test run, so a declaration
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

bad=$(
    grep -o '^| `[a-z_]*\.[A-Za-z_][A-Za-z_0-9]*`' "$readme" |
        sed 's/^| `//; s/`$//' |
        while IFS=. read -r mod sym; do
            f="$root/src/$mod.zig"
            [ -f "$f" ] || continue
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
