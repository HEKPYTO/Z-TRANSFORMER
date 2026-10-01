#!/bin/sh
# The command behind the mutation numbers. How good is this suite, as a number
# somebody can check?
#
#   sh tools/mutation/run.sh                    the default set
#   sh tools/mutation/run.sh matmul,rope        only those, matched as substrings
#   MUTATION_MODE=debug sh tools/mutation/run.sh    re-adjudicate in Debug
#
# It mutates source, runs the suite, and reports which tests caught the defect.
# Every other property of a mutation run follows from two facts, both of which a
# hand-run got wrong here first.
#
# It never touches this tree. The mutants are applied in a `git worktree`
# pinned to HEAD, thrown away at the end. A harness that mutates the checkout it
# is running from is a harness whose crash leaves the repository broken, and a
# repository with a mutant in it is worse than a repository with no harness: the
# next `zig build` reports a failure nobody caused, in a line nobody wrote, and
# the honest fix — finding out which line — is the one thing nobody thinks to do.
# The trap below removes the worktree on every exit path including SIGINT, and
# the last thing printed is proof it is gone — and a check that fails sets the
# exit status, because a check that only prints is a check nobody has to believe.
# SIGKILL cannot be trapped, so a killed run leaves the scratch directory, which
# opens with a file naming the repository and the one command that clears it.
#
# It runs `zig fmt` on each mutant before the suite. `verify` gates its two
# test binaries on `zig fmt --check`, so an unformatted mutant is rejected by
# the formatter before a single test executes — and the harness would record
# that as "caught by the suite". It was caught by nothing. Formatting first is
# the difference between a count that means something and a count that is
# accidentally a measure of how hard the mutation is to type.
#
# It exits 0 whenever the run completed and every mutation got a verdict,
# survivors included. A coverage number is a measurement, not a pass/fail gate:
# a suite is not imperfect, it is measured, and a command that went red the
# first time it was run would be deleted rather than fixed. Exit 1 means
# something makes the output untrustworthy — a mutation whose pattern no longer
# matches, one that does not compile so a number was reported for code that never
# ran, or a restoration check that failed, which means the run left something of
# itself behind. Exit 2 means the harness could not run at all.
#
# Restoration failure is an exit status and not a line of output. A FAIL that
# prints and returns 0 is worse than no check at all: it trains the reader to
# see the word FAIL and reach for the number in front of the shell, which is
# exactly the habit that lets a mutant sit in a checkout for a week.
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd) || exit 2
here=$root/tools/mutation
cd "$root" || exit 2

if ! command -v zig >/dev/null 2>&1; then
    echo "mutation: no zig on PATH (exit 2, not a coverage result)" >&2
    exit 2
fi

# ReleaseFast by default, and the reason is measured rather than assumed. On this
# machine the Debug suite takes 145 s and ReleaseFast 33 s, and the difference is
# the suite executing, not compiling — a mutation costs a rebuild of one file
# either way. A Debug-only default is 44 minutes for 18 mutations, which is a
# number nobody runs, and a default nobody runs measures nothing. ReleaseFast is
# also the mode the `train` binary ships in, so it is the mode whose silence a
# reader would actually be misled by.
#
# What this costs is stated rather than assumed, because it is a real limit and
# not a formality: a defect only Debug's safety checks trip is invisible to the
# default run, so Debug-only catches are not counted. The two survivors in the
# default set were re-run through this same script with MUTATION_MODE=debug and
# both survive Debug too, so on this set the choice cost nothing measured.
# `MUTATION_MODE=debug` re-adjudicates the whole selection in Debug for anyone
# who wants the other gate, which is how a survivor gets promoted to caught
# without paying for the set twice.
mode=${MUTATION_MODE:-release}
case "$mode" in
    release) set_cmd="test -Doptimize=ReleaseFast" ;;
    debug)   set_cmd="test" ;;
    *)       set_cmd="verify" ;;
esac

# A digest of the whole of src/ used to be taken here and compared after. It is
# gone, because it could only ever detect one of two things and the two are not
# distinguishable from inside the run: this harness writing to the caller's
# checkout, and somebody else editing the checkout at the same time. It never
# does the first, and it did not once fail to be wrong about the second — in a
# shared tree a colleague's `vim src/model.zig` produced "FAIL src/ changed",
# and the fix for a check that cries wolf every ten minutes is to delete it, not
# to tune it.
#
# What is left is the claim the harness can actually stand behind: it wrote in
# two places, both of which it owns and both of which it destroys. The worktree,
# checked through git's own registry rather than by looking for a directory,
# because a directory that is gone but still registered is what outlives a run;
# and the scratch directory holding the mutator, its build cache and the log.
wt_list() {
    git -C "$root" worktree list --porcelain 2>/dev/null |
        sed -n 's/^worktree //p' | LC_ALL=C sort
}

before=$(wt_list)
if [ -z "$before" ]; then
    echo "mutation: could not read \`git worktree list\` (exit 2)" >&2
    exit 2
fi

# A separate assignment, because `${TMPDIR:-/tmp%/}` only strips the slash from
# the fallback: TMPDIR on macOS ends in one, and the path is printed twice in the
# output, where a `//` reads like a different path.
tmp=${TMPDIR:-/tmp}
scratch=$(mktemp -d "${tmp%/}/ztransformer-mutation.XXXXXX") || exit 2
wt=$scratch/tree
log=$scratch/run.log
mutate_bin=$scratch/mutate

# SIGKILL cannot be trapped, so nothing runs when it lands. What is left is this
# directory, and it has to identify itself to whoever trips over it weeks later.
# Two lines, because the name alone says "some mutation run happened here" and
# not which repository the worktree inside it belongs to — which is the one
# thing the reader needs in order to run `git worktree remove` and be done.
{
    echo "ztransformer mutation scratch, left by a run that was killed."
    echo "repository: $root"
    echo "worktree:   $wt"
    echo "clean up:   git -C '$root' worktree remove --force '$wt'; rm -rf '$scratch'"
} >"$scratch/owner.txt"

# One trap, one job: the worktree goes, the evidence is printed, and a failed
# check sets the exit status. `git worktree remove` is the documented way to
# undo `git worktree add`; the `rm -rf` behind it is for the case where the
# removal itself fails, which is the case where a stale worktree would
# otherwise be registered in this repository's admin data and outlive the run.
# A survivor of the trap is a scratch directory on disk and nothing more: the
# working tree was never written to.
cleanup() {
    status=$?
    restore=0
    # stdout is discarded, stderr is not. A removal that fails is the single
    # most useful thing this script can tell you and `2>&1` was throwing it
    # away, which is how the one real failure of this check seen while building
    # it cost a second run to diagnose: git's own "fatal:" line is the
    # diagnosis, and the registry comparison below only says something survived.
    if [ -d "$wt" ]; then
        git -C "$root" worktree remove --force "$wt" 2>&1 >/dev/null | sed 's/^/          git: /'
    fi
    if [ -d "$wt" ]; then
        rm -rf "$wt"
    fi
    git -C "$root" worktree prune 2>&1 >/dev/null | sed 's/^/          git: /'
    echo
    echo "restoration"
    # One check, not two. The first draft of this asked git to search its
    # registry for the worktree path and reported ok when it did not find it —
    # which it never does on macOS, where `mktemp -d` hands back `/var/...` and
    # git reports the resolved `/private/var/...`, so a `grep -F` on the literal
    # string is a check that cannot fail. Comparing the whole registry against
    # the whole registry taken at start needs no path string to match, catches
    # the same worktree, and catches a second one besides.
    after=$(wt_list)
    if [ "$after" = "$before" ]; then
        echo "  ok      $wt is gone, and the set of registered worktrees is the one this run started with"
    else
        # The only thing that can differ here is a worktree this run added and
        # did not remove, so name it rather than printing two lists and asking
        # the reader to diff them.
        echo "  FAIL    these worktrees are registered now and were not at start:"
        printf '%s\n' "$after" | while IFS= read -r line; do
            printf '%s\n' "$before" | grep -qxF "$line" || echo "            $line"
        done
        echo "          run: git -C '$root' worktree prune  (then remove anything left over by hand)"
        restore=1
    fi
    rm -rf "$scratch"
    if [ -e "$scratch" ]; then
        echo "  FAIL    the scratch directory survived: $scratch"
        echo "          it holds only the mutator, its build cache and the log; rm -rf it"
        restore=1
    else
        echo "  ok      the mutator binary, its build cache and the logs were scratch, and are gone"
    fi
    # The one line the whole trap exists for. `status` is the run's own verdict
    # and is preserved unless a check failed, so a clean run over a messy result
    # still reports the mess, and a clean result over a failed restoration
    # reports the restoration.
    if [ "$restore" -ne 0 ]; then
        if [ "$status" -eq 0 ]; then
            status=1
        fi
        echo "  exit 1  the run left something of itself behind; the output above is not a clean measurement"
    fi
    exit $status
}
trap cleanup EXIT INT TERM

# The mutator is built outside the worktree, into the scratch directory, so that
# removing the worktree at the end does not delete the program that is still
# running. It is a Zig program rather than a `sed` line because applying a
# pattern that no longer occurs must fail rather than silently write an
# unmutated file: an unmutated file passes every test, and the harness would
# report a live defect as a coverage hole. See tools/mutation/mutate.zig.
if ! zig build-exe "$here/mutate.zig" -femit-bin="$mutate_bin" \
    --cache-dir "$scratch/zig-cache" -OReleaseSafe 2>"$log"; then
    echo "mutation: the mutator did not build (exit 2, not a coverage result)" >&2
    sed 's/^/  /' "$log" >&2
    exit 2
fi

if ! git -C "$root" worktree add --detach "$wt" HEAD >"$log" 2>&1; then
    echo "mutation: could not create the throwaway worktree (exit 2)" >&2
    sed 's/^/  /' "$log" >&2
    exit 2
fi

# The selection. Empty means every mutation, which is the set a reviewer should
# run: the twelve that guard a load-bearing claim and the four that probe the
# edges. Names are comma separated, and each one is a substring match, so
# `matmul,rope` is the two families rather than zero mutations and a puzzled
# reader: `case` patterns match the whole word, and every name in the table
# carries a suffix, so an undecorated pattern is a name that does not exist.
# Wrapping the pattern in `*` costs two characters and is the difference between
# the subset selector working and the one command a developer would reach for
# during a run failing with exit 2.
all=$("$mutate_bin" list) || exit 2
if [ "$#" -eq 0 ]; then
    selected=$all
else
    selected=""
    for pattern in $(printf '%s' "$1" | tr ',' ' '); do
        for name in $all; do
            case $name in
                *$pattern*) case " $selected " in *" $name "*) ;; *) selected="$selected $name" ;; esac ;;
            esac
        done
    done
    selected=$(printf '%s' "$selected" | sed 's/^ *//')
    if [ -z "$selected" ]; then
        echo "mutation: no mutation matches '$1' (exit 2)" >&2
        "$mutate_bin" list >&2
        exit 2
    fi
fi

count=0
caught=0
unbuildable=0
broken=0
survivors=""
for name in $selected; do
    if [ "$count" -eq 0 ]; then
        echo "mutation: $(git -C "$wt" rev-parse --short HEAD) in a throwaway worktree, gate = \`zig build $set_cmd\`"
        echo "mutation: one mutant at a time; the suite is the only judge"
    fi
    count=$((count + 1))

    # Restore before applying, not after. Restoring first means a mutant that
    # crashes the harness cannot leak into the next one, so the only thing that
    # has to be true at any point is that the worktree is the only place a
    # mutant can be.
    git -C "$wt" checkout -- . 2>/dev/null

    if ! "$mutate_bin" apply "$name" "$wt" >"$log" 2>&1; then
        broken=$((broken + 1))
        printf '  ERROR   %-20s the pattern did not apply; nothing was run\n' "$name"
        sed 's/^/          /' "$log"
        continue
    fi
    file=$("$mutate_bin" show "$name" | sed -n 's/^  file  //p')

    # `zig fmt` on the one mutated file, before the gate. Not `zig fmt .` and not
    # `--check`: the mutant is what has to become well formed, and `--check` is
    # the mode that fails on it.
    if ! zig fmt "$wt/$file" >/dev/null 2>&1; then
        broken=$((broken + 1))
        printf '  ERROR   %-20s zig fmt rejected the mutant; nothing was run\n' "$name"
        continue
    fi

    # Three verdicts, and the middle one is not the same claim as the third. A
    # suite that failed is caught. A build that failed is caught by the compiler,
    # which says the mutant does not compile and nothing about the suite. And a
    # build that succeeded is survived. Collapsing the first two would overstate
    # what the suite is worth; collapsing the second into the third would hide
    # that the mutant was never exercised at all.
    if (cd "$wt" && zig build $set_cmd) >"$log" 2>&1; then
        verdict=survived
    elif grep -q 'run test [0-9]* pass, [1-9][0-9]* fail' "$log"; then
        verdict=caught
    elif grep -qE 'error: [0-9]+ compilation errors|error: the following command failed' "$log"; then
        verdict=unbuildable
    else
        verdict=error
    fi

    case $verdict in
        caught)
            caught=$((caught + 1))
            # Both failure shapes, because they are both real: most tests print
            # a message, and a test that asserts on an expectEqual has nothing
            # to print. Counting only the first would have reported
            # `loss-no-rowmax` as caught by 0 tests, which reads as a mistake in
            # the harness rather than a test that fails silently.
            n=$(sed -n "s/^error: '\\(.*\\)' failed.*/\\1/p" "$log" | wc -l | tr -d ' ')
            first=$(sed -n "s/^error: '\\(.*\\)' failed.*/\\1/p" "$log" | head -1)
            printf '  caught  %-20s %s test(s), first: %s\n' "$name" "$n" "$first"
            ;;
        unbuildable)
            unbuildable=$((unbuildable + 1))
            broken=$((broken + 1))
            printf '  BUILD   %-20s does not compile; fix the mutation, it measures nothing\n' "$name"
            ;;
        survived)
            survivors="$survivors $name"
            printf '  SURVIVED %-19s no test in the suite failed\n' "$name"
            ;;
        *)
            broken=$((broken + 1))
            printf '  ERROR   %-20s the gate itself failed\n' "$name"
            sed 's/^/          /' "$log" | head -20
            ;;
    esac
done

# The summary, and the survivors in the form the next decision needs: not a
# count, but a list with each one's meaning attached. A survivor that is
# behaviourally identical is not a hole in the suite, and a survivor that moves
# the numbers is one. Reporting the count alone would let a reader assume the
# worse of the two for every entry, which is exactly the rounding-up this
# project does not do anywhere else.
echo
# Three numbers, because they are three different claims. `caught` is the only
# one that says anything about the suite. `BUILD` is a mutant that does not
# compile, which says the mutation was badly written and nothing else, so it is
# not folded into the headline: folding it in is how a harness ends up claiming
# coverage it measured nothing for. `survived` is the rest.
echo "summary: caught $caught of $count, gate \`zig build $set_cmd\`"
if [ "$unbuildable" -ne 0 ]; then
    echo "         $unbuildable did not compile and are not counted above; they measure nothing"
fi
if [ -n "$survivors" ]; then
    echo
    echo "survivors"
    # Reached only for names whose verdict was `survived`, which is the one
    # branch that appends to `$survivors`. A class is unreachable for a caught
    # mutation by construction, and any survivor without one lands on
    # UNCLASSIFIED rather than being described in language the run no longer
    # supports.
    for name in $survivors; do
        "$mutate_bin" show "$name" | sed 's/^/  /'
        case $name in
            norm-reassociate)
                echo "  class   below-gate: v / rms * w and v * w / rms differ only in rounding, and"
                echo "          every assertion here is looser than that rounding. This is not a hole"
                echo "          in the suite. Catching it means a test that pins the exact order, which"
                echo "          pins one f32 rounding rather than the function."
                ;;
            clip-ge)
                echo "  class   EQUIVALENT, and provably so rather than hopefully. The mutant differs"
                echo "          from the original only when the norm is exactly equal to max_grad_norm,"
                echo "          and there scale = max/norm = 1 exactly, so every element is multiplied"
                echo "          by 1.0 and both return the same norm. The f64 scale is exact, so this is"
                echo "          a bit-for-bit identical run, not a close one. No test can ever catch it"
                echo "          and writing one would assert nothing. It is here because a reader"
                echo "          deserves the proof rather than a guess."
                ;;
            *)
                echo "  class   UNCLASSIFIED. Either behaviourally identical, or a real hole in the"
                echo "          suite. Nobody has decided which, and this harness does not guess."
                echo "          Classify it before quoting the number."
                ;;
        esac
    done
fi

[ "$broken" -eq 0 ] || exit 1
exit 0
