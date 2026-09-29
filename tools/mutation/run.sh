#!/bin/sh
# The command behind the mutation numbers. How good is this suite, as a number
# somebody can check?
#
#   sh tools/mutation/run.sh                    the default set
#   sh tools/mutation/run.sh matmul,rope        only those, by name
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
# the last thing printed is proof it is gone.
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
# first time it was run would be deleted rather than fixed. Exit 1 means the
# measurement is untrustworthy — a mutation whose pattern no longer matches, or
# one that does not compile, so a number was reported for code that never ran.
# Exit 2 means the harness could not run at all.
set -u

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
# either way. A Debug-only default is 39 minutes for 16 mutations, which is a
# number nobody runs, and a default nobody runs measures nothing. ReleaseFast is
# also the mode the `train` binary ships in, so it is the mode whose silence a
# reader would actually be misled by.
#
# What this costs is stated rather than assumed, because it is a real limit and
# not a formality: a defect only Debug's safety checks trip is invisible to the
# default run, so Debug-only catches are not counted. The three survivors in the
# default set were re-run through this same script with MUTATION_MODE=debug and
# all three survive Debug too, so on this set the choice cost nothing measured.
# `MUTATION_MODE=debug` re-adjudicates the whole selection in Debug for anyone
# who wants the other gate, which is how a survivor gets promoted to caught
# without paying for the set twice.
mode=${MUTATION_MODE:-release}
case "$mode" in
    release) set_cmd="test -Doptimize=ReleaseFast" ;;
    debug)   set_cmd="test" ;;
    *)       set_cmd="verify" ;;
esac

# A digest of the whole of src/, taken before anything happens and compared
# after. Not `git status`, because this repository's working tree legitimately
# carries uncommitted work from other work at times, and a status-based check
# would then either report a false failure or — worse, to pass it — be
# loosened until it reported nothing. A digest of the bytes is the claim being
# made: this harness did not write to src/, and it can be checked without
# knowing anything about what else was going on.
src_digest() {
    find "$root/src" -type f -name '*.zig' -print0 |
        LC_ALL=C sort -z |
        xargs -0 shasum -a 256 2>/dev/null |
        shasum -a 256 2>/dev/null | cut -d' ' -f1
}

before=$(src_digest)
if [ -z "$before" ]; then
    echo "mutation: could not digest src/ (exit 2)" >&2
    exit 2
fi

scratch=$(mktemp -d "${TMPDIR:-/tmp}/ztransformer-mutation.XXXXXX") || exit 2
wt=$scratch/tree
log=$scratch/run.log
mutate_bin=$scratch/mutate

# One trap, one job: the worktree goes, and the evidence is printed. `git
# worktree remove` is the documented way to undo `git worktree add`; the `rm -rf`
# behind it is for the case where the removal itself fails, which is the case
# where a stale worktree would otherwise be registered in this repository's
# admin data and outlive the run. A survivor of the trap is a scratch directory
# on disk and nothing more: the working tree was never written to.
cleanup() {
    status=$?
    after=$(src_digest)
    if [ -d "$wt" ]; then
        git -C "$root" worktree remove --force "$wt" >/dev/null 2>&1 || rm -rf "$wt"
    fi
    git -C "$root" worktree prune >/dev/null 2>&1 || true
    echo
    echo "restoration"
    if git -C "$root" worktree list | grep -qF "$wt"; then
        echo "  FAIL    $wt is still a registered worktree"
    else
        echo "  ok      $wt is no longer a registered worktree"
    fi
    if [ "$after" = "$before" ]; then
        echo "  ok      src/ in the checkout this ran from is byte for byte as it was at start"
    else
        echo "  FAIL    src/ in the checkout this ran from changed: ${before} -> ${after}"
    fi
    rm -rf "$scratch"
    if [ -e "$scratch" ]; then
        echo "  FAIL    the scratch directory survived"
    else
        echo "  ok      the mutator binary, its build cache and the logs were scratch, and are gone"
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
# edges. Names are comma separated and may be globs, because the point of a
# subset is to re-check one area in another mode without re-running the world.
all=$("$mutate_bin" list) || exit 2
if [ "$#" -eq 0 ]; then
    selected=$all
else
    selected=""
    for pattern in $(printf '%s' "$1" | tr ',' ' '); do
        for name in $all; do
            # shellcheck disable=SC2254  # the pattern is meant to glob.
            case $name in $pattern) case " $selected " in *" $name "*) ;; *) selected="$selected $name" ;; esac ;; esac
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
    for name in $survivors; do
        "$mutate_bin" show "$name" | sed 's/^/  /'
        case $name in
            norm-reassociate)
                echo "  class   below-gate: v / rms * w and v * w / rms differ only in rounding, and"
                echo "          every assertion here is looser than that rounding. This is not a hole"
                echo "          in the suite. Catching it means a test that pins the exact order, which"
                echo "          pins one f32 rounding rather than the function."
                ;;
            matmul-f64-acc)
                echo "  class   HOLE, and a measured one. The two reductions are not equivalent."
                echo "          Reducing k weights drawn at the init scale in f32 rather than f64"
                echo "          diverges by 5.1e-6 relative at k=64, 2.7e-5 at the ffn width of 256,"
                echo "          and 1.2e-4 at k=4096, worst case over 200 draws at seed 7. The suite's"
                echo "          matmul tests assert a relative bound tighter than that and still pass,"
                echo "          because no test here multiplies a tall and wide matrix. So the f32"
                echo "          accumulator in tensor.zig is an unverified choice, not a tested one,"
                echo "          and the honest fix is a matmul test that reduces in f64 at a wide k."
                ;;
            norm-f32-acc)
                echo "  class   below-gate at the tested widths: d_model 64 to 4096 is 4096 f32"
                echo "          squares at most, a drift under 1e-6, under the 2e-6 gate. The f64"
                echo "          accumulator is right; this suite does not prove it, and the parity"
                echo "          sweep at d_model 64 does not either. tools/README.md says the same."
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
