# tools

The mutation harness: how good the test suite is, as a number somebody can check, measured by
breaking the code on purpose and requiring the suite to notice. Plus the shell scripts that hold the
documented tables in this repository to the code they describe, and the host-state gate the
benchmark scripts run before they will print a ratio.

## Mutation

How good is the test suite, as a number somebody can check.

    sh tools/mutation/run.sh                      # 18 mutations, about 10 minutes
    sh tools/mutation/run.sh matmul,rope          # a subset, matched as substrings
    MUTATION_MODE=debug sh tools/mutation/run.sh  # the same set in Debug, 44 minutes

Each argument is matched as a substring, so `matmul` is the whole matmul family and `rope` is the
rope mutation. That is the command a developer reaches for mid-run, and it used to exit 2: a `case`
pattern matches the whole word, every name in the table carries a suffix, and a name that does not
exist is not a useful thing for a selector to report.

It applies a deliberate defect to the source, runs the suite, and reports which tests caught it.
Every quality claim this repository makes about its own tests is a version of that question, and
until this existed the answer was a number in a chat message. `sensitivity.sh` asks it of the
numeric gates; this asks it of the suite.

Exit 0 whenever the run completed and every mutation got a verdict, survivors included. A coverage
number is a measurement, not a pass/fail gate: a suite is not imperfect, it is measured, and a
command that went red the first time anyone ran it would be deleted rather than fixed. Exit 1 means
something makes the output untrustworthy — a mutation whose pattern stopped matching, one that does
not compile so a number was reported for code that never ran, or a restoration check that failed,
which means the run left something of itself behind. Exit 2 means the harness could not run.

### The two traps a hand-run falls into

**It never mutates this tree, and it says so in a way you can check.** The mutants go into a
`git worktree` pinned to HEAD, removed by a trap on every exit path including SIGINT. The run ends
by comparing the whole of `git worktree list` against the copy it took before the first mutation and
printing the outcome. A harness that edits the checkout it runs from leaves a mutant behind when it
crashes, and a repository with a mutant in it is worse than one with no harness — the next
`zig build` reports a failure nobody caused, on a line nobody wrote.

The check is scoped to the two places the harness writes, both of which it owns and destroys: the
worktree, compared through git's registry rather than by looking for a directory, because a directory
that is gone but still registered is what outlives a run; and the scratch directory holding the
mutator, its build cache and the log. An earlier version digested all of `src/` instead, and that
check could not tell the two things it was watching for apart: this harness writing to your
checkout, and a colleague editing it at the same time. It never did the first, and in a shared tree
it reported the second as a failure often enough to be worth deleting. A check that cannot separate
its causes is not a check, so it is gone rather than tuned.

SIGKILL cannot be trapped, so a killed run does not clean up. What it leaves behind is a scratch
directory whose `owner.txt` names the repository and the one command that clears it, which is the
only thing worth writing down for a signal that leaves no chance to write anything else.

**It runs `zig fmt` on the mutant before the suite.** `verify` gates both of its test binaries on
`zig fmt --check`, so an unformatted mutant is rejected by the formatter before a single test
executes. A naive harness records that as "caught by the suite" when the suite was never entered,
and every such mutation reads as coverage that does not exist.

### ReleaseFast, and what that costs

The Debug suite and the ReleaseFast one are both timed by the two commands below, on this host
with a warm build cache; a cold cache adds compilation to both and moves neither ratio.

    zig build test                              # Debug: 145 s here
    zig build test -Doptimize=ReleaseFast        # ReleaseFast: 33 s here

The difference is the suite executing, not compiling: a mutation invalidates one file either way. A
Debug-only default is about 44 minutes for 18 mutations, at the 145 s a Debug mutation measured in
`run.sh`'s own timing comment rather than at a second, conflicting figure, which is a number nobody runs, and a default
nobody runs measures nothing. ReleaseFast is also the mode the `train` binary ships in.

The cost is real and is stated rather than hidden: a defect only Debug's safety checks trip is
invisible to the default run, so Debug-only catches are not counted. Both survivors below were
re-run through this same script with `MUTATION_MODE=debug` and both survive Debug too, so on
this set the choice cost nothing measured. `MUTATION_MODE=verify` runs both configurations.

### Three verdicts, and a class for every survivor

`caught` is a failing test. `BUILD` is a mutant that does not compile, which says the mutation was
badly written and nothing about the suite, so it is reported separately and never folded into the
headline. `SURVIVED` is the rest, and each survivor is classified in the output rather than counted,
because a survivor that is behaviourally identical is not a hole in the suite and one that moves the
numbers is.

**Measured `2026-10-03`: caught 15 of 18, on both hosts.** On the development machine and on the
Linux host of record, which is the platform family CI runs on. Same fifteen, same three survivors,
on both. The three are classified below, and all three are equivalent or below the gates, so none
of them is a hole in the suite -- which is what makes the CI floor a measurement rather than a guess
carried from one host to another.

The two runs are recorded by date and by host rather than by commit, deliberately. This repository's
history was rewritten after they were taken, so the SHAs they named no longer resolve and citing
them would send a reader to `Not a valid object name`. `sh tools/mutation/run.sh` re-prints the count
for whatever commit it is on, and that number is the one to trust.

Getting there took two fixes, and the second is the one worth remembering.

**Three patterns had gone stale**, because they were anchored on code that was rewritten. `matmul-f64-acc`
still described the scalar accumulator `src/tensor.zig` had before the eight-lane rewrite; `norm-f32-acc`
pinned the justification comment above `rms`, which was rewritten when the 4096-wide row it argued from was
disowned by `norm_test.zig`, so it stopped matching a block of arithmetic that never changed; and
`loss-no-rowmax` described a loop reading `row[1..]` that gained a finiteness guard. All three are
re-anchored on the code as it stands. `norm-f32-acc` is deliberately anchored on two lines rather than
spanning the comment and the `rms` line, so a future comment edit cannot stale it again.

**The classifier was mis-scoring caught mutants as errors**, and that is why an earlier measurement
reported `caught 0 of 18` and looked like a dead harness. The `caught` branch matched Zig's summary line
with `run test [0-9]* pass, [1-9][0-9]* fail`, which requires `pass` and `fail` to be adjacent. They are
not, whenever anything is skipped, and on this tree something always is -- two tests are gated behind
`comptime model.cuda_attn`, so every line reads `224 pass, 2 skip, 2 fail`. The pattern therefore never
matched, and every mutation that a test actually caught fell through to the `error` branch. A log full of
`error: '...' failed` was being reported as broken machinery. The skip count is optional in the output and
the pattern now says so.

Both were worth fixing for the same reason: each one produced a number a reader would believe, and the
number was wrong.

| Mutation | Class | What it means |
|---|---|---|
| `clip-ge` | **equivalent, provably** | The two differ only when the norm is exactly equal to `max_grad_norm`, where the scale is exactly 1 and every element is multiplied by 1.0. A bit-for-bit identical run, not a close one. |
| `loss-no-rowmax` | **equivalent, algebraically** | `loss.forward` adds `max + log(sum(exp(z - max))) - target`. Seeding `max` with 0 instead of the row maximum shifts every term and they cancel: a max of `max - c` multiplies the sum by `e^c`, which adds `c` to its logarithm, and the explicit `-c` takes it straight back out. |
| `norm-reassociate` | **below the gates** | `v / rms * w` and `v * w / rms` differ only in rounding, and every assertion here is looser than that rounding. Catching it means pinning one f32 rounding rather than the function. |

A survivor table is a claim about the suite at one commit, and it is wrong the moment a test lands — which
is why these numbers come from the harness's own output rather than being kept in step by hand. All three
classes above are printed by the run itself, with the proof, in `run.sh`'s classification arm for each.


### Why it is a Zig program and not a sed line

`mutate.zig` holds the table and is the only thing that ever writes a mutant. It refuses to apply
a pattern that does not occur exactly the expected number of times. A `sed` line that has stopped
matching — because the source moved — writes an unmutated file, the suite passes, and the harness
reports a live defect as a coverage hole. A false survivor is worse than no harness, because it is
a number a reader would believe.

the classification in that table is an author's claim, not a measurement: the suite's exit code alone decides
caught or survived, and the class only labels the damage afterwards. `mutate list` prints the
names, and `mutate show <name>` prints the exact before and after for any one of them.

## Symbols

`sh tools/symbols.sh [readme] [root]` fails unless every `module.Symbol` row in a symbol table names
a declaration that is `pub` in `src/module.zig`. It exits 0 silently, or 1 listing every row that
lies. `zig build verify` runs it against `src/README.md`; the arguments exist so a falsified copy can
be checked without editing the real one.

| | |
|---|---|
| Direction | Documented-as-pub, actually private. The one drift that has bitten: a commit that made five `scale` helpers private also had to hand-edit five table rows, and this script run against that commit's parent names all five. Named by what it did rather than by SHA, because this repository's history was rewritten and the commit no longer resolves. |
| Module to file | The file stem. The tables already say `tensor.Tensor` and the file is `src/tensor.zig`, so there is no map to keep in sync and no list of the script's own to go stale. |
| Not checked | The other direction. A `pub` nobody documented is an omission and costs a reader nothing; `gradcheck` and `tokenizer` expose many, and listing them all would be churn for no safety. |
| Skipped | A first cell naming no module on disk, such as `init.arena`, which is a column of the allocator table rather than a symbol. A missing `src/<mod>.zig` is a skip, not a failure. |

It reads one shape — a first cell of exactly `` `module.Symbol` `` — because that is the only one the
tables use. A row written differently is not seen, which is a hole in the check rather than in the
README, and the failure mode is silence, so a reformat of the tables is the thing to re-run it after.

## Quoted tables

`sh tools/table-block.sh <tool-output> <readme> <block> <floor> <header>...` fails unless the tables a
tool prints are byte-for-byte the tables a README quotes between `<!-- <block>:begin -->` and
`<!-- <block>:end -->`. `zig build verify` runs it on the `scale-profile` block in `src/README.md`,
with a floor of 20 and the three header lines `zig build scale-profile` prints. That call site is the
only implementation, so protecting a second table is one more call in `build.zig` rather than a
second copy of fifty lines of shell. It exits 0 silently, or 1 with both markers named and a
`diff -u` of the two sides.

| | |
|---|---|
| Why a script | `sh tools/table-block-check.sh` has to break this check on purpose and watch it fail, and a gate that lives as a string inside `build.zig` can only be broken by running a build. |
| Floor | `cmp -s` exits 0 on two empty files, so an extraction below the floor is refused before the comparison. 20 against a real 29: adding a row does not require editing the call, and the failure names the block rather than reporting a bare line count. |
| Markers | Tested for before the block is extracted. A deleted marker extracts as empty and would otherwise fail with a diff that says nothing about which of the two happened. |
| Not checked | The prose around the block, and the length of the README side: an emptied block is still diffed against a non-empty tool extraction, so `cmp` fails on it without a floor of its own. Only the fenced lines between the markers are compared. |
| Byte identity | The check asserts equality, so the tool must write no timestamp, no address and no float whose formatting can drift. `scale-profile` writes none; a tool that did would need a projection in the README rather than a transcript. |

`sh tools/table-block-check.sh`, or `zig build table-block-check`, is the negative control: it breaks
that check three ways — a digit edited by hand, a marker deleted, a tool that prints nothing — plus
the case where the two sides match, which is what stops a harness that fails everything from reading
as a control. It also asserts that the missing-marker failure NAMES the missing marker, because with
a non-empty tool side a deleted marker and a drifted digit both exit 1 and the exit code alone cannot
tell them apart. Every case prints its output and its exit code. It is outside `verify` because it
prints: CI asserts that a passing `zig build verify` writes no bytes to either stream, and this
step's entire output is the evidence that the three broken tables were caught.

## Host state

`sh tools/host-clean.sh`, or `zig build host-check`, refuses when this host is in a state that would
contaminate a timing, and names which condition failed. Four: GPU utilisation, GPU memory, load
average, and a resident training or benchmark process. Thresholds are `MAX_GPU_UTIL` 5%,
`MAX_GPU_MEM_MIB` 1536, `MAX_LOAD` 8, and a reader who disagrees changes one number and says so in
the commit.

It exists because the generated-block check above has a limit worth stating plainly. **A table can
only be a verified block if the tool that prints it is deterministic.** `scale-profile` is: it is
Config arithmetic, it writes no timestamp, and two runs are byte-identical. `attn-bench`, `bench` and
`run-attn.sh` are not, because each prints a time, and a time is a property of the machine and the
hour. Comparing those blocks byte-for-byte would either be permanently red or, worse, pass by
rounding. So the rule "no benchmark number is ever typed by hand" holds where it can hold — the
arithmetic projections — and for the timings what holds instead is a committed transcript plus a
host that refuses to produce a contaminated one.

| | |
|---|---|
| Unreadable GPU | `nvidia-smi` returning nothing is a **refusal**, not a clean bill of health. `[ "" -gt 5 ]` is a comparison error, an error inside an `if` condition is false, and all four checks would pass on a card the script never read. |
| VRAM ceiling | Above this host's resting desktop baseline rather than at zero. A GUI session moves between roughly 250 and 1050 MiB depending on what is drawn, so a ceiling below that reports a clean host as busy whenever a window repaints. |
| Sibling match | The pattern names the training and benchmark binaries, not `zig build`, because this runs from inside a build step and would otherwise match its own parent. |
| Necessary, not sufficient | The `ctx256` bimodality in `outputs/bench/ctx256-sweep.csv` was measured on a host this script calls clean, and is still a 1.7384x gap with nothing between two tight clusters. A green `host-check` says the machine was idle, not that the number is reproducible; the sweep says which statistic to trust. |

## Step profile

`sh tools/step-profile.sh <ztransformer-train binary>`, or `zig build step-profile`, times each op
kind in a training step and checks two things of the result. One run gives: `backward` at 77% of a
step, `forward` at 19%, and every other op at most 1.21% individually.

It is a file rather than a `sh -c` blob in `build.zig` for the same reason the two scripts above are:
a gate has to be breakable on purpose and readable on its own. Every failure it raised while it was
an argv string was unreadable, because the whole script printed inside zig's `failed command` line
and its own stderr never got out. `sh -x tools/step-profile.sh <bin>` now shows all of it.

| | |
|---|---|
| Control 1 — the denominator | `SPAN > SUM`, compared as the two integers the table prints. **This check is close to an arithmetic identity and is documented as such**: the probes partition `[first probe, finish]`, so `bucketSum <= span` holds for any placement, count or naming, and the only thing that can fail it is `denominatorNs` returning `bucketSumNs`. It cannot detect a mis-placed probe. |
| Placement control — the one that bites | Asserts `loop` carries real time and `fetch` is smaller than `loop`. `loop` is the frees, the counters and the row append; `fetch` is the batch shuffle. Added because `fetch` once read **0.310% of a step while measuring the previous step's four `defer`s** — Zig runs `defer` at the scope's closing brace, after the last op and before the next one — and every other check passed throughout. A person reading the loop caught it; no gate did. |
| Expected refusal | A short profiled run *must* be refused: two rows cannot match the committed curve's five-row digest — 123 is its step count, not its row count — so `settleCsv` exits 1 with `error.CurveShape`. The predicate is "did the profiler print a table", not "did the curve match". |
| Control 2 — they move | Two runs at 16384 and 49152 bytes: identical per-step arithmetic, three times the steps, so the fixed per-RUN cost is amortised further and the per-step ops take a larger share. `backward` moves about 77% to 82%. `eval` moves 0.01 points and is **not** the mechanism. The 0.5-point threshold sits under the ~5 the arithmetic predicts. |
| Not in `verify` | These are times. A threshold on a time is a property of the machine and the hour. What is checked is a shape — the denominator and the movement — and both hold on a loaded host as much as an idle one. |
