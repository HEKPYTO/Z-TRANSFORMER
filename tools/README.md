# tools

The parity harness: three files that compare this project's block against a Llama
reference, and the committed record of that comparison. Plus one shell script that holds the symbol
tables in `src/README.md` to the code they describe.

## Parity

`sh tools/removed/check.sh` compares this project's block against a Llama reference and
exits 0 when they agree, 1 when a tensor disagrees, and 2 when the oracle could not run at all. The
three are separate on purpose: a machine without torch has not failed a parity check.

| | |
|---|---|
| Reference | `reference-library==4.57.3`, `torch` 2.14.0, CPU, float32, attention `eager` |
| Shape | d_model 64, 2 layers, 4 heads over 2 kv heads of 16, ffn 256, vocab 256, ctx 512, batch 1 |
| Sweep | sequence lengths 1, 8 and 257, at seeds 7 and 8. Six runs, 156 tensor rows, 532 argmax comparisons |
| Last verdict | OK. 14 tensor kinds inside their gates, worst 2.1e-06 against a 2.0e-05 gate |
| Argmax | 532 of 532 rows pick the same token |
| Record | `report.csv` in this directory, one row per tensor per run. `check.sh` checks the report it just wrote against two digests `build.zig` holds, so the two rows above are checked on every run rather than true once. The portable one covers the row set, the shapes, the gates, the verdicts and the argmax count, and runs on every host. The byte digest covers `max_abs_delta` as well, and runs only where this host's oracle reports the same six version columns the committed file records. |

All three are pinned, to the versions the committed `report.csv` records in every one of its 156
rows, and each pin earns its place.

`reference-library` is the one that changes the answer. v5 moved `rope_theta` into `rope_parameters`, so
an unpinned oracle silently builds the reference with theta 10000 and disagrees with this model's
500000 on RoPE while looking like a numerics bug. `oracle.txt` does not rely on the pin alone: it
reads the theta the reference actually resolved and refuses to run if it is not the one this
repository's config states, so that trap is caught as a setup error whatever the version is.

`torch` and `numpy` do not change the answer; they change how much of the run is checked. Their pins
are what turn a different build into a `pip` message instead of a *quieter* gate: a torch whose
version string differs fails `removed-digest`'s environment predicate, and that host then gets the
projection check without the byte digest, so its `max_abs_delta` values are never compared against
the committed ones. A CUDA build spells its own version `2.14.0+cu130` and takes that path on its
own, pins or no pins. They are also what makes the `torch` version the `Reference` row above prints
the one `pip` was told to install rather than the one that happened to be on the machine.

The environment predicate is the report's own six version columns, because the report records nothing
else about the host that produced it. A host that reports the same six gets the byte digest, and a
byte difference there is one the report cannot explain — so it is a red, not a skip. That is a real
red, not a hypothetical one: a Linux host with a CPU-only the pinned build reports the committed columns
and still rounds differently, and this gate is what says so.

`_attn_implementation` defaults to `sdpa` in 4.57, which is a different kernel and a different
reduction order, so `eager` is forced and the resolved value is printed in the header. That trap
turns a setup error into a plausible-looking numerical one, which is the failure mode this harness
exists to remove.

Set up once, into a virtualenv inside the repository:

    python3 -m venv venv-removed
    venv-removed/bin/pip install -r tools/removed/requirements.txt
    sh tools/removed/check.sh

### Do the gates catch anything?

    sh tools/removed/sensitivity.sh

Perturbs the exported weight blob on one side only and requires the oracle to fail. It is the
command behind the sensitivity numbers in the root `README.md`, and it exists because those numbers
used to sit in a README with nothing that could produce them — a claim no reader could check and no
maintainer could falsify. Exit 0 means every perturbation was caught, 1 means one slipped through,
2 means the oracle could not run.

`PARITY_PYTHON` overrides the interpreter. The repository-local `venv-removed` wins over the system
one when both exist, so the pin is what actually runs.

### The three files

`oracle.txt` is the whole oracle: it reads the export, builds the reference, and writes
`report.csv`. It is never imported by the shipped binary and nothing in `src/` imports it.

`check.sh` is the entry point, and the only place the two halves meet. It also runs
`zig build removed-digest` over the report the oracle just wrote, which is the only thing that checks
that report against the digests `build.zig` holds. The step does both halves in one script and prints
which ran, on every invocation rather than only on a failure: a gate that quietly stopped checking
the bytes is indistinguishable from one that never did.

    $ zig build removed-digest
    removed-digest: projection OK; byte digest OK, environment matches the committed report.

What the projection is, and what it gives up, is the honest part of this. It keeps eight of the
report's fifteen columns — `kind`, `case`, `seq_len`, `tensor`, `rows`, `cols`, `gate`, `verdict` —
and drops `max_abs_delta` plus the six environment columns. So a flipped verdict, a changed gate, a
row added or removed, a changed argmax count and a report that is `sensitivity.sh`'s export instead
of a comparison all fail on any host, and the delta *magnitudes* are checked only on a host whose
versions match. A delta that grew from `5.96e-08` to `9.9e-06` — inside its gate, two orders of
magnitude worse — passes on a host whose torch build differs. The projection is a check on the
gates, not on the arithmetic behind them, and the byte digest is what covers the arithmetic.

`zig build verify` cannot do it, and the boundary is worth stating once: the report is a product of
the Python oracle, and AGENTS.md keeps Python out of the build graph. So the split is by what each
side can reach. `verify` checks everything reproducible from Zig alone — the corpus digest and the
committed loss curve, plus the weaker statement that the report carries no `FAIL` row, which is a
property of a green run rather than of one byte sequence. The two parity digests are asserted at the
moment the report exists, by the command that produced it, on a machine that already has the
virtualenv. A hash check sitting in `verify` would prove only that some bytes are committed, not that
a comparison reproduces them, which is the claim the record is for.

`report.csv` is the committed record, and it lives beside the file that writes it so the two cannot
drift apart. The export it describes is scratch, under `outputs/parity/`, and is not committed.

### What is Zig and what is Python

The export is `ztransformer parity`, which is Zig, and it runs with no Python anywhere in it. It
writes fixed weights from `model.initParams` at a stated seed, every intermediate the forward pass
can hand over, the token ids, and the shape. Python only reads those files back.

That split is the point rather than an accident of tooling. AGENTS.md requires that a committed
artifact never depend on a language the project does not otherwise need, and the weights here are a
build product: one seed has to reproduce them exactly, on a machine with only Zig. If the oracle
wrote them, a reader without torch could not regenerate them, and every number below would rest on
a dependency the repository does not claim.

The formats are raw little-endian f32 with a text index, not safetensors. There is one dtype, no
mmap on either side, and the oracle never calls `from_pretrained` because it builds a model and
copies tensors in, so it never wants a checkpoint. A JSON header, an 8-byte alignment rule and
external's key names would be more code than the rest of the harness, for a reader that does not
exist.

### What it does not check

Initialisation, because the harness pins the weights on both sides and never compares how they were
drawn. Batched forward, because this model has no batch axis and the harness runs batch 1. A KV
cache, quantized weights, or CUDA. The attention probabilities and the SwiGLU hidden state, which
`attention.forward` and `mlp.forward` reduce internally and never return; reaching either means
changing those two functions, which changes the numbers everything else is gating.

And sensitivity, which is worth stating precisely rather than in the abstract. The sweep shape was
chosen for legibility — `d_model` 64 over four heads of 16 keeps every tensor small enough to read.
Measured at that shape by `sh tools/removed/sensitivity.sh`: perturbing one element of `wq` by 1e-3
fails 21 gates, and scaling the whole 4096-element block by 1.001 still fails 27. A tenth of a percent is inside this harness's reach. What
does slip under the gates is narrower: dropping the f64 accumulator in `norm.zig` to f32 passes,
because at `d_model` 64 the drift that 512-element tests exist to catch is still below the 2e-6 gate
here. Sensitivity grows with the row width the gates are set against, not with the tensor count.

Both have to hold: all fourteen gates and the same token on every row. The gates are the sensitive
one. The smallest reference top1-top2 margin on this sweep is 8.5e-04 and the widest gate on the
logits is 2e-4, so a run that passes the gates cannot have flipped a token, and the argmax cannot
fail on its own. Its job is the diagnosis: for any row where the two disagree, the reference's own
top1-top2 margin is printed, so a near-tie is visibly a near-tie.

## Mutation

How good is the test suite, as a number somebody can check.

    sh tools/mutation/run.sh                      # 16 mutations, 8 minutes
    sh tools/mutation/run.sh matmul,rope          # a subset, matched as substrings
    MUTATION_MODE=debug sh tools/mutation/run.sh  # the same set in Debug, 39 minutes

Each argument is matched as a substring, so `matmul` is the whole matmul family and `rope` is the
rope mutation. That is the command a developer reaches for mid-run, and it used to exit 2: a `case`
pattern matches the whole word, every name in the table carries a suffix, and a name that does not
exist is not a useful thing for a selector to report.

It applies a deliberate defect to the source, runs the suite, and reports which tests caught it.
Every quality claim this repository makes about its own tests is a version of that question, and
until this existed the answer was a number in a chat message. `sensitivity.sh` asks it of the
parity gates; this asks it of the suite.

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

    zig build test                              # Debug: 161 s here
    zig build test -Doptimize=ReleaseFast        # ReleaseFast: 7 s here

The difference is the suite executing, not compiling: a mutation invalidates one file either way. A
Debug-only default is about 39 minutes for 16 mutations, at the 145 s a Debug mutation measured in
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
numbers is. At HEAD, 14 of 16 caught, by between 1 and 18 tests each:

| Mutation | Class | What it means |
|---|---|---|
| `clip-ge` | **equivalent, provably** | The two differ only when the norm equals the limit exactly, where the scale is `max / norm = 1` exactly and every element is multiplied by `1.0`. A bit-for-bit identical run, not a close one. No test can ever catch it, and one that tried would assert nothing. |
| `norm-reassociate` | **below the gates** | `v / rms * w` and `v * w / rms` differ only in rounding, and every assertion in the suite is looser than that rounding. Catching it means pinning one f32 rounding rather than the function. |

`matmul-f64-acc` and `norm-f32-acc` were on that list and are not any more. Both are caught now, by
one test each, so neither is a hole and neither is described here. A survivor table is a claim about
the suite at one commit, and it is wrong the moment a test lands, which is why the numbers above are
copied out of the harness's own output rather than kept in step by hand.

### Why it is a Zig program and not a sed line

`mutate.zig` holds the table and is the only thing that ever writes a mutant. It refuses to apply
a pattern that does not occur exactly the expected number of times. A `sed` line that has stopped
matching — because the source moved — writes an unmutated file, the suite passes, and the harness
reports a live defect as a coverage hole. A false survivor is worse than no harness, because it is
a number a reader would believe.

`class` in that table is an author's claim, not a measurement: the suite's exit code alone decides
caught or survived, and the class only labels the damage afterwards. `mutate list` prints the
names, and `mutate show <name>` prints the exact before and after for any one of them.

## Symbols

`sh tools/symbols.sh [readme] [root]` fails unless every `module.Symbol` row in a symbol table names
a declaration that is `pub` in `src/module.zig`. It exits 0 silently, or 1 listing every row that
lies. `zig build verify` runs it against `src/README.md`; the arguments exist so a falsified copy can
be checked without editing the real one.

| | |
|---|---|
| Direction | Documented-as-pub, actually private. The one drift that has bitten: commit `6c2a9c5` made five `scale` helpers private and had to hand-edit five table rows, and this script run against that commit's parent names all five. |
| Module to file | The file stem. The tables already say `tensor.Tensor` and the file is `src/tensor.zig`, so there is no map to keep in sync and no list of the script's own to go stale. |
| Not checked | The other direction. A `pub` nobody documented is an omission and costs a reader nothing; `gradcheck` and `tokenizer` expose many, and listing them all would be churn for no safety. |
| Skipped | A first cell naming no module on disk, such as `init.arena`, which is a column of the allocator table rather than a symbol. A missing `src/<mod>.zig` is a skip, not a failure. |

It reads one shape — a first cell of exactly `` `module.Symbol` `` — because that is the only one the
tables use. A row written differently is not seen, which is a hole in the check rather than in the
README, and the failure mode is silence, so a reformat of the tables is the thing to re-run it after.
