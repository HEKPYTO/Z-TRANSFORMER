# tools

The parity harness: three files that compare this project's block against a Llama
reference, and the committed record of that comparison.

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
| Record | `report.csv` in this directory, one row per tensor per run |

The pinned versions are not decoration. `reference-library` v5 moved `rope_theta` into
`rope_parameters`, so an unpinned oracle silently builds the reference with theta 10000 and
disagrees with this model's 500000 on RoPE while looking like a numerics bug. `_attn_implementation`
defaults to `sdpa` in 4.57, which is a different kernel and a different reduction order, so `eager`
is forced and the resolved value is printed in the header. Both traps turn a setup error into a
plausible-looking numerical one, which is the failure mode this harness exists to remove.

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

`check.sh` is the entry point, and the only place the two halves meet.

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
    sh tools/mutation/run.sh matmul,rope          # a subset, by name or glob
    MUTATION_MODE=debug sh tools/mutation/run.sh  # the same set in Debug, 39 minutes

It applies a deliberate defect to the source, runs the suite, and reports which tests caught it.
Every quality claim this repository makes about its own tests is a version of that question, and
until this existed the answer was a number in a chat message. `sensitivity.sh` asks it of the
parity gates; this asks it of the suite.

Exit 0 whenever the run completed and every mutation got a verdict, survivors included. A coverage
number is a measurement, not a pass/fail gate: a suite is not imperfect, it is measured, and a
command that went red the first time anyone ran it would be deleted rather than fixed. Exit 1 means
the measurement is untrustworthy — a mutation whose pattern stopped matching, or one that does not
compile, so a number was reported for code that never ran. Exit 2 means the harness could not run.

### The two traps a hand-run falls into

**It never mutates this tree.** The mutants go into a `git worktree` pinned to HEAD, removed by a
trap on every exit path including SIGINT. The run ends by printing its own evidence: the worktree is
no longer registered, and a SHA-256 of all of `src/` matches the one taken before the first
mutation. A harness that edits the checkout it runs from leaves a mutant behind when it crashes, and
a repository with a mutant in it is worse than one with no harness — the next `zig build` reports a
failure nobody caused, on a line nobody wrote.

**It runs `zig fmt` on the mutant before the suite.** `verify` gates both of its test binaries on
`zig fmt --check`, so an unformatted mutant is rejected by the formatter before a single test
executes. A naive harness records that as "caught by the suite" when the suite was never entered,
and every such mutation reads as coverage that does not exist.

### ReleaseFast, and what that costs

The Debug suite takes 145 s and ReleaseFast 33 s on this machine. The difference is the suite
executing, not compiling: a mutation invalidates one file either way. A Debug-only default is 39
minutes for 16 mutations, which is a number nobody runs, and a default nobody runs measures nothing.
ReleaseFast is also the mode the `train` binary ships in.

The cost is real and is stated rather than hidden: a defect only Debug's safety checks trip is
invisible to the default run, so Debug-only catches are not counted. All three survivors below were
re-run through this same script with `MUTATION_MODE=debug` and all three survive Debug too, so on
this set the choice cost nothing measured. `MUTATION_MODE=verify` runs both configurations.

### Three verdicts, and a class for every survivor

`caught` is a failing test. `BUILD` is a mutant that does not compile, which says the mutation was
badly written and nothing about the suite, so it is reported separately and never folded into the
headline. `SURVIVED` is the rest, and each survivor is classified in the output rather than counted,
because a survivor that is behaviourally identical is not a hole in the suite and one that moves the
numbers is. At HEAD, 13 of 16 caught, by between 1 and 17 tests each:

| Mutation | Class | What it means |
|---|---|---|
| `matmul-f64-acc` | **hole** | Not equivalent, and the divergence is measured rather than argued: reducing k weights at the init scale in f32 rather than f64 diverges by 5.1e-6 relative at k=64, 2.7e-5 at the ffn width of 256, and 1.2e-4 at k=4096. No matmul test here multiplies a tall and wide matrix, so the f32 accumulator in `tensor.zig` is an unverified choice rather than a tested one. |
| `clip-ge` | **equivalent, provably** | The two differ only when the norm equals the limit exactly, where the scale is `max / norm = 1` exactly and every element is multiplied by `1.0`. A bit-for-bit identical run, not a close one. No test can ever catch it, and one that tried would assert nothing. |
| `norm-reassociate` | **below the gates** | `v / rms * w` and `v * w / rms` differ only in rounding, and every assertion in the suite is looser than that rounding. Catching it means pinning one f32 rounding rather than the function. |

### Why it is a Zig program and not a sed line

`mutate.zig` holds the table and is the only thing that ever writes a mutant. It refuses to apply
a pattern that does not occur exactly the expected number of times. A `sed` line that has stopped
matching — because the source moved — writes an unmutated file, the suite passes, and the harness
reports a live defect as a coverage hole. A false survivor is worse than no harness, because it is
a number a reader would believe.

`class` in that table is an author's claim, not a measurement: the suite's exit code alone decides
caught or survived, and the class only labels the damage afterwards. `mutate list` prints the
names, and `mutate show <name>` prints the exact before and after for any one of them.
