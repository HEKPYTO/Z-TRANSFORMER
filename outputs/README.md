# outputs

Everything the repository generates lands here, and two files are committed: `loss.csv`, the
measured loss curve of a real run, and `bench/ctx256-sweep.csv`, the ten samples that show the
attention benchmark's CPU column is bimodal at one shape. A repository that claims reproducible
numbers has to carry the measurements behind them, and those two are measurements; the rest is
numbers has to carry the measurements behind them, and that is one; the rest is ignored by
`.gitignore`: `outputs/*.bin`, scratch that nothing
reads any more. A loss curve is the run's output and cannot be rebuilt without paying for
the run, so it ships. Nothing here is ever written by hand.

| File | Committed | Written by | What it is |
|---|---|---|---|
| `loss.csv` | yes | `zig build train` | The loss curve of one training run. Header `step,train_loss,val_loss,lr`, then one line per logged step, six decimals on the two losses and eight on the rate. A `val_loss` that was never measured is an **empty field** and nothing else, so a row reads `24,6.489256,,0.29888502`; a measured zero is `0.000000` and a reader can tell the two apart. `zig build verify` checks it against the digest `src/main.zig` owns and `build.zig` reads, and a `train` run replaces it only by matching those bytes. |
| `loss.pending.csv` | no | `zig build train` | The curve the last run produced, before anything has compared it to the committed one. Renamed onto `loss.csv` and deleted when the bytes match, which is the ordinary outcome. When they do not, it is kept and the run **fails**: a difference is a failure to reproduce the committed claim, and the default has to be the loud one. `ZTRANSFORMER_ACCEPT_LOSS_DIFFERENCE=1` downgrades that to a reported exit 0 — and still does not promote, because acknowledging a difference and changing the claim are different acts. Ignored, so a run that differs leaves the tree the way `git status` reports it. |
| `bench/ctx256-sweep.csv` | yes | ten runs of `src/cuda/attn_twin.zig` | The distribution behind the min-of-3 rule, and the reason that rule exists rather than a convention: `ctx256`'s CPU column lands in one of two clusters 1.7384x apart with no sample between them, and 70% of single runs land in the higher one. `src/cuda/README.md` reads it out. |
| `bench/run-attn.log` | when produced | `sh src/cuda/run-attn.sh 2>&1 \| tee outputs/bench/run-attn.log` on the CUDA host | The full parity + speedup transcript the root README table is derived from. Committed only from a run on the host of record (`host-check` clean, GPU idle); this host has no GPU and produces none. |


`train_loss` on a row is the mean over the logged steps of the epoch it closes, not the loss of the
step named in `step`. `val_loss` is measured once per epoch, on held-out tokens the training batcher
never touches, and is written onto the last row of that epoch only. Every earlier row carries an
**empty field**: the column is optional in `train.Row` and an empty one means no measurement has
been taken, which is a different statement from a measurement of zero. The file used to print
`0.000000` on those rows instead, and a reader could not tell "the validation loss was exactly
zero" from "nothing has been measured yet" — the same bytes said both. `0.000000` in that column
now means the run measured it and it was zero.

`train.divergence` follows the same distinction. Two cells that were both unmeasured agree. A cell
that was measured on one curve and not on the other is refused with `error.CurveShape` rather than
compared: there is no number on one side of the subtraction, and treating the missing one as zero
would report two different curves as agreeing.

The committed curve is the default `zig build train` run: 123 windows, one epoch, 64 KiB of corpus,
ending at train loss 5.596625 and validation loss 5.155396, the last row of the file, measured on
held-out tokens the training batcher never touches. A smoke run showing the loop completes and the
loss falls; not a benchmark, not a throughput figure, not a model-quality claim. There is no
inference benchmark. The committed shape, as printed:

```
6.5 |*                                                                  6.489 step 24
6.0 |    *                                                            6.062 step 49
5.8 |         *                                                      5.814 step 74
5.7 |              *                                               5.687 step 99
5.6 |                   *                                          5.597 step 122
    +--------------------------------------------------------------------
      0        25        50        75       100       125   step
```

Three gates cover it, and they catch different things. `zig build verify` checks the file against the
digest in `src/main.zig`, so it catches **an edited curve**. `zig build determinism`, also inside
`verify`, runs two short training passes and requires byte-identical curves, so it catches **code
that no longer reproduces its own output**. Both passes have to HAPPEN: the step clears the pending
curve before the second one, so a pass that fails and writes nothing leaves the `cp` failing rather
than handing `cmp` the first pass's leftovers to compare against themselves. It did not do that
once, and a second pass that never ran read as two that agreed. `zig build train` re-runs the full
training and refuses to promote a curve that differs, so it catches **a changed codebase** at the
shipped shape.

The middle one is what this paragraph used to concede was missing. `verify` hashed the committed
file, which is an assertion about the file and not about the program that wrote it, so nothing in the
graph re-derived the curve and a build producing different bytes from the same source stayed green.
It is **host-relative and deliberately not cross-machine** -- a different libm gives different bytes
legitimately -- so it never asserts `csv_sha256`; it asserts that two runs on this host agree with
each other, which is what the committed curve's reproducibility claim actually rests on.

**Where the two hosts part company, measured over the whole curve rather than assumed.** The shipped
`log_every` is 25, so the committed curve's first row is step 24 -- by which point a genuine defect and
an f32 epsilon grown by chaos look identical, and nothing in the file can say which it was. Re-run with
`log_every` 1 on both hosts settles it over all 123 steps. Both runs were taken from the same tree
-- the one the numbers come from, not a later document commit. They are not attributed to a commit
SHA because this repository's history was rewritten after they were taken, so the one they were
taken at no longer resolves. Each ran from a detached worktree with the committed file untouched,
and both refused to promote as they should:

| | result |
|---|---|
| Steps **0, 1, 2** | **bitwise identical** -- 6.123955, 6.080079, 5.996188 on both |
| First divergent step | **3** -- 6.083721 against 6.083728, a difference of 7e-06 |
| Peak absolute difference | **0.084131**, at step 16 |
| Peak as a share of the curve's largest loss | 1.192% |
| Difference at the final step 122 | 0.003170 -- **decayed** from the peak |

The shape is the finding, not the size. The difference rises, peaks at step 16 and then **falls** by
more than an order of magnitude. A wrong gradient or a systematic libm bias would grow monotonically;
a perturbation amplified and then damped by the optimiser does what this does. Peak-then-decay is the
signature, and a single end-of-curve number would have hidden it.

The two hosts, so the result is checkable rather than asserted:

| | macOS | Linux |
|---|---|---|
| Zig | 0.16.0 | 0.16.0 |
| CPU | Apple M2 | Intel i9-13900K |
| libc | Apple, macOS 27.0.1 (libSystem) | glibc 2.43 |

**They differ in CPU as well as libc**, so this pair alone cannot attribute the divergence to libc. What
makes libc the plausible driver is the separate result: two different-CPU glibc hosts agree byte for
byte. Same Zig on both, so the toolchain is not a variable either.

Three bitwise steps and then amplification is an arithmetic epsilon, not a defect that was present
from the start: a wrong gradient or a wrong schedule would have separated the hosts at step 0. The peak
is the same order as the **0.087865** this repository already records for the CUDA-versus-CPU arm, which
is the comparison it is directly comparable to. This is what "host-relative" rests on, and until it was
measured it was an assertion.

One unexplained observation, recorded so the next person to hit it does not assume they found it
first. On `2026-10-03` a `zig build verify` on the Linux host of record returned **exit 1 with 5120 bytes of
output**, and the same command returned **exit 0 with 0 bytes** immediately after with no code
change between them. It has not reproduced since: three consecutive `determinism` then `verify`
sequences, the failing invocation replayed verbatim, and five deliberate races of `determinism`
against `peak-rss` -- two steps that both invoke `train` and both write `outputs/loss.pending.csv`
with no ordering declared between them -- all came back clean. **The cause is unknown**, so nothing
here claims one, and no locking was added for a failure that has never been produced. The shared
pending file is the obvious suspect and is stated as one.

The determinism step is cheap because it is short: `ZTRANSFORMER_CORPUS_BYTES=16384` is the
smallest corpus that still yields a validation window, since `data.split` keeps 5% and one
window is `ctx` 256 tokens -- 8 KiB leaves fewer than 256 validation tokens and the run refuses
with `error: EmptyValidation`. That is 30 steps, about a second a pass, against seven minutes
for two full runs. Reproducibility is a property of the arithmetic and not of the run length,
because trajectory chaos needs many steps to amplify. The variable only ever shrinks the corpus, so a check cannot be made slow by accident.

Both in-graph gates are **silent on success**, because CI asserts a passing `zig build verify` writes
no bytes to either stream. `zig build train` stays a manual command on one host, for the reason above:
a CI gate on a curve digest would be permanently red on any host with a different libm. A run that
does produce the committed curve finds `loss.pending.csv` byte-identical, promotes it, and leaves
nothing behind.

Every run also writes `checkpoint.bin` (gitignored, `outputs/*.bin`): the raw LE f32 weights `train.run` returned, in `train.flatten` order behind the `model.Config` they were built from. `zig build infer` reads it back. `bench/` is the one
benchmark measurement committed here, and it is committed for the reason the other numbers in this
repository are not: it is the evidence that a measurement is unreliable in a way a reader cannot
nothing reads them and nothing grades them, so they are not recorded here.
