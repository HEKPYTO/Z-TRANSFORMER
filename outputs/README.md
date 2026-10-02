# outputs

Everything the repository generates lands here, and two files are committed: `loss.csv`, the
measured loss curve of a real run, and `bench/ctx256-sweep.csv`, the ten samples that show the
attention benchmark's CPU column is bimodal at one shape. A repository that claims reproducible
numbers has to carry the measurements behind them, and those two are measurements; the rest is
ignored by `.gitignore`: `outputs/*.bin` and `outputs/parity/`, the last being the largest class
of generated file here. A loss curve is the run's output and cannot be rebuilt without paying for
the run, so it ships; the parity export is scratch that the committed `tools/removed/report.csv`
already stands in for. Nothing here is ever written by hand.

| File | Committed | Written by | What it is |
|---|---|---|---|
| `loss.csv` | yes | `zig build train` | The loss curve of one training run. Header `step,train_loss,val_loss,lr`, then one line per logged step, six decimals on the two losses and eight on the rate. A `val_loss` that was never measured is an **empty field** and nothing else, so a row reads `24,6.489256,,0.29888502`; a measured zero is `0.000000` and a reader can tell the two apart. `zig build verify` checks it against the digest `build.zig` holds, and a `train` run replaces it only by matching those bytes. |
| `loss.pending.csv` | no | `zig build train` | The curve the last run produced, before anything has compared it to the committed one. Renamed onto `loss.csv` and deleted when the bytes match, which is the ordinary outcome. When they do not, it is kept and the run **fails**: a difference is a failure to reproduce the committed claim, and the default has to be the loud one. `ZTRANSFORMER_ACCEPT_LOSS_DIFFERENCE=1` downgrades that to a reported exit 0 — and still does not promote, because acknowledging a difference and changing the claim are different acts. Ignored, so a run that differs leaves the tree the way `git status` reports it. |
| `parity/` | no | `ztransformer parity`, see `tools/README.md` | Scratch for the external comparison: the exported weights, an index, the config line, and the per-case inputs. Regenerated on every run of `sh tools/removed/check.sh` and never committed, because the committed record of that comparison is `tools/removed/report.csv`. It sits beside its writer rather than here so it cannot drift away from it. |
| `bench/ctx256-sweep.csv` | yes | ten runs of `src/cuda/attn_twin.zig` | The distribution behind the min-of-3 rule, and the reason that rule exists rather than a convention: `ctx256`'s CPU column lands in one of two clusters 1.7384x apart with no sample between them, and 70% of single runs land in the higher one. `src/cuda/README.md` reads it out. |


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

The committed curve is the default `zig build train` run: 123 windows, one epoch, 64 KiB of corpus.
The root `README.md` quotes the two final numbers out of it.

Three gates cover it, and they catch different things. `zig build verify` checks the file against the
digest in `src/main.zig`, so it catches **an edited curve**. `zig build determinism`, also inside
`verify`, runs two short training passes and requires byte-identical curves, so it catches **code
that no longer reproduces its own output**. `zig build train` re-runs the full training and refuses
to promote a curve that differs, so it catches **a changed codebase** at the shipped shape.

The middle one is what this paragraph used to concede was missing. `verify` hashed the committed
file, which is an assertion about the file and not about the program that wrote it, so nothing in the
graph re-derived the curve and a build producing different bytes from the same source stayed green.
It is **host-relative and deliberately not cross-machine** -- a different libm gives different bytes
legitimately -- so it never asserts `csv_sha256`; it asserts that two runs on this host agree with
each other, which is what the committed curve's reproducibility claim actually rests on.

It is cheap because it is short: `ZTRANSFORMER_CORPUS_BYTES=16384` is the smallest corpus that still
yields a validation window, since `data.split` keeps 5% and one window is `ctx` 256 tokens -- 8 KiB
leaves fewer than 256 validation tokens and the run refuses with `error: EmptyValidation`. That is
30 steps, about a second a pass, against seven minutes for two full runs. Reproducibility is a
property of the arithmetic and not of the run length, because trajectory chaos needs many steps to
amplify. The variable only ever shrinks the corpus, so a check cannot be made slow by accident.

Both in-graph gates are **silent on success**, because CI asserts a passing `zig build verify` writes
no bytes to either stream. `zig build train` stays a manual command on one host, for the reason above:
a CI gate on a curve digest would be permanently red on any host with a different libm. A run that
does produce the committed curve finds `loss.pending.csv` byte-identical, promotes it, and leaves
nothing behind.

There is still no checkpoint: `train.run` returns the trained weights in its `Result` and nothing
writes them to disk, so a training run that ends leaves a curve and no model. `bench/` is the one
benchmark measurement committed here, and it is committed for the reason the other numbers in this
repository are not: it is the evidence that a measurement is unreliable in a way a reader cannot
see from a single sample. It records ten runs of one shape rather than a timing of many, because
the distribution is the finding and the fastest of the ten is not. The parity report is a real
result, but it is a record of a comparison rather than a product of this directory, so it is
committed under `tools/removed/`.
