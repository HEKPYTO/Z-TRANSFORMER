# outputs

Everything the repository generates lands here, and one file is committed: `loss.csv`, the measured
loss curve of a real run, because a repository that claims reproducible numbers has to carry at
least one of them. The rest is ignored by `.gitignore`: `outputs/*.bin` and `outputs/parity/`, the
last being the largest class of generated file here. A loss curve is the run's output and cannot be
rebuilt without paying for the run, so it ships; the parity export is scratch that the committed
`tools/removed/report.csv` already stands in for. Nothing here is ever written by hand.

| File | Committed | Written by | What it is |
|---|---|---|---|
| `loss.csv` | yes | `zig build train` | The loss curve of one training run. Header `step,train_loss,val_loss,lr`, then one line per logged step, six decimals on the two losses and eight on the rate. `zig build verify` checks it against the digest `build.zig` holds, and a `train` run replaces it only by matching those bytes. |
| `loss.pending.csv` | no | `zig build train` | The curve the last run produced, before anything has compared it to the committed one. Renamed onto `loss.csv` and deleted when the bytes match, which is the ordinary outcome. When they do not, it is kept and the run **fails**: a difference is a failure to reproduce the committed claim, and the default has to be the loud one. `ZTRANSFORMER_ACCEPT_LOSS_DIFFERENCE=1` downgrades that to a reported exit 0 — and still does not promote, because acknowledging a difference and changing the claim are different acts. Ignored, so a run that differs leaves the tree the way `git status` reports it. |
| `parity/` | no | `ztransformer parity`, see `tools/README.md` | Scratch for the external comparison: the exported weights, an index, the config line, and the per-case inputs. Regenerated on every run of `sh tools/removed/check.sh` and never committed, because the committed record of that comparison is `tools/removed/report.csv`. It sits beside its writer rather than here so it cannot drift away from it. |


`train_loss` on a row is the mean over the logged steps of the epoch it closes, not the loss of the
step named in `step`. `val_loss` is measured once per epoch, on held-out tokens the training batcher
never touches, and is written onto the last row of that epoch only, so every earlier row carries
`0.000000`. That is the shape of the measurement, not a missing one.

The committed curve is the default `zig build train` run: 123 windows, one epoch, 64 KiB of corpus.
The root `README.md` quotes the two final numbers out of it.

Two gates cover it, and they catch different things. `zig build verify` checks the file against the
digest in `src/main.zig`, so it catches **an edited curve**. `zig build train` re-runs the training
and refuses to promote a curve that differs, so it catches **a changed codebase**. Only the first is
in `verify` and therefore in CI; the second is a manual command on one host, because a host with a
different libm produces different bytes legitimately and a CI gate on it would be permanently red
for a reason that has nothing to do with the code. An earlier version of this paragraph said `verify`
made a stale run "a failing gate rather than a stale paragraph", which is true of an edited file and
false of changed arithmetic, and the distinction is the whole point. A run that does produce it finds `loss.pending.csv` byte-identical, promotes it, and
leaves nothing behind.

Nothing else is here yet. There is no benchmark file and no checkpoint: `train.run` returns the
trained weights in its `Result` and nothing writes them to disk, so a run that ends leaves a curve
and no model. The parity report is a real result, but it is a record of a comparison rather than a
product of this directory, so it is committed under `tools/removed/`.
