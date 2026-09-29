# outputs

Everything the repository generates lands here, and one file is committed: `loss.csv`, the measured
loss curve of a real run, because a repository that claims reproducible numbers has to carry at
least one of them. The rest is ignored by `.gitignore`: `outputs/*.bin` and `outputs/parity/`, the
last being the largest class of generated file here. A loss curve is the run's output and cannot be
rebuilt without paying for the run, so it ships; the parity export is scratch that the committed
`tools/removed/report.csv` already stands in for. Nothing here is ever written by hand.

| File | Committed | Written by | What it is |
|---|---|---|---|
| `loss.csv` | yes | `zig build train` | The loss curve of one training run. Header `step,train_loss,val_loss,lr`, then one line per logged step, six decimals on the two losses and eight on the rate. The file is truncated first, so a shorter run leaves no tail of the previous curve behind. |
| `parity/` | no | `ztransformer parity`, see `tools/README.md` | Scratch for the external comparison: the exported weights, an index, the config line, and the per-case inputs. Regenerated on every run of `sh tools/removed/check.sh` and never committed, because the committed record of that comparison is `tools/removed/report.csv`. It sits beside its writer rather than here so it cannot drift away from it. |


`train_loss` on a row is the mean over the logged steps of the epoch it closes, not the loss of the
step named in `step`. `val_loss` is measured once per epoch, on held-out tokens the training batcher
never touches, and is written onto the last row of that epoch only, so every earlier row carries
`0.000000`. That is the shape of the measurement, not a missing one.

The committed curve is the default `zig build train` run: 123 windows, one epoch, 64 KiB of corpus.
The root `README.md` quotes the two final numbers out of it.

Nothing else is here yet. There is no benchmark file and no checkpoint: `train.run` returns the
trained weights in its `Result` and nothing writes them to disk, so a run that ends leaves a curve
and no model. The parity report is a real result, but it is a record of a comparison rather than a
product of this directory, so it is committed under `tools/removed/`.
