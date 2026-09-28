# tools

Programs that produce build artifacts. Each one runs on its own and writes a file under
`outputs/`, and the parity harness below writes its one committed record here.

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
Measured at that shape: scaling one element of `wq` by 1e-3 fails 51 gates, and scaling all 4096
elements of `wq` by 1.001 still fails 16. A tenth of a percent is inside this harness's reach. What
does slip under the gates is narrower: dropping the f64 accumulator in `norm.zig` to f32 passes,
because at `d_model` 64 the drift that 512-element tests exist to catch is still below the 2e-6 gate
here. Sensitivity grows with the row width the gates are set against, not with the tensor count.

Both have to hold: all fourteen gates and the same token on every row. The gates are the sensitive
one. The smallest reference top1-top2 margin on this sweep is 8.5e-04 and the widest gate on the
logits is 2e-4, so a run that passes the gates cannot have flipped a token, and the argmax cannot
fail on its own. Its job is the diagnosis: for any row where the two disagree, the reference's own
top1-top2 margin is printed, so a near-tie is visibly a near-tie.

## train_bpe.zig

Trains the byte-level BPE vocabulary on a corpus and writes it as JSON.

    zig build-exe --dep tokenizer -Mroot=tools/train_bpe.zig \
      -Mtokenizer=src/tokenizer.zig -femit-bin=tools/train_bpe
    ./tools/train_bpe data/tinyshakespeare.txt outputs/vocab.json
    ./tools/train_bpe data/tinyshakespeare.txt 20000 outputs/vocab.json

Arguments are the corpus path, then optionally the merge count, then the output path. The merge
count defaults to 200. The output path is relative to the repository root, and its directory is
created if it does not exist. An absolute path is refused: a sub path is resolved against the
working directory, so one that starts with a slash lands somewhere other than where it reads.

The merge count is a request, not a promise. Training stops when no pair occurs often enough to be
worth merging, so a small corpus yields fewer merges than asked for.

### Cost

`train` is O(len × n_merges) and rescans the corpus once per merge, so wall time is linear in the
merge count. Measured with a debug build on this machine against `data/tinyshakespeare.txt`
(1,115,394 bytes):

| Merges | Wall time | Peak RSS | Vocabulary file |
|---|---|---|---|
| 200 (default) | 59 s | 18.0 MiB | 4.9 KB |
| 1000 | 259 s | 18.8 MiB | 24.6 KB |
| 20000 | about 1.4 h, extrapolated | not measured | not measured |

20000 is a hundred times the work for a vocabulary that only pays off once a model is trained on
it, so it is opt-in rather than the default. The extrapolated row is scaled from the 1000-merge
run, not measured; the row is here so the cost is visible rather than discovered. Wall time also
depends on the rest of the machine: the same 200-merge run took 59 s alone and 110 s while five
other builds were competing for cores.

### Memory

The tokenizer runs on the general purpose allocator. The arena keeps only the corpus and the
encoded ids, the two allocations that live for the whole run. On an arena `deinit` is a no-op, so
each pass leaves its id buffers and the grown pair-count table resident for the rest of the run:
26.4 MiB peak at 200 merges against 18.0 MiB with the gpa, which also reports a leak rather than
hiding one.

### Self check

After writing the file the tool loads it back, encodes a 4 KiB slice of the corpus with the loaded
vocabulary, decodes it, and compares the result to the original bytes. A non-zero exit means the
artifact does not rebuild the text it was trained on. The check lives in the tool rather than in a
script beside it, so the proof ships with the artifact.

## The two-module build is not incidental

A module root in Zig 0.16 may not import a file outside its own directory, so
`zig build-exe tools/train_bpe.zig` fails outright and `-femit-bin` does not rescue it. Declaring
`src/tokenizer.zig` as a second module is the supported arrangement, and duplicating the tokenizer
into this directory to avoid it would give us two copies of the vocabulary format and a silent
divergence between them. The binary therefore lands in the repository as `tools/train_bpe`, which
is gitignored, rather than somewhere outside the tree.

## Why a Zig tool and not a script

The vocabulary is a build product. If a script in another language were the only way to produce
it, the repository would ship a file the reader cannot regenerate with the toolchain the
repository claims to require. The corpus is public domain, no seed is involved, and the merge
order is fully determined by the corpus, so re-running this reproduces the file byte for byte.

## Adding a tool

Read the corpus path from `argv` and return a non-zero exit code on failure. Do not print progress
to stdout if a later step will parse stdout. Do not write outside the repository, and do not leave a
compiled binary in the tree. A build target inside the tree is fine when it carries an ignore
entry, which is the case for `tools/train_bpe`.
