# tools

Programs that produce build artifacts. Each one runs on its own and writes a file under
`outputs/`.

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
