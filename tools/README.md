# tools

Programs that produce committed artifacts. Each one is runnable on its own and writes a file the
repository keeps.

## train_bpe.zig

Trains the byte-level BPE vocabulary on a corpus and writes it as JSON.

    zig build-exe --dep tokenizer -Mroot=tools/train_bpe.zig \
      -Mtokenizer=src/tokenizer.zig -femit-bin=/tmp/train_bpe
    /tmp/train_bpe data/tinyshakespeare.txt 20000 data/vocab.json

Arguments, in order: corpus path, number of merges requested, output path. The merge count is a
request, not a promise. Training stops when no pair occurs often enough to be worth merging, so a
small corpus yields fewer merges than asked for.

The two-module command above is not incidental. A module root in Zig 0.16 may not import a file
outside its own directory, so `zig build-exe tools/train_bpe.zig` fails outright. Declaring
`src/tokenizer.zig` as a second module is the supported arrangement, and duplicating the tokenizer
into this directory to avoid it would give us two copies of the vocabulary format and a silent
divergence between them.

## Why a Zig tool and not a script

The vocabulary is a committed artifact. If a script in another language were the only way to produce
it, the repository would ship a file that the reader cannot regenerate with the toolchain the
repository claims to require. The corpus is public domain, the training is deterministic given a
seed, and the merge order is written to the file, so re-running this reproduces the artifact
byte for byte.

## Adding a tool

Read the corpus path from `argv` and return a non-zero exit code on failure. Do not print progress
to stdout if a later step will parse stdout. Do not write outside the repository, and do not leave a
compiled binary in the tree.
