# data

Training and validation text, vendored so the repo reproduces without a network fetch.

## Files

| File | Bytes | Description |
|---|---|---|
| `tinyshakespeare.txt` | 1,115,394 | Complete works of William Shakespeare, 40,000 lines, 202,651 words. |

## Provenance

Public domain. The complete works of Shakespeare carry no copyright. Retrieved from the
`karpathy/char-rnn` repository, which is distributed under the MIT licence and carries this
text as public domain.

    https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

    sha256  86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed

The file is committed rather than fetched at build time. A build that downloads its own input
cannot be reproduced offline, and a corpus that changes silently turns every loss number derived
from it into a lie.

## Use

The last 5% of the token stream is the validation split. The cut is positional, on the id
array `data.split` is handed, and it happens before any shuffle, so the same tokens always
yield the same split. It is a cut in the stream, not on a line boundary.

## Adding a corpus

Drop the file in this directory, add a row to the table above with its byte count, its source
URL, and its sha256, then commit it. Generated artifacts do not belong here. Tokenized output,
merged vocabularies, and checkpoints are build products and belong in `outputs/`.
