# outputs

Generated benchmark and parity artifacts. Every number here is produced by Zig or
CUDA code in this repository, never by hand.

Binary artifacts are gitignored (`.gitignore` covers `outputs/*.bin`), so a
checkout carries no executables. `zig build verify-publish` regenerates this
directory once that step exists; today the command does not exist.

The directory is currently empty because no benchmark has been generated. The
first artifact arrives with the training and parity phases.
