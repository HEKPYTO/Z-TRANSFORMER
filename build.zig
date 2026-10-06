const std = @import("std");

comptime {
    // The README names 0.16.0 and CI pins it, but neither stops a developer
    // building on a different one. Nothing in the standard library's build API
    // reports the version — `Graph` carries only `zig_exe` — so this is the only
    // place it can be checked, and unchecked it is a confusing std-API error on
    // the way in rather than a sentence that says what to install.
    const v = @import("builtin").zig_version_string;
    if (!std.mem.eql(u8, v, "0.16.0"))
        @compileError("Z-TRANSFORMER requires Zig 0.16.0; this is " ++ v);
}

/// The library source, read for the one string `verify` asserts on. Taken from
/// the same file the binary is built from, so a version bump cannot leave a
/// stale expectation behind here.
const lib_source = @import("src/lib.zig");

/// The binary's source, read for the one digest the `train` step and the
/// `verify` gate have to agree on. Imported rather than repeated, so the two
/// cannot drift; see `loss_csv_sha256` below for why that matters here and not
/// only in principle.
const main_source = @import("src/main.zig");

/// The model's source, read for ONE constant: whether the attention forward runs
/// on `src/cuda/attn_kernels.cu` or on the CPU. The build script has to know,
/// because the object that makes a CUDA call resolvable is linked by two named
/// steps and by nothing else, and both of those have to refuse rather than build
/// something that links an object no code references. Imported rather than
/// parsed out of the text, for the reason `main_source` above is: two spellings
/// of one flag is a flag that will be wrong in one of them.
const model_source = @import("src/model.zig");

/// The corpus and the digest `data/README.md` documents for it. A corpus that
/// changes silently turns every number derived from it into a lie.
const corpus_path = "data/tinyshakespeare.txt";
const corpus_sha256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed";

/// `outputs/loss.csv`: the loss curve of the committed training run, produced
/// by `zig build train` — seed 7, 200 BPE merges, a 64 KiB corpus prefix, 123
/// windows, one epoch — through the `train` step's own ReleaseFast binary.
///
/// The digest is claimed under the criterion `src/README.md` states in
/// Reproducibility, which is one seed, one build configuration, one host, and
/// that is the whole of it. Measured on the host that committed it: two runs at
/// seed 7 in ReleaseFast produced these exact bytes. It is not a claim across
/// hosts or across optimization levels, and `src/README.md` says why rather than
/// leaving it a mystery — `@exp`, `@sqrt` and `@cos` resolve to the platform
/// libm, and **optimization level is not currently a source of divergence**: a Debug build of this
/// source reproduces the committed ReleaseFast curve byte for byte on this toolchain, measured, so the
/// libm is the thing that has produced different bytes. So a reader on another host whose run
/// produced different bytes has found a different libm, not a broken build, and `zig build train`
/// reports it in those words and leaves this file alone rather than replacing it.
///
/// `zig build verify` checks the file, and that proves exactly one thing: the
/// curve in the tree is the curve this repository documents. The other half —
/// that the run can produce those bytes — is `zig build train`, which writes a
/// pending file, promotes it only on a byte-for-byte match, and fails on a
/// mismatch rather than assuming the difference is benign.
///
/// The digest itself is not spelled here. It is read from
/// `src/main.zig:csv_sha256`, which owns it, because two copies of one digest
/// with the tool's own recovery instructions pointing at one of them is a wedge:
/// a reader who updates the copy `verify` checks has left `train` unable to
/// promote forever, with nothing in the output saying so. One literal, one place
/// to change, and the two callers cannot disagree.
const loss_csv_path = "outputs/loss.csv";
const loss_csv_sha256 = main_source.csv_sha256;

/// What `zig build train` is allowed to peak at, in bytes, as macOS
/// `/usr/bin/time -l` reports the ReleaseFast binary's own maximum resident set
/// size. The gate is `zig build peak-rss`, and `verify` runs it.
///
/// This number is here because nothing measured peak memory for this
/// repository's entire history. The loss curve was gated on a byte
/// digest, and a defect that retained every one of 123 steps' working sets in
/// an arena whose `free` is a no-op shipped anyway, because the
/// documentation of it was a table in `src/README.md` and no command behind
/// the table. A measurement with no gate is a note.
///
/// 134217728 is 128 MiB, and it sits between two measured populations rather
/// than beside either one. The fixed build read 45,694,976 to 49,070,080 over
/// five runs, so this is 2.7x the worst of them. The arena build read
/// 3,180,314,624 to 3,842,310,144 over three, and 387,661,824 over a fourth —
/// that last one is why the budget is not a number just under the 3.72 GiB the
/// defect is famous for, because a fourth of the arena runs came in at a tenth
/// of it, and a gate above that sample would pass on the defect it exists to
/// catch. 128 MiB is under the lowest arena sample by 2.9x and over the highest
/// gpa sample by 2.7x, so the two arms do not overlap it. A budget that only
/// fires on the exact regression is a test, and this one is chosen to fire on
/// the whole population.
///
/// Peak resident size, unlike wall time, does not inflate under load, which is
/// what makes this gate runnable inside `verify` at all and what the 2.7x is
/// for: it is headroom for a differing allocator and libc, not for a busy box.
const peak_rss_budget: []const u8 = "134217728";

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // A step, not a build-script-time call. Run from `build()` it fired on every
    // invocation, which meant a missing corpus took `zig build --help` and
    // `zig build --list-steps` down with it: you could not ask the build system
    // what steps exist, and the failure arrived as `std.process.exit` noise
    // rather than as a build failure. `verify` and `train` both depend on it, and
    // those are the two entry points that read the corpus, so the coverage the
    // original comment wanted survives the move.
    const corpus = addDigestCheck(b, "corpus", corpus_path, corpus_sha256);
    // The committed curve, checked where a reader will already be looking: the
    // gate that already checks the corpus it was derived from.
    // The label is this long on purpose. `addDigestCheck` is generic, so a failure
    // here printed "digest check failed" and two hex strings -- and a reader who had
    // legitimately changed the arithmetic could not tell from that whether the
    // committed FILE was stale or the CODE was. It names the owner of the claim
    // and the command that re-derives it, because those are the two things a
    // reader in that position needs and neither was in the output.
    const loss_csv = addDigestCheck(b, "committed loss curve (owned by src/main.zig:csv_sha256; re-derive with 'zig build train', which is the gate that checks the CODE still produces these bytes -- this one checks only that the FILE is unchanged)", loss_csv_path, loss_csv_sha256);

    const lib = b.addModule("ztransformer", .{
        .root_source_file = b.path("src/lib.zig"),
        .target = target,
        .optimize = optimize,
    });

    const exe = b.addExecutable(.{
        .name = "ztransformer",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{.{ .name = "ztransformer", .module = lib }},
        }),
    });
    b.installArtifact(exe);

    const run = b.addRunArtifact(exe);
    run.step.dependOn(b.getInstallStep());
    if (b.args) |args| run.addArgs(args);
    const run_step = b.step("run", "Run ztransformer");
    run_step.dependOn(&run.step);

    // `zig build train` runs the `train` subcommand with no extra wiring, so the
    // path a reviewer runs is the one the run step can also reach as
    // `zig build run -- train`.
    //
    // Its own ReleaseFast binary, because a training run under the default Debug
    // build is not slow but unrunnable: the whole cost is a dense f32 forward
    // and backward per step, Debug leaves it unoptimised, and one epoch over the
    // corpus runs for hours instead of minutes. The library is unchanged either
    // way, so `zig build` and `zig build test` keep the mode the user asked for.
    //
    // No module import, because `src/main.zig` reaches the implementation by
    // relative path like every `src/*_test.zig` does.
    const train_exe = b.addExecutable(.{
        .name = "ztransformer-train",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = .ReleaseFast,
        }),
    });
    const train_run = b.addRunArtifact(train_exe);
    train_run.addArg("train");
    train_run.step.dependOn(&corpus.step);
    const train_step = b.step("train", "Train and write outputs/loss.csv");
    train_step.dependOn(&train_run.step);

    // `zig build infer -- <prompt words...>`: greedy continuation from the
    // checkpoint `train` just wrote. Same ReleaseFast binary (generation is a
    // forward pass per token), prompt forwarded from `b.args` after `infer`.
    const infer_run = b.addRunArtifact(train_exe);
    infer_run.addArg("infer");
    if (b.args) |args| infer_run.addArgs(args);
    const infer_step = b.step("infer", "Generate text from outputs/checkpoint.bin (usage: zig build infer -- <prompt>)");
    infer_step.dependOn(&infer_run.step);

    // `zig build dbg-train` is the Debug build of the same run, as a step.
    //
    // It exists because the need is real — the loss curve differs between
    // optimization levels, so reproducing a difference on this host means
    // running Debug — and because three separate attempts to satisfy that need
    // replaced this whole file with a sixteen-line stub, on the reasoning that
    // the build graph was in the way. It was never in the way; a Debug
    // executable is four lines, and the reason it did not get written is that
    // nobody had asked for the step. Two of those attempts cost a restored
    // build graph and a commit made against a stale green.
    const dbg_train_exe = b.addExecutable(.{
        .name = "ztransformer-dbg-train",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = .Debug,
        }),
    });
    const dbg_train_run = b.addRunArtifact(dbg_train_exe);
    dbg_train_run.addArg("train");
    // This step's entire purpose is the comparison, and the comparison has a
    // result: measured on the Linux host of record, a Debug build of identical source DOES
    // reproduce the committed ReleaseFast curve byte for byte -- `settleCsv`
    // matches the digest and promotes. So this step is no longer demonstrating a
    // difference between optimization levels, and the flag it sets is kept for a
    // different reason: a host whose libm differs makes `zig build train` refuse
    // on the digest, and that guard is right for `train` and would make this
    // step fail for the same uninteresting reason. The flag acknowledges the
    // difference and still does not replace the committed curve, so `train` keeps
    // the guard and only this step relaxes it. The Debug build is still covered
    // by `zig build test`.
    dbg_train_run.setEnvironmentVariable("ZTRANSFORMER_ACCEPT_LOSS_DIFFERENCE", "1");
    const dbg_train_step = b.step("dbg-train", "Train in Debug, to compare optimization levels against the committed curve");
    dbg_train_step.dependOn(&dbg_train_run.step);

    // `zig build scale-profile` prints the projections behind the deferred work.
    // The same Debug binary as `run`, because it runs no arithmetic of its own
    // over a tensor: it is Config arithmetic and a table, so there is nothing
    // here for ReleaseFast to make faster. No corpus dependency, so it works in
    // a clone that has not fetched one, and it writes nothing.
    const scale_run = b.addRunArtifact(exe);
    scale_run.addArg("scale-profile");
    const scale_step = b.step("scale-profile", "Print the projected cost of shapes this model cannot be run at");
    scale_step.dependOn(&scale_run.step);

    // `zig build attn-bench` measures what one CPU attention call costs at each
    // context length and prints it beside the PCIe floor a GPU version would
    // have to clear. It reuses the same ReleaseFast binary as `zig build train`
    // rather than adding an executable: ReleaseFast because the whole point is
    // the time, and a Debug figure would describe code the project does not
    // ship; the train step's corpus dependency is on the run step, not the
    // artifact, so nothing is fetched. It writes nothing.
    const attn_bench_run = b.addRunArtifact(train_exe);
    attn_bench_run.addArg("attn-bench");
    const attn_bench_step = b.step("attn-bench", "Measure CPU attention per context length against the PCIe floor a GPU kernel must clear");
    attn_bench_step.dependOn(&attn_bench_run.step);

    // One test binary per mode. The two run concurrently: `train_test.zig`
    // writes its scratch CSVs under a per-process directory, so two copies in one
    // tree no longer delete each other's files. They used to, which is why this
    // used to serialise the modes.
    //
    // ReleaseFast is the mode the train binary ships in, and before `verify` no
    // test ever ran in it. A Debug-only suite is blind to the class of bug that
    // only the optimizer exposes, and it is the release build a reviewer runs.
    const default_tests = addTests(b, lib, target, optimize);
    const release_tests = addTests(b, lib, target, .ReleaseFast);

    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&default_tests.step);

    // The whole gate behind one command: formatting, the test suite in the mode
    // a developer runs and in ReleaseFast, the version banner, the corpus digest
    // checked above before the graph existed, and the digest of the one committed
    // measurement the root README quotes numbers from. Silent when it passes,
    // because a gate that prints on every green run trains everyone to ignore
    // it. Nothing here writes anything outside the build cache.
    //
    // The description below names every sub-check this gate runs. It used to
    // name fewer than it ran: it listed neither `tools/symbols.sh` nor the scale
    // tables in `src/README.md`, so `zig build --list-steps` understated the gate
    // and a reader weighing a green run against a shorter list than the one that
    // ran was reading a stale copy. `README.md` already named them all, which is
    // how only this one went stale.
    //
    // It now also names the ONE thing the gate does not do, because that list has
    // no other place to say it and `verify` is silent on success: the CUDA
    // attention comparison needs a CUDA toolchain and a GitHub runner has none,
    // so it lives in `zig build cuda-attn-check` and a green `verify` says
    // nothing at all about it. A gate whose omissions are unstated is how a
    // reader concludes a check happened.
    const verify_step = b.step("verify", "Check fmt, the test suite in Debug, the test suite in ReleaseFast, the version banner, the corpus digest, the committed loss curve, the attention benchmark's CPU half compiles, the scale tables in src/README.md, the symbol table in src/README.md, and the training run's peak memory -- but NOT the CUDA attention comparison, which needs a CUDA toolchain and is 'zig build cuda-attn-check', and NOT the negative controls for that scale-table gate, which print what they caught and are 'zig build table-block-check'");

    // `b.graph.zig_exe` rather than `zig` off `PATH`, so the formatter that
    // decides whether the tree is formatted is the same compiler running the
    // build. `--check` exits non-zero and names the files it would rewrite, and
    // the cwd is the build root so the relative paths resolve wherever the
    // command was typed from.
    //
    // Scoped to the source, not to `.`. With `.`, a contributor's scratch file
    // anywhere under the tree fails the gate — and the two places that actually
    // happen are `zig-out/` and `.zig-cache/`, both gitignored, so the file is
    // one the contributor cannot add, has no reason to know about, and gets
    // named in the error as if it were part of the repo. The gate is about the
    // source, so it reads the source.
    const fmt = b.addSystemCommand(&.{
        b.graph.zig_exe, "fmt",   "--check",
        "src",           "tools", "build.zig",
    });
    fmt.setCwd(b.path("."));
    // The cheapest check goes first and the tests wait for it, so an unformatted
    // tree is reported in a second instead of after a minute of tests.
    //
    // Both test binaries gate on fmt, and `verify` depends on both. When the two
    // modes were serialised, that serialisation was the only edge pulling the
    // Debug suite into `verify`; removing it to stop two copies of the suite
    // racing left `verify` running 160 tests while the README promised 320. The
    // dependency is written out here rather than inherited from an ordering, so
    // removing an ordering cannot silently halve the gate again.
    default_tests.step.dependOn(&fmt.step);
    release_tests.step.dependOn(&fmt.step);
    verify_step.dependOn(&fmt.step);
    verify_step.dependOn(&default_tests.step);
    verify_step.dependOn(&release_tests.step);
    verify_step.dependOn(&corpus.step);
    verify_step.dependOn(&loss_csv.step);

    // `src/cuda/attn_twin.zig` is the CPU half of the attention benchmark: it
    // writes the inputs, the reference output and the manifest that
    // `src/cuda/attn.cu` is graded against, so the published 57.9x to 168.1x
    // table cannot be reproduced without it compiling. (55.4x to 169.4x is
    // one of the three ranges withdrawn as a floor; see the root README.)
    //
    // It had no target here at all, because `run-attn.sh` invokes `zig
    // build-exe` itself, and that made this a permanent hole rather than a slow
    // one: a rename or a signature change in that file stayed green here
    // indefinitely, since nothing in this graph ever read it. Compiling it is
    // the whole addition, and it is CPU only -- no CUDA object, no libcudart --
    // so it costs a runner with no card nothing.
    //
    // `autograd` rather than `src/attention.zig` alongside it, because the twin
    // needs `attention.forward` and `attentionBackward` and passing the two
    // files as separate modules compiles every symbol they share twice. The
    // same wiring `run-attn.sh` passes, for the same reason.
    const attn_twin_exe = b.addExecutable(.{
        .name = "attn-twin",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/cuda/attn_twin.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .imports = &.{.{
                .name = "autograd",
                .module = b.createModule(.{
                    .root_source_file = b.path("src/autograd.zig"),
                    .target = target,
                    .optimize = .ReleaseFast,
                }),
            }},
        }),
    });
    verify_step.dependOn(&attn_twin_exe.step);
    b.step("attn-twin", "Build the CPU half of the attention benchmark").dependOn(&attn_twin_exe.step);

    // The scale tables `src/README.md` quotes, checked against the tool that
    // prints them. `scale-profile` writes no timestamp, no address and no float
    // whose formatting can drift, so two runs are byte-identical and the
    // comparison is a `cmp` rather than a judgment call. Without this the
    // README's claim to have been copied from the tool is a promise, and a
    // promise is what b168cf9 broke: it grew the activation cache, and six table
    // rows kept printing what the previous cache cost.
    //
    // A run step of its own rather than `scale_step`'s, because that one
    // INHERITS stdout -- it is the step a reader runs to read the tables -- and a
    // step that both printed the tables and was compared against them would have
    // to capture them, which would silence the reader's copy.
    const scale_capture = b.addRunArtifact(exe);
    scale_capture.addArg("scale-profile");
    verify_step.dependOn(&addTableBlockCheck(b, scale_capture.captureStdOut(.{}), "src/README.md", "scale-profile", &.{
        "ARITHMETIC, PROJECTED, GFLOP",
        "BYTES, PROJECTED, GiB",
        "VERDICTS, one line per deferred item, at 32 GiB of host memory",
    }, 20).step);

    // That the gate above can fail, on the real script rather than on a copy of
    // it. Deliberately NOT in `verify`, and the reason is the silence contract
    // rather than a missing toolchain: CI asserts that a passing `zig build
    // verify` writes no bytes to either stream, and this step's entire output IS
    // the evidence that the three broken tables were caught. `cuda-check` is
    // outside `verify` for the same shape of reason with a different cause.
    const table_block_check = b.addSystemCommand(&.{"sh"});
    table_block_check.addFileArg(b.path("tools/table-block-check.sh"));
    table_block_check.setCwd(b.path("."));
    b.step("table-block-check", "Require the README table-block gate to catch a drifted digit, a deleted marker and a tool that prints nothing, and print what each one produced").dependOn(&table_block_check.step);

    // Is this host in a state where a timing means anything? Run it before a
    // benchmark, not inside `verify`: `verify` is silent by contract and runs on
    // every runner, and a check that reads `nvidia-smi` belongs to the host that
    // has one. The thresholds are in the script and overridable in the
    // environment, so a reader who disagrees changes one number and says so.
    //
    // It is here because the alternative is the failure this repository has
    // already published twice: a table generated from a contaminated host, which
    // is a plausible-looking table and the worst kind.
    const host_check = b.addSystemCommand(&.{"sh"});
    host_check.addFileArg(b.path("tools/host-clean.sh"));
    host_check.setCwd(b.path("."));
    b.step("host-check", "Refuse when the host is in a state that would contaminate a timing, and name which condition failed").dependOn(&host_check.step);

    // Two short training passes that must produce the same bytes. `verify`
    // hashes the committed curve and does NOT re-derive it, so nothing in the
    // graph proves the code still produces those bytes -- and a reader is told
    // they do. This is the step that makes that true.
    //
    // Short on purpose: 16 KiB of corpus is the smallest that still yields a
    // validation window. `data.split` keeps 5% for validation and one window is
    // `ctx` 256 tokens, so 8 KiB leaves fewer than 256 val tokens and the run
    // refuses with `error: EmptyValidation` before writing anything -- 4096 and
    // 8192 were both tried and both refuse. At 16 KiB it is 30 steps, so the pair
    // costs about a second. The property is the arithmetic's rather than the run
    // length's: trajectory chaos needs many steps to amplify, which is exactly
    // why a SHORT pair is the cheap way to test reproducibility.
    //
    // Host-relative and not cross-machine, deliberately. A different libm gives
    // different bytes legitimately, so this can never assert a curve digest; it
    // asserts that two runs on THIS host agree with each other. The committed
    // `csv_sha256` keeps its meaning as one host's claim, and nothing here turns
    // it into a cross-machine one.
    //
    // Both passes write `outputs/loss.pending.csv` and neither promotes, because
    // a three-row curve is not the committed five-row one and `settleCsv` refuses
    // on the digest. The pending file is moved aside after each pass and deleted
    // at the end, so a failed run leaves the tree as it found it.
    const determinism = b.addSystemCommand(&.{
        "sh",
        "-c",
        \\set -eu
        \\bin=$1
        \\work=$(mktemp -d)
        \\trap 'rm -rf "$work"' EXIT
        \\export ZTRANSFORMER_CORPUS_BYTES=$2
        \\rm -f outputs/loss.pending.csv
        \\"$bin" train > "$work/a.log" 2>&1 || true
        \\## `2>/dev/null || true` is load-bearing, not decoration. Under `set -eu` a
        \\## bare `cp` that finds nothing ABORTS the step on the spot, which is why the
        \\## first version of this guard could never fire: it sat behind a command that
        \\## had already ended the script. Pass two has had the suppression all along;
        \\## this gives pass one the same, so the guard below is reachable at all.
        \\cp outputs/loss.pending.csv "$work/a.csv" 2>/dev/null || true
        \\if [ ! -f "$work/a.csv" ]; then
        \\  echo "determinism: the FIRST pass produced no curve, so there is" >&2
        \\  echo "  nothing to compare. A pass that never ran is not evidence of" >&2
        \\  echo "  determinism, in either direction. The first pass said:" >&2
        \\  tail -5 "$work/a.log" >&2 || true
        \\  exit 1
        \\fi
        \\# Removed AGAIN before the second pass, and the reason is a hole this
        \\## step had: `|| true` swallows a second pass that fails, and without
        \\## this rm the `cp` below would copy the leftover from pass one into
        \\## b.csv. `cmp` then matches and the step exits 0 having measured one
        \\## run and called it two. With the file gone, a pass that writes
        \\## nothing leaves `cp` failing under `set -eu`, which is the honest
        \\## outcome: a second pass that did not happen is not evidence of
        \\## determinism.
        \\rm -f outputs/loss.pending.csv
        \\"$bin" train > "$work/b.log" 2>&1 || true
        \\# `|| true` here too, and for the same reason: the rm above means a pass
        \\## that wrote nothing leaves nothing to copy, and under `set -eu` a bare
        \\## `cp: cannot stat` would be the whole diagnostic. This step promises a
        \\## message that NAMES what failed, so the missing file is reported in the
        \\## step's own words and with the pass-two log attached.
        \\cp outputs/loss.pending.csv "$work/b.csv" 2>/dev/null || true
        \\if [ ! -f "$work/b.csv" ]; then
        \\  echo "determinism: the SECOND pass produced no curve, so there is" >&2
        \\  echo "  nothing to compare the first against. A pass that never ran" >&2
        \\  echo "  is not evidence of determinism. Pass one wrote" >&2
        \\  echo "  $(( $(wc -l < "$work/a.csv") - 1 )) steps; the second pass said:" >&2
        \\  tail -5 "$work/b.log" >&2 || true
        \\  exit 1
        \\fi
        \\rm -f outputs/loss.pending.csv
        \\# SILENT ON SUCCESS. `verify` is silent by contract and CI asserts that a
        \\# passing run writes no bytes to either stream, so a step inside it that
        \\# announced its digest would break that assertion to print a number
        \\# nobody reads. `table-block-check` is outside `verify` for exactly this
        \\# reason; this one is inside because determinism is a property worth
        \\# gating, so it pays for the silence instead.
        \\# A floor on BOTH copies, before the comparison. `cmp -s` exits 0 on two
        \\## zero-byte files, so an empty pair certifies the arithmetic having
        \\## measured nothing -- and `outputs/loss.pending.csv` is gitignored, so an
        \\## empty one can sit in a working tree from an earlier run. This is the
        \\## same hole `tools/table-block.sh` closes with a line-count floor, and the
        \\## shape of the fix is copied from there rather than invented here.
        \\##
        \\## The number is MEASURED, and the first guess was wrong. These passes run at
        \\## 16 KiB and write THREE lines: a header and two logged steps, because
        \\## `log_every` is far larger than the run. An audit proposed five, from the
        \\## thirty steps `outputs/README.md` quotes, and that turned a passing gate
        \\## red on its own host. The floor has to catch an empty file and a
        \\## header-only file, so two -- one data row -- is what it takes, and the
        \\## rest of the margin is `cmp`'s to argue about.
        \\a_lines=$(wc -l < "$work/a.csv" 2>/dev/null || echo 0)
        \\b_lines=$(wc -l < "$work/b.csv" 2>/dev/null || echo 0)
        \\if [ "$a_lines" -lt 2 ] || [ "$b_lines" -lt 2 ]; then
        \\  echo "determinism: a pass produced too few lines to be a curve." >&2
        \\  echo "  a.csv $a_lines lines, b.csv $b_lines lines. Two is a header and" >&2
        \\  echo "  one step; these passes write three. Two files that short can" >&2
        \\  echo "  still cmp equal, so comparing them would certify the" >&2
        \\  echo "  arithmetic on no data at all." >&2
        \\  exit 1
        \\fi
        \\if cmp -s "$work/a.csv" "$work/b.csv"; then
        \\  exit 0
        \\else
        \\  echo "determinism: two runs of one seed DISAGREED on this host, so the" >&2
        \\  echo "  arithmetic is not reproducible here and the committed curve is" >&2
        \\  echo "  a claim this build cannot back. $(( $(wc -l < "$work/a.csv") - 1 )) steps each." >&2
        \\  diff -u "$work/a.csv" "$work/b.csv" >&2 || true
        \\  echo "determinism: the first pass's own output, in case it never got far:" >&2
        \\  tail -5 "$work/a.log" >&2 || true
        \\  exit 1
        \\fi
        ,
        "determinism",
    });
    determinism.addArtifactArg(train_exe);
    determinism.addArg("16384");
    determinism.setCwd(b.path("."));
    const determinism_step = b.step("determinism", "Run two short training passes and require them to produce byte-identical curves");
    determinism_step.dependOn(&determinism.step);
    verify_step.dependOn(determinism_step);

    // Per-op attribution for one training step, with the two negative controls
    // that make it a gate rather than a print.
    //
    // The body is `tools/step-profile.sh`, not a `sh -c` blob here, for the same
    // reason `tools/host-clean.sh` and `tools/symbols.sh` are files: a gate has
    // to be breakable on purpose and readable on its own, and every failure this
    // step raised was unreadable because the script was one argv string printed
    // inside zig's "failed command" line. `sh -x tools/step-profile.sh <bin>`
    // now shows the whole thing.
    //
    // NOT in `verify`, for the reason `bench` is not: these are times, and a
    // threshold on a time is a property of the machine and the hour. What is
    // checked is a SHAPE -- the denominator is elapsed time rather than the sum
    // of the buckets, and the shares move when the work does. Both hold on a
    // loaded host as much as on an idle one, which is why they can gate and a
    // magnitude cannot.
    //
    // Two runs at different corpus sizes: identical per-step arithmetic, three
    // times the steps. The fixed per-RUN cost -- tokenizer, `initParams` -- is
    // amortised over three times as many steps, so the per-step ops take a
    // larger share. `backward` moves about 77% to 82% and `forward` 19% to 15%,
    // which is the signal control 2 measures. An earlier version of this comment
    // credited `eval`, which runs once per epoch; it moves 0.01 points and is
    // not the mechanism.
    const step_profile = b.addSystemCommand(&.{"sh"});
    step_profile.addFileArg(b.path("tools/step-profile.sh"));
    step_profile.addArtifactArg(train_exe);
    step_profile.setCwd(b.path("."));
    b.step("step-profile", "Time each op kind in a training step, and require the denominator to be elapsed time and the shares to move when the work does").dependOn(&step_profile.step);

    // The peak of the training run, against `peak_rss_budget`. `verify`
    // depends on it, and the reason it is here rather than in the one-command
    // list of digests is that it is the only gate in this file that runs the
    // model. It costs the run, and it is worth the run: the four digests above
    // all assert that committed bytes are the committed bytes, and none of them
    // can notice a program that holds a whole epoch's allocations at once, which
    // is the one defect in this repository that no digest could ever have
    // caught.
    //
    // Inside `verify` rather than beside it, because `verify` is the command a
    // reviewer runs to find out whether the tree is sound, and a gate a reader
    // has to know the name of is a gate that does not get run. It is also a
    // named step, so it can be run on its own when the point is the number
    // rather than the gate.
    const peak_rss = b.step("peak-rss", "Check the training run's peak resident set size against the budget in build.zig");
    peak_rss.dependOn(&addPeakRssCheck(b, train_exe).step);
    verify_step.dependOn(peak_rss);

    // ORDERING, and it is load-bearing rather than tidy. `peak-rss` and
    // `determinism` both invoke the train binary, and `train` writes its curve to
    // ONE path, `outputs/loss.pending.csv`, with no per-run name in it. Zig runs
    // independent top-level dependencies concurrently, so with no edge between
    // these two the runs overlap on that file: determinism's `rm -f` deletes the
    // curve peak-rss just wrote, or determinism's `cp` copies peak-rss's
    // 123-step curve and compares it against its own 30-step one. The second
    // outcome is the bad one -- the step then reports that the arithmetic is not
    // reproducible on this host, confidently, from a race rather than from the
    // arithmetic. A `verify` run was observed returning exit 1 with 5120 bytes of
    // output and then exit 0 with none, unchanged in between, and this is where
    // that came from. Any step added later that invokes `train` needs the same
    // edge; sharing one writable path between concurrent steps is the defect.
    determinism.step.dependOn(peak_rss);

    // CPU seconds per training step, median of a few runs. It reports and it does
    // not gate, and the asymmetry is the point: `peak-rss` can gate because a
    // peak in bytes is a property of the program, while a time in seconds is a
    // property of the program *and the machine and the hour*. A threshold on it
    // would fire on someone else's load average and teach the gate to be ignored.
    //
    // It exists because the backward pass went 0.4522 to 0.2721 s/step and the
    // only record of that was a console transcript, which is the same defect
    // this file already carries three of. A number in a README needs a command
    // beside it; this is that command for the one number a reader is most
    // likely to want to check.
    const bench = b.step("bench", "Report median CPU seconds per training step over N runs (default 3)");
    bench.dependOn(&addBench(b, train_exe).step);

    // Nothing in this graph reached `src/cuda/` until now, which `AGENTS.md`
    // said out loud: a change there was unchecked until a person ran
    // `sh src/cuda/run-norm.sh` on an NVIDIA host. This compiles three CUDA sources
    // files, so a syntax or type error in them is caught by the build system.
    //
    // Deliberately NOT in `verify`: `verify` is silent on success and CI asserts that, so
    // anything in it runs on every ubuntu runner. A GitHub runner has no CUDA
    // toolchain and never will, so a compile step there would be a red for a
    // reason that has nothing to do with the code. A named step a reader has to
    // ask for is better than a gate that is always red or a directory nothing
    // compiles.
    //
    // It sources `src/cuda/cuda.sh` and compiles through `cuda()`, rather than
    // restating the flags or calling the host's nvcc, so the architecture
    // derivation, the `-Werror -fPIC` set and the toolchain itself keep exactly
    // one owner. That matters more than it looks: a host may well have a second,
    // newer toolkit installed, and compiling with that while measuring with the
    // pinned one would mean two toolchains in one repository, which is the
    // condition that lets a benchmark quietly stop describing the build that
    // produced it. So this says nothing about what any host has installed; the
    // invariant it enforces is that the check compiles what is measured.
    //
    // It also inherits cuda.sh's guard: with no toolchain and no GPU, sourcing
    // fails and the step says so rather than skipping. A compile check that
    // quietly compiles nothing is the same defect as a gate that quietly checks
    // nothing.
    const cuda_check = b.addSystemCommand(&.{
        "sh",
        "-c",
        \\set -eu
        \\CUDA_ROOT_DIR=$PWD
        \\export CUDA_ROOT_DIR
        \\. src/cuda/cuda.sh
        \\cuda_pull
        \\flags=$(cuda_nvcc_flags)
        \\for f in norm probe attn; do
        \\  cuda "nvcc $flags -c -o .zig-cache/cuda-check-$f.o src/cuda/$f.cu"
        \\  echo "cuda-check: src/cuda/$f.cu compiled for $CUDA_ARCH"
        \\done
        \\for f in norm probe attn; do rm -f ".zig-cache/cuda-check-$f.o"; done
        \\echo "cuda-check: every source compiled with the pinned toolchain, which is"
        \\echo "the one sh src/cuda/run-norm.sh measures with."
        ,
    });
    cuda_check.setCwd(b.path("."));
    const cuda_check_step = b.step("cuda-check", "Compile src/cuda/*.cu with the pinned toolchain, or fail loudly if there is none");
    cuda_check_step.dependOn(&cuda_check.step);

    // The CUDA attention kernels, reachable from a training step. This is the
    // third of the three pieces `src/cuda/device.zig`'s header names: the C
    // entry points exist, the Zig binding for them exists, and until now nothing
    // linked the two.
    //
    // Two steps rather than one, and the split is what keeps `verify` green on a
    // runner with no CUDA toolchain. `cuda-attn-check` builds the gate in
    // `src/train.zig` -- the smallest tree with both halves of the seam in it --
    // and `cuda-train` builds the training binary. Neither is a dependency of
    // anything: `train`, `bench`, `peak-rss` and `verify` all go through
    // `train_exe`, which links no object and no CUDA runtime, so every one of
    // them is the build it was before this and `outputs/loss.csv` is reproduced
    // by the same bytes.
    //
    // BOTH STEPS FAIL while `src/model.zig:cuda_attn` is false, rather than
    // building something that links an object nothing calls. That build would
    // compile, run a whole 123-step training pass on the CPU and exit 0 -- a
    // green that checked nothing, which is the defect this repository keeps
    // finding in its own gates, and which `run-attn.sh` writes a paragraph about
    // when `ATTN_GROUP_Q` is set on the host instead of inside the container.
    // The two descriptions name the one thing each step needs that is not in the
    // graph. `cuda-attn-check` builds a binary that LINKS against `libcudart` and
    // then RUNS, so it needs the pinned 12.6.3 runtime copied out of the pinned
    // image -- `src/cuda/README.md` has the recipe -- on BOTH `LIBRARY_PATH`, which
    // the linker reads, and `LD_LIBRARY_PATH`, which the loader reads and
    // `LIBRARY_PATH` does not feed. Without the second the step fails with a
    // message about a missing shared library on a machine that has it, which is
    // the worst shape a missing-argument error has.
    const cuda_check_desc = "Grade the CUDA attention forward against the CPU one at a real step's scale, AND the CUDA decode step against the CPU one over a filling cache (needs the pinned libcudart on LIBRARY_PATH *and* LD_LIBRARY_PATH; recipe in src/cuda/README.md)";
    const cuda_train_desc = "Train with the CUDA attention forward (needs the pinned libcudart on LIBRARY_PATH *and* LD_LIBRARY_PATH; recipe in src/cuda/README.md)";
    if (model_source.cuda_attn) {
        const nvcc = cudaAttnObject(b);

        // `src/train.zig` as its own test root. Rooted at `src/tests.zig` it
        // would also re-run the other 202 tests against a CUDA-linked binary,
        // which is a different and much slower check than the one this step is
        // named for -- and the one gate here is the only test in the tree that
        // needs a device.
        const cuda_tests = b.addTest(.{
            .root_module = b.createModule(.{
                .root_source_file = b.path("src/train.zig"),
                .target = target,
                .optimize = .ReleaseFast,
                .link_libc = true,
            }),
        });
        linkCudaAttn(cuda_tests.root_module);
        cuda_tests.step.dependOn(&nvcc.step);
        const cuda_test_run = b.addRunArtifact(cuda_tests);

        // The decode gate, as its own test root for the same reason the training one
        // is: `src/decode_test.zig` carries one test that needs a device -- a decode
        // step's GPU answer against the CPU `attnStep` over a filling cache -- and
        // rooting the suite at `src/tests.zig` would re-run the other 217 against a
        // CUDA-linked binary.
        //
        // It lives in this step rather than beside a note telling the reader to run it
        // by hand, because that is the defect this repository keeps finding in its own
        // gates: a check whose only invocation is prose is a check nobody runs. Its
        // `error.SkipZigTest` arm still reports SKIP in every step that links no
        // object, so it cannot pass silently either.
        const cuda_decode_tests = b.addTest(.{
            .root_module = b.createModule(.{
                .root_source_file = b.path("src/decode_test.zig"),
                .target = target,
                .optimize = .ReleaseFast,
                .link_libc = true,
            }),
        });
        linkCudaAttn(cuda_decode_tests.root_module);
        cuda_decode_tests.step.dependOn(&nvcc.step);
        const cuda_decode_run = b.addRunArtifact(cuda_decode_tests);

        // Two `dependOn` calls rather than one chain: it returns void, so a chain is a
        // compile error rather than a longer statement.
        const cuda_attn_check = b.step("cuda-attn-check", cuda_check_desc);
        cuda_attn_check.dependOn(&cuda_test_run.step);
        cuda_attn_check.dependOn(&cuda_decode_run.step);

        const cuda_train_exe = b.addExecutable(.{
            .name = "ztransformer-cuda-train",
            .root_module = b.createModule(.{
                .root_source_file = b.path("src/main.zig"),
                .target = target,
                .optimize = .ReleaseFast,
                .link_libc = true,
            }),
        });
        linkCudaAttn(cuda_train_exe.root_module);
        cuda_train_exe.step.dependOn(&nvcc.step);
        const cuda_train_run = b.addRunArtifact(cuda_train_exe);
        cuda_train_run.addArg("train");
        // `settleCsv` refuses to promote a curve that is not the committed bytes
        // and exits 1, so without this every run of this step would fail on a
        // difference that is the point: the kernel computes the same function in
        // a different order and a different last bit, which is the whole subject
        // of `train.zig`'s `divergence`. Nothing is promoted by it either way --
        // the committed curve is still only replaced on a digest match -- and the
        // reader gets the curve and the divergence printed alongside it. The
        // same variable `dbg-train` sets, for the same reason.
        cuda_train_run.setEnvironmentVariable("ZTRANSFORMER_ACCEPT_LOSS_DIFFERENCE", "1");
        cuda_train_run.step.dependOn(&corpus.step);
        b.step("cuda-train", cuda_train_desc).dependOn(&cuda_train_run.step);
    } else {
        const off = b.addSystemCommand(&.{
            "sh",
            "-c",
            \\echo "src/model.zig has cuda_attn = false, so no binary in this build calls" >&2
            \\echo "src/cuda/attn_kernels.cu and this step would link an object nothing" >&2
            \\echo "references. That is a training run on the CPU and a green result." >&2
            \\echo >&2
            \\echo "  edit src/model.zig: pub const cuda_attn: bool = true;" >&2
            \\echo "  zig build cuda-attn-check   # the gate" >&2
            \\echo "  zig build cuda-train        # a training run" >&2
            \\echo "  edit it back to false, and 'git diff src/model.zig' is how you" >&2
            \\echo "  check -- not 'git checkout', which would take your other" >&2
            \\echo "  edits to that file with it" >&2
            \\echo >&2
            \\echo "It is a source constant and not -D because it has to be comptime," >&2
            \\echo "and a build option could not reach src/model.zig anyway: this file" >&2
            \\echo "imports it." >&2
            \\echo >&2
            \\echo "Leaving it TRUE is the mirror image and it is worse -- then" >&2
            \\echo "'zig build test' fails at the LINK step on zt_attn_forward and" >&2
            \\echo "cudaFree, on every host with no CUDA toolchain, which is every" >&2
            \\echo "GitHub runner. Both halves of that are in src/model.zig's own" >&2
            \\echo "comment on cuda_attn."
            \\exit 1
            ,
        });
        off.setCwd(b.path("."));
        b.step("cuda-attn-check", cuda_check_desc).dependOn(&off.step);
        b.step("cuda-train", cuda_train_desc).dependOn(&off.step);
    }

    // Every `module.Symbol` row in src/README.md's tables names a declaration the
    // code actually makes pub. The drift is not hypothetical: 6c2a9c5 had to
    // hand-edit five table rows because nothing said otherwise, and the script
    // written to catch it shipped unwired -- tools/README.md claimed `verify` ran
    // it and this file never mentioned it. A gate nothing calls is a claim, and
    // the drift it exists to catch is the kind that reads as true.
    //
    // Both files go in as `addFileArg` rather than as argv so the build system
    // tracks them. The step also depends on the library compile and on the test
    // run, which is what makes the gate able to see every module: `exe` alone
    // reaches only what `main.zig` imports, and `src/gradcheck.zig` is imported
    // by nothing except `gradcheck_test.zig` -- so the module with the most
    // heavily documented public surface was the one module the gate was blind
    // to. `gradcheck` has six rows in the same table shape the script reads, and
    // making any of them private changed no tracked input, so the step was a
    // cache hit and `verify` stayed green over a lying README. The test run is
    // already a `verify` dependency, so this costs no extra work.
    const symbols = b.addSystemCommand(&.{"sh"});
    symbols.addFileArg(b.path("tools/symbols.sh"));
    symbols.addFileArg(b.path("src/README.md"));
    symbols.addArg(".");
    symbols.setCwd(b.path("."));
    symbols.step.dependOn(&exe.step);
    symbols.step.dependOn(&default_tests.step);
    verify_step.dependOn(&symbols.step);

    // The banner, captured rather than inherited: `verify` has to be silent, and
    // an exact stdout match asserts more than the exit code alone, which is the
    // only thing a bare run step would check.
    const banner_run = b.addRunArtifact(exe);
    _ = banner_run.captureStdOut(.{});
    banner_run.expectStdOutEqual(b.fmt("{s} {s}\n", .{ lib_source.name(), lib_source.version }));
    verify_step.dependOn(&banner_run.step);
}

/// `src/cuda/attn_kernels.cu`, compiled into an object the build graph links.
///
/// `attn_kernels.cu` and NOT `attn.cu`, and that is the whole of the difference:
/// `attn.cu` is the benchmark harness, it `#include`s this file and defines
/// `main`, so linking it fails at `multiple definition of 'main'` -- a message
/// that names neither of the two files it could have meant. `attn_kernels.cu`
/// carries its own `#include`s, holds no `main`, and compiles to exactly the six
/// `zt_attn_*` entry points.
///
/// The flags and the toolchain are NOT restated. `src/cuda/cuda.sh` is sourced
/// and `cuda()` is called, exactly as `cuda-check` above does, so the
/// architecture derivation, the `-Werror -fPIC` set and the pinned image keep
/// one owner and this cannot compile something the published table was never
/// measured from. `CUDA_ROOT_DIR` is exported for the reason that header gives:
/// a sourced file sees the CALLER's `$0`, which here is `sh`, so without it the
/// container bind-mounts a directory that is not this one.
///
/// The object lands under `.zig-cache/cuda/`, which is gitignored and which
/// `run-attn.sh`'s EXIT trap does not remove -- it deletes
/// `.zig-cache/cuda/attn`, the sibling. It is written by the container under the
/// host's own uid, for the reason `cuda.sh`'s `--user` comment gives.
fn cudaAttnObject(b: *std.Build) *std.Build.Step.Run {
    const nvcc = b.addSystemCommand(&.{
        "sh",
        "-c",
        \\set -eu
        \\CUDA_ROOT_DIR=$PWD
        \\export CUDA_ROOT_DIR
        \\. src/cuda/cuda.sh
        \\cuda_pull
        \\flags=$(cuda_nvcc_flags)
        \\mkdir -p .zig-cache/cuda
        \\cuda "nvcc $flags -c -o .zig-cache/cuda/attn_kernels.o src/cuda/attn_kernels.cu"
        ,
    });
    nvcc.setCwd(b.path("."));
    return nvcc;
}

/// The compiled kernels and the runtime they resolve against, on one module.
///
/// `cudart` is `-lcudart`, and WHICH one is the entire point. the host of record
/// carries a CUDA 13.4 toolkit and `ldconfig` resolves its `libcudart.so` to
/// `/usr/local/cuda/targets/x86_64-linux/lib/`, so a linker there finds `-lcudart`
/// with no help at all -- while the object above was built by the pinned 12.6.3
/// image. Linking the host's copy would put two toolchains in one build, the
/// condition `AGENTS.md` refuses, and it would do so invisibly: nothing fails and
/// every number still prints.
///
/// The pinned runtime therefore has to be copied out of the image first, and
/// `src/cuda/README.md` carries that recipe -- `cp -P` the `libcudart.so*`
/// chain into a directory and put it on `LIBRARY_PATH`. It is NOT in `cuda.sh`,
/// because every script here links inside the container and this is the first
/// one that links outside it. Stated here rather than assumed, and nothing in
/// this repository can check it: the copy-out is outside the build graph, so the
/// one thing that would catch a wrong `LIBRARY_PATH` is a reader who runs the
/// recipe.
///
/// TWO VARIABLES, NOT ONE, and the README's recipe as written is only half of it.
/// `LIBRARY_PATH` is what the LINKER reads, and it is enough to get past the link;
/// the steps built above then RUN, and the LOADER resolves `libcudart.so.12` from
/// its own search path, which `LIBRARY_PATH` does not feed. Without
/// `LD_LIBRARY_PATH` the gate fails as
/// `error while loading shared libraries: libcudart.so.12: cannot open shared
/// object file` -- a message about a missing library rather than about a missing
/// loader variable, on a machine that demonstrably has the library. So a reader
/// needs both, and the fact that is not in the README is the reason it is here.
///
/// `dl` is on `nvcc`'s own link line, because the runtime reaches the driver with
/// `dlopen`. glibc 2.34 folded `libdl` into libc, so it is redundant on a host
/// that new and required on one that is older, and it costs nothing either way.
///
/// `link_libc` is set on the modules themselves rather than here, and it is not
/// optional: the entry points print their refusals with `fprintf` and abort with
/// `exit`.
fn linkCudaAttn(m: *std.Build.Module) void {
    m.addObjectFile(m.owner.path(".zig-cache/cuda/attn_kernels.o"));
    m.linkSystemLibrary("cudart", .{});
    m.linkSystemLibrary("dl", .{});
}

/// One test binary, in one optimize mode, as a run step.
///
/// The module is rooted at `tests.zig`, because Zig has no test globbing. A
/// `*_test.zig` reaches its module by relative path and `tests.zig` itself needs
/// `version` and `name`, so it is the one file that wants the library by name.
/// `lib.zig` imports nothing, so the module graph holds no file twice: the
/// implementations belong to the test module, the library is reachable only as
/// `ztransformer`, and both are satisfied at once.
fn addTests(
    b: *std.Build,
    lib: *std.Build.Module,
    target: std.Build.ResolvedTarget,
    mode: std.builtin.OptimizeMode,
) *std.Build.Step.Run {
    const tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/tests.zig"),
            .target = target,
            .optimize = mode,
            // `src/train_test.zig` builds its scratch path from
            // `std.c.getpid()`, so the test module reaches libc. macOS links it
            // implicitly and the suite passes there, which is why this survived:
            // the first run of `zig build verify` on Linux failed with
            // "dependency on libc must be explicitly specified in the build
            // command", and `.github/workflows/ci.yml` runs on ubuntu-latest.
            // One call in one test file, and the platform nobody developed on
            // was the one that could not build.
            .link_libc = true,
            .imports = &.{.{ .name = "ztransformer", .module = lib }},
        }),
    });
    return b.addRunArtifact(tests);
}

/// A step that fails unless the tables a tool prints are byte-for-byte the tables
/// a README quotes between two markers.
///
/// `src/README.md`'s `scale-profile` block is the only such table today, and
/// this is the only implementation of the rule, so a second one is a call rather
/// than the fifty lines of shell that used to be copied into this file for it.
///
/// The comparison is `tools/table-block.sh` rather than shell here, for the
/// reason `tools/symbols.sh` is a file and not a string: this gate has to be
/// breakable on purpose, and fifty lines of `awk` inside a build script can only
/// be broken by running a build. The script finishes in a second and prints what
/// each broken case produced, which is what `zig build table-block-check` runs.
///
/// `tool_out` is a captured stdout rather than an inherited one, for the same
/// reason the version banner's is: `verify` is silent by contract, and a run
/// step that printed the whole profile on every green run would train everyone to
/// scroll past it. Handing it over as a file argument is also what makes the run
/// that produces it a dependency of this step, so the comparison can never read
/// a stale file.
///
/// `table_headers` is the one thing that cannot be derived from `block`: a tool
/// prints its tables under whatever headings its own output gives them, and the
/// markers are named after the step rather than after the tables. Everything else
/// — the marker names, the README, the floor, the comparison, and both failure
/// messages — is identical for every table, and repeating it per call site is the
/// copy-paste this exists to remove.
///
/// `floor` is the smallest extraction the check will accept, and it means one
/// thing: below this many lines an extraction is treated as having found nothing
/// rather than as a table to diff. `cmp -s` exits 0 on two empty files, and keying
/// on header LINES is what makes that reachable — rename a header in the tool,
/// delete the block from the README, and both extractions come back empty while
/// the comparison passes on nothing. That is the same defect as the `sed` pipeline
/// the other platform had, where an empty derivation becomes a silently wrong
/// `-arch` flag. It is a parameter because one shared check cannot know how long
/// an arbitrary table is, and 20 against a real 29 is both well clear of zero and
/// well under the real length, so adding a row does not require editing this
/// call.
fn addTableBlockCheck(
    b: *std.Build,
    tool_out: std.Build.LazyPath,
    readme_path: []const u8,
    block: []const u8,
    table_headers: []const []const u8,
    floor: usize,
) *std.Build.Step.Run {
    const check = b.addSystemCommand(&.{"sh"});
    check.addFileArg(b.path("tools/table-block.sh"));
    check.addFileArg(tool_out);
    check.addFileArg(b.path(readme_path));
    check.addArg(block);
    check.addArg(b.fmt("{d}", .{floor}));
    for (table_headers) |header| check.addArg(header);
    check.setCwd(b.path("."));
    return check;
}

/// A step that fails unless the ReleaseFast training run's own maximum resident
/// set size is at or under `peak_rss_budget`.
///
/// The binary is timed directly, and that is the whole reason this gate is not
/// built out of a `zig build train` run step. `zig build` runs the binary as a
/// grandchild, and the grandchild's peak does not reach the number `/usr/bin/time`
/// reports for its own child: timed that way the same binary read 570,769,408
/// bytes before the allocator split and 561,856,512 after it, both of which are
/// mostly the build and neither of which has anything to do with the run. A gate
/// built that way would have watched 3.72 GiB happen and printed a number two
/// orders of magnitude under it, so `addArtifactArg` hands the path of the built
/// binary to the script and the script execs it as its own child. This is the
/// one gate that needs a grandchild, and it is a measurement of the process
/// rather than of the build around it.
///
/// `sh` and `sed` because the alternatives are worse. Parsing this in the build
/// script means reading the tool's report through a Zig build API that is not
/// designed to report anything, and hardcoding `/usr/bin/time`'s layout into Zig
/// rather than into the one place that already knows about it. The number is
/// taken with one `sed` rule rather than `awk`'s `%d` conversion, because a
/// `float`-typed print of a value over 2^31 is a formatting question and this is
/// not the place to have one; the rule captures digits and nothing else.
///
/// The environment variable is not an afterthought. A run on a host whose libm
/// differs from the one that produced `outputs/loss.csv` exits non-zero on the
/// curve digest, and a memory gate that inherited that failure would fire on
/// every other contributor's machine while saying nothing about memory. The run
/// is asked, explicitly, to tolerate a curve difference it is not here to check,
/// and the variable does not promote anything: the committed curve is still only
/// replaced on a digest match. Every other non-zero exit is still this gate's
/// failure, and it prints the run's own output, because a run that crashed
/// before finishing has a small peak and would otherwise pass.
///
/// Darwin and Linux both measure, and the gate used to be Darwin only. That was
/// a real red, not a hypothetical one: the skip announced itself on stderr, and
/// `.github/workflows/ci.yml` asserts that a passing `zig build verify` writes no
/// bytes to either stream, on `ubuntu-latest`. The commit that introduced the
/// gate never ran CI, so the two facts were never in the same place.
///
/// The Linux branch reports `ru_maxrss` in KiB through a `peak %M` marker, since
/// the budget is in bytes and the run's own stderr shares the file. If
/// `/usr/bin/time` is absent on Linux the gate **fails** rather than skipping: CI
/// runs there, so a skip would be a green run that checked nothing, and this
/// repository does not ship a gate that cannot fail. `ci.yml` installs `time` so
/// that does not happen on a clean runner.
///
/// The only silent exit was a platform that is neither, and it was silent on
/// purpose and wrong. The reason given was that a line there breaks the silence
/// contract CI asserts on every run -- but that contract is about a PASSING run,
/// and this branch turned a run that measured nothing into a passing one, so the
/// silence was the defect rather than the design. It now fails, in the same
/// shape as the missing-`time` case above, and the check is cheap: all three
/// jobs in `ci.yml` are `ubuntu-latest` and the only other platform anyone
/// builds this on is Darwin, so the branch is unreachable rather than a new red.
fn addPeakRssCheck(b: *std.Build, train_exe: *std.Build.Step.Compile) *std.Build.Step.Run {
    const check = b.addSystemCommand(&.{
        "sh",
        "-c",
        \\t=$(mktemp) || exit 1
        \\case $(uname -s) in
        \\Darwin)
        \\  ZTRANSFORMER_ACCEPT_LOSS_DIFFERENCE=1 /usr/bin/time -l "$1" train >/dev/null 2>"$t"
        \\  st=$?
        \\  peak=$(sed -n 's/^[[:space:]]*\([0-9][0-9]*\)[[:space:]].*maximum resident set size.*/\1/p' "$t")
        \\  how="/usr/bin/time -l, ReleaseFast ztransformer-train, cwd the build root"
        \\  ;;
        \\Linux)
        \\  if [ ! -x /usr/bin/time ]; then
        \\    echo "zig build peak-rss: /usr/bin/time is not on this host, so peak memory is" >&2
        \\    echo "UNCHECKED here, and a gate that cannot run is a claim. Install it:" >&2
        \\    echo "  apt-get install -y time     # Debian, Ubuntu" >&2
        \\    echo "  dnf install -y time         # Fedora" >&2
        \\    rm -f "$t"; exit 1
        \\  fi
        \\  # GNU time reports %M in KiB and the budget is in bytes. The marker
        \\  # keeps this off any other bare-number line, because the run's own
        \\  # stderr lands in the same file.
        \\  ZTRANSFORMER_ACCEPT_LOSS_DIFFERENCE=1 /usr/bin/time -f 'peak %M' "$1" train >/dev/null 2>"$t"
        \\  st=$?
        \\  peak=$(sed -n 's/^peak \([0-9][0-9]*\)$/\1/p' "$t")
        \\  [ -n "$peak" ] && peak=$((peak * 1024))
        \\  how="/usr/bin/time -f, ReleaseFast ztransformer-train, ru_maxrss in KiB"
        \\  ;;
        \\*)
        \\  # Neither platform this repository is built on, so there is no
        \\  # /usr/bin/time invocation to make and no number to compare. It
        \\  # fails anyway, for the reason the Linux case above gives: a gate
        \\  # that cannot run is a claim. This branch used to `rm -f "$t"; exit
        \\  # 0`, justified by `verify`'s silence contract -- which is a claim
        \\  # about a PASSING run, and this branch was how a run that measured
        \\  # nothing passed. The silence was the defect, not the design. And it
        \\  # is unreachable rather than a new red: all three jobs in ci.yml are
        \\  # ubuntu-latest, and Darwin is the only other platform this is built
        \\  # on, so both of them take a branch above.
        \\  echo "zig build peak-rss: $(uname -s) is neither Darwin nor Linux, so" >&2
        \\  echo "peak memory is UNCHECKED here, and a gate that cannot run is" >&2
        \\  echo "a claim. The two platforms this gate measures are the two this" >&2
        \\  echo "repository is built on. Add a branch to the case above before" >&2
        \\  echo "trusting this gate anywhere else." >&2
        \\  rm -f "$t"; exit 1
        \\  ;;
        \\esac
        \\if [ "$st" -ne 0 ]; then
        \\  echo "zig build peak-rss: the training run exited $st. A run that did not finish is not a measurement." >&2
        \\  cat "$t" >&2; rm -f "$t"; exit 1
        \\fi
        \\rm -f "$t"
        \\if [ -z "$peak" ]; then
        \\  echo "zig build peak-rss: $how reported no peak, so there is nothing to compare the budget against." >&2
        \\  exit 1
        \\fi
        \\if [ "$peak" -le "$2" ]; then exit 0; fi
        \\echo "zig build peak-rss: the training run peaked over its budget." >&2
        \\echo "  budget   $2 bytes, the constant at the top of build.zig" >&2
        \\echo "  peak     $peak bytes" >&2
        \\echo "  measured $how" >&2
        \\echo "Read 'The allocator train.run is handed' in src/README.md. A peak this far over is an allocator that never gives the memory back: init.arena's free is a no-op, and handing train.run init.gpa is the fix. A peak a little over the budget is a larger working set rather than this defect, and the fix is to raise the constant in build.zig with a measured run added to the table beside it." >&2
        \\exit 1
        ,
        "sh",
    });
    check.addArtifactArg(train_exe);
    // The budget, as `$2`, so the script compares against the one constant at
    // the top of this file and there is no second copy of the number in it.
    check.addArg(peak_rss_budget);
    check.setCwd(b.path("."));
    return check;
}

/// Reports the ReleaseFast training run's CPU seconds per step, median of N.
///
/// The step count is read from the run rather than written here, so the figure
/// is total CPU over the steps the run actually did.
///
/// Reports, never gates; `addPeakRssCheck` explains why the two differ. CPU
/// time rather than wall clock because this box is shared and wall clock tracks
/// whatever else is running, not this program. `/usr/bin/time` is used for the
/// same reason the memory gate uses it: `zig build` makes the binary a
/// grandchild and the number never reaches the build runner.
///
/// That distinction is load-bearing and the first version of this script got it
/// wrong in a way that is invisible from its own output. macOS `/usr/bin/time
/// -l` prints one line holding all three columns -- `1.00 real  0.00 user  0.00
/// sys` -- so a pattern of `\([0-9.]*\).*user.*` matches that line and captures
/// the **first** number on it, which is `real`. It read 1.00 on `sleep 1` and
/// the step time came out as wall clock under a label saying CPU, in two
/// READMEs. The pattern now anchors on the column and on the rest of the line --
/// `[ \t]\([0-9.]*\) user[ \t]*[0-9.]*[ \t]*sys` -- so the number captured is
/// the one that precedes the word, a program line that happens to say "user"
/// cannot match, and `tail -1` means a second match collapses to one value
/// instead of turning the sample list into two rows per run and printing a
/// plausible wrong median. The single-space spelling of that pattern looks
/// equivalent and matches nothing at all, because macOS pads every column to
/// its own width; it was tried first and read empty.
///
/// The Linux branch calls `time` with a literal format rather than an
/// unquoted `$var`. `-f user %U` held in a variable and expanded unquoted is
/// three words under `sh`, so GNU time read `user` as the format, `%U` as the
/// command, and exited 127 -- on the platform CI runs on, where the step had
/// never been executed.
///
/// `ZTRANSFORMER_ACCEPT_LOSS_DIFFERENCE=1` for the reason the memory gate sets
/// it: a libm difference on another host must not turn a timing run into a
/// failure. The curve digest is not this step's business, and nothing is
/// promoted by it either way beyond what `train` always does on a digest match.
///
/// It runs the whole training run N times, tokenizer startup included, and
/// divides by the steps the run reports rather than by a constant here. So the
/// figure is "CPU seconds per step of this run", and on a longer corpus the
/// fixed startup cost dilutes it. That is the right property for comparing one
/// code change against another and the wrong one for a model-wide cost model,
/// which is what `zig build scale-profile` is for.
fn addBench(b: *std.Build, train_exe: *std.Build.Step.Compile) *std.Build.Step.Run {
    const run = b.addSystemCommand(&.{
        "sh",
        "-c",
        \\if [ ! -x /usr/bin/time ]; then
        \\  echo "zig build bench: /usr/bin/time is not on this host, so there is no" >&2
        \\  echo "measurement to report. It runs on Darwin and Linux." >&2
        \\  exit 1
        \\fi
        \\t=$(mktemp) || exit 1
        \\i=0
        \\while [ "$i" -lt "$2" ]; do
        \\  i=$((i + 1))
        \\  case $(uname -s) in
        \\  Darwin) out=$(ZTRANSFORMER_ACCEPT_LOSS_DIFFERENCE=1 /usr/bin/time -l "$1" train 2>&1) ;;
        \\  Linux)  out=$(ZTRANSFORMER_ACCEPT_LOSS_DIFFERENCE=1 /usr/bin/time -f 'user %U' "$1" train 2>&1) ;;
        \\  *)
        \\    echo "zig build bench: unsupported platform for /usr/bin/time." >&2
        \\    rm -f "$t"; exit 1
        \\    ;;
        \\  esac
        \\  st=$?
        \\  if [ "$st" -ne 0 ]; then
        \\    echo "zig build bench: run $i exited $st, which is not a measurement:" >&2
        \\    echo "$out" >&2; rm -f "$t"; exit 1
        \\  fi
        \\  steps=$(printf '%s\n' "$out" | sed -n 's/^steps \([0-9][0-9]*\)$/\1/p' | tail -1)
        \\  case $(uname -s) in
        \\  Darwin) c=$(printf '%s\n' "$out" | sed -n 's/.*[ \t]\([0-9.]*\) user[ \t]*[0-9.]*[ \t]*sys.*/\1/p' | tail -1) ;;
        \\  Linux)  c=$(printf '%s\n' "$out" | sed -n 's/^user \([0-9.]*\)$/\1/p' | tail -1) ;;
        \\  esac
        \\  if [ -z "$steps" ] || [ -z "$c" ]; then
        \\    echo "zig build bench: run $i reported steps='$steps' cpu='$c'; a run that" >&2
        \\    echo "does not print both is not a measurement. Refusing to divide." >&2
        \\    rm -f "$t"; exit 1
        \\  fi
        \\  echo "$c" >> "$t"
        \\done
        \\runs=$(wc -l < "$t" | tr -d ' ')
        \\if [ "$runs" -ne "$2" ] || [ -z "$steps" ] || [ -z "$c" ]; then
        \\  echo "zig build bench: $runs of $2 runs reported a usable figure." >&2
        \\  echo "steps='$steps' cpu='$c'. Zero runs is not a median." >&2
        \\  rm -f "$t"; exit 1
        \\fi
        \\median=$(sort -g "$t" | awk '{a[NR]=$1} END {print (NR%2) ? a[(NR+1)/2] : (a[NR/2]+a[NR/2+1])/2}')
        \\lo=$(sort -g "$t" | head -1); hi=$(sort -g "$t" | tail -1)
        \\awk -v m="$median" -v lo="$lo" -v hi="$hi" -v s="$steps" -v n="$2" 'BEGIN {
        \\  printf "bench  %d run%s of %s steps, user CPU seconds per step\n", n, (n==1?"":"s"), s
        \\  printf "       median %.4f   min %.4f   max %.4f\n", m/s, lo/s, hi/s
        \\  printf "       spread %.1f%% of the median\n", (m>0 ? 100*(hi-lo)/m : 0)
        \\  printf "       whole run including tokenizer startup, over the steps it did\n"
        \\}'
        \\rm -f "$t"
        ,
        "sh",
    });
    run.addArtifactArg(train_exe);
    // The iteration count, so `zig build bench -Dbench-runs=9` is one argument
    // rather than a second copy of the number written into this file.
    const n = b.option(usize, "bench-runs", "Iterations for `zig build bench` (default 3)") orelse 3;
    run.addArg(b.fmt("{d}", .{n}));
    run.setCwd(b.path("."));
    return run;
}
/// A step that fails unless one committed file still hashes to the digest named
/// beside its path at the top of this file.
///
/// A `sha256` tool rather than a Zig hash here, and the reason is placement: as
/// a build step this can fail the way a build step is supposed to, with a non-zero
/// exit and its own message naming the file that did not match. Hashing in the
/// build script meant `std.process.exit`, which killed the build runner during
/// configuration and took `zig build --help` down with it.
///
/// The tool is chosen, not assumed. This said "`shasum` is present on macOS and
/// on the GitHub ubuntu runners this repository targets", and the second half of
/// that was never true: `shasum` is a Perl script that ships with macOS, while
/// Linux ships GNU `sha256sum`. So the corpus and loss-curve digests failed on
/// Linux with `shasum: command not found`, on
/// the platform `.github/workflows/ci.yml` runs, and the comment asserting
/// otherwise is why it went unnoticed. `sha256sum` is preferred and `shasum -a
/// 256` is the fallback; a machine with neither fails loudly.
///
/// The redirect is the other half. `verify` is silent on success by contract, and
/// both tools print `<path>: OK` on the way past, so the success path is
/// swallowed and the failure path is not. The failure names both sides: the
/// expected digest this file was committed with, and the digest the bytes on disk
/// actually have, which is the one number that says whether the artifact drifted
/// or is simply not the one that was committed.
/// Every parameter is a compile-time literal, which is what lets the shell
/// command below be assembled at comptime: the four call sites are the whole
/// configuration, so there is nothing to pass at runtime and nothing that can
/// disagree with the constant it is meant to check.
fn addDigestCheck(
    b: *std.Build,
    comptime label: []const u8,
    comptime path: []const u8,
    comptime sha256: []const u8,
) *std.Build.Step.Run {
    const check = b.addSystemCommand(&.{
        "sh",
        "-c",
        "if command -v sha256sum >/dev/null 2>&1; then h() { sha256sum \"$@\"; }; " ++
            "elif command -v shasum >/dev/null 2>&1; then h() { shasum -a 256 \"$@\"; }; " ++
            "else echo '" ++ label ++ ": no sha256 tool on this host (looked for sha256sum and shasum).'" ++
            " >&2; exit 1; fi; " ++
            "want='" ++ sha256 ++ "  " ++ path ++ "'; " ++
            "if echo \"$want\" | h -c - >/dev/null 2>&1; then exit 0; " ++
            "else echo '" ++ label ++ " digest check failed:' >&2; " ++
            "echo \"expected  $want\" >&2; " ++
            "h '" ++ path ++ "' >&2; exit 1; fi",
    });
    check.setCwd(b.path("."));
    return check;
}
