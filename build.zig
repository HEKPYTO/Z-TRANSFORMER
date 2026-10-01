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
/// libm, and Debug differs from release by about one f32 ulp per step. So a
/// reader on another host whose run produced different bytes has found a
/// different libm, not a broken build, and `zig build train` reports it in those
/// words and leaves this file alone rather than replacing it.
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

/// `tools/removed/report.csv`: the 204 tensor rows of the Llama
/// comparison, written by `tools/removed/oracle.txt` while
/// `sh tools/removed/check.sh` ran it against the pinned `reference-library` 4.57.3
/// and `torch` 2.14.0 reference in a repo-local virtualenv.
///
/// Deliberately not wired into `verify`, and the reason is not that hashing
/// needs Python. It does not. A `verify` that only hashes committed bytes
/// asserts that somebody committed those bytes, not that a comparison produces
/// them, and producing them is the entire claim. So the digest is checked at
/// the one moment the claim is testable, by the `removed-digest` step that
/// `check.sh` runs over the report it has just written — after the oracle, on a
/// green comparison, where the reader has already paid for the virtualenv. The
/// step exists rather than a constant inside `check.sh` so the digest has one
/// owner; it is not a second spelling of the comparison, which is why
/// `check.sh` still is the only entry point to that.
///
/// Two digests rather than one, because this one is reachable only on the
/// platform that committed it, and that is a property of the version string
/// rather than of the arithmetic. The report carries `max_abs_delta` per row
/// and a six-column environment block. The Linux torch wheel bakes its CUDA
/// build into the version it reports at runtime — `2.14.0+cu130` where the
/// committed record says `2.14.0` — while the macOS wheel is CPU-only and
/// reports the bare `2.14.0`. So on every Linux host, Fedora and
/// ubuntu-latest alike, the environment guard fires and these bytes are not
/// compared; on macOS they are. Measured on Fedora: the projection matched
/// exactly, all 18 kinds and 206 rows, and the byte digest was skipped for the
/// version string, not for a float.
///
/// What that costs is stated rather than hidden: the cross-host question for
/// `max_abs_delta` is not answered by this gate, on any host, ever. Nothing
/// here shows that two hosts round the delta the same way, and the two values
/// once quoted as evidence of a host difference were `l0.v` at `T=1` and `T=8`
/// in one host's own report. `addParityReportCheck` states the two halves: a
/// projection every host checks, and these bytes on the one host whose version
/// string matches.
const removed_report_path = "tools/removed/report.csv";
const removed_report_sha256 = "d0d501f0dcc5170b7b3bb8f324ed14badae9a2282b7f14245e80bdd8525586b1";

/// `report.csv` with the two groups of columns that are not a property of the
/// comparison removed: `max_abs_delta`, which is the float another host rounds
/// differently, and the six environment columns.
///
/// What survives is the claim rather than the measurement — which tensors were
/// compared, at which shapes, against which gate, each with which verdict, plus
/// the argmax row's own match and total. Every one of those is an integer or a
/// literal the oracle wrote, so the projection is byte-identical from any host
/// that runs the same comparison, and a report that is a perturbed export
/// rather than a comparison that passed cannot reproduce it: `sensitivity.sh`
/// flips verdicts to `FAIL`, and adding or dropping a row moves the set.
///
/// The carriage returns Python's csv writer terminates lines with are stripped
/// first, so the projection is about the report's content and not about which
/// line ending a `csv.DictWriter` happened to emit. They are the same on every
/// host today; the byte digest below is the check that says so.
///
/// What it gives up is the magnitude of each delta. On a host whose BLAS
/// differs, a `max_abs_delta` that moved from `5.96e-08` to `9.9e-06` — still
/// inside the gate, but two orders of magnitude worse — is invisible here, and
/// that is stated in the READMEs rather than left to be discovered. The
/// projection is a check on the gates, not on the arithmetic behind them.
const removed_report_projection_sha256 = "eb303e0b4d2a84c9c98c7af373dc165643df0058e4364b58ed6580a789db18f4";

/// The environment columns of the committed report, and the predicate for
/// running the byte digest over it.
///
/// The report records nothing else about the host that produced it, so this
/// string is the whole of what "the platform where the bytes are comparable"
/// can be tested against. A host reporting the same six columns is a host that
/// installed the same pins, and a byte difference there is one the report
/// cannot explain — which is a red, not a skip. A host reporting different ones
/// gets the projection and is told so on stdout, because a check that quietly
/// ran less than the one before it is a claim about itself nobody can check.
const removed_report_environment = "4.57.3,2.14.0,2.5.3,eager,float32,cpu";

/// What `zig build train` is allowed to peak at, in bytes, as macOS
/// `/usr/bin/time -l` reports the ReleaseFast binary's own maximum resident set
/// size. The gate is `zig build peak-rss`, and `verify` runs it.
///
/// This number is here because nothing measured peak memory for this
/// repository's entire history. The five block ops were checked tensor by
/// tensor against a external reference, the loss curve was gated on a byte
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
    const loss_csv = addDigestCheck(b, "committed loss curve", loss_csv_path, loss_csv_sha256);
    // Its own step, deliberately not reached from `verify`, for the reason on
    // `removed_report_sha256`. `check.sh` runs it.
    const parity_digest_step = b.step("removed-digest", "Check tools/removed/report.csv: the projection everywhere, the bytes where the environment matches the committed report");
    parity_digest_step.dependOn(&addParityReportCheck(b).step);

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
    // This step's entire purpose is the comparison, and a Debug build of
    // identical source does not reproduce the committed ReleaseFast curve, so
    // `settleCsv` refuses to promote it and exits 1. That guard is right for
    // `zig build train` and wrong here: without this the step failed on every
    // run, which is the guard working and the step being useless at the same
    // time. The flag acknowledges the difference and still does not replace the
    // committed curve, so `train` keeps the guard and only this step relaxes it,
    // and the Debug build is still covered by `zig build test`.
    dbg_train_run.setEnvironmentVariable("ZTRANSFORMER_ACCEPT_LOSS_DIFFERENCE", "1");
    const dbg_train_step = b.step("dbg-train", "Train in Debug, for a host-difference comparison");
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
    // The parity report's digest is the one that is not here, and its absence is
    // the point rather than a gap: it is written by the only command in this
    // repository that needs Python, so a reader can check every claim this gate
    // can check without installing anything. `sh tools/removed/check.sh` checks
    // the report it just wrote, and `zig build removed-digest` is the step it
    // calls to do it.
    const verify_step = b.step("verify", "Check fmt, tests in Debug and ReleaseFast, the banner, the corpus digest, the committed loss curve, the training run's peak memory, and the scale tables in src/README.md");

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

    // The scale tables `src/README.md` quotes, checked against the tool that
    // prints them. `scale-profile` writes no timestamp, no address and no float
    // whose formatting can drift, so two runs are byte-identical and the
    // comparison is a `cmp` rather than a judgment call. Without this the
    // README's claim to have been copied from the tool is a promise, and a
    // promise is what b168cf9 broke: it grew the activation cache, and six table
    // rows kept printing what the previous cache cost.
    verify_step.dependOn(&addScaleTableCheck(b, exe).step);

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
    // `sh src/cuda/run-norm.sh` on an NVIDIA host. This compiles the two `.cu`
    // files, so a syntax or type error in them is caught by the build system.
    //
    // Deliberately NOT in `verify`, and the reason is the same one that keeps
    // `removed-digest` out: `verify` is silent on success and CI asserts that, so
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

    // A committed parity report carrying a failing row is a corrupted artifact,
    // and `verify` used to pass on one. `check.sh` and `sensitivity.sh` both
    // write this file; run concurrently they leave a perturbed export behind,
    // and this repository did exactly that: 18 rows reading FAIL where the
    // committed baseline has none, and `verify` was green throughout.
    //
    // `removed-digest` would catch that too and is deliberately not in `verify`,
    // for the reason on `removed_report_sha256`: hashing committed bytes is not
    // the claim. This is a different check and does not conflict with that
    // decision. A comparison that passed writes no FAIL row, so their absence is
    // a property of a green run rather than of one byte sequence -- a legitimate
    // re-run that produces a different but equally valid report still passes,
    // which is exactly the case `removed-digest` would reject.
    const report_rows = b.addSystemCommand(&.{
        "sh",
        "-c",
        \\# `-r` and the case-insensitive match are both load-bearing, and both
        \\# were found by breaking this gate rather than by reading it. `grep -q`
        \\# exits 2 on a missing file, so the `if` below was false and the step
        \\# exited 0: a committed report that is *absent* passed the check
        \\# written to catch one that is corrupt. And `FAIL` is the oracle
        \\# writer's casing, not a contract this gate asserts, so a one-character
        \\# change to `fail` turned the gate into a no-op. And a readable but EMPTY
        \\# file satisfied `-r` while grep found no failing row in nothing, so an
        \\# empty export was green too. So there are three tests and they fail
        \\# closed: unreadable or absent, near-empty, and a failing row. What
        \\# they do NOT catch is a partial export that keeps several rows and
        \\# drops the rest -- nothing here compares the row count to 204 -- and
        \\# the digest check that would catch that is deliberately out of
        \\# `verify`, so this gate is narrower than the failure it names.
        \\if [ ! -r "$1" ]; then
        \\  echo "tools/removed/report.csv is missing or unreadable, so there is" >&2
        \\  echo "no committed comparison for this gate to inspect. A gate that" >&2
        \\  echo "cannot read its subject has not passed it." >&2
        \\  exit 1
        \\fi
        \\if [ ! -s "$1" ] || [ "$(grep -c . "$1")" -lt 3 ]; then
        \\  echo "tools/removed/report.csv has fewer than three lines, so it is" >&2
        \\  echo "empty or truncated rather than a comparison. Readable is not" >&2
        \\  echo "populated, and the failing-row grep below finds nothing in" >&2
        \\  echo "nothing, so without this both tests pass a zero-byte report." >&2
        \\  exit 1
        \\fi
        \\if grep -qi fail "$1"; then
        \\  echo "tools/removed/report.csv carries failing rows, so the committed" >&2
        \\  echo "report is a perturbed export rather than a comparison that" >&2
        \\  echo "passed. check.sh and sensitivity.sh both write this file:" >&2
        \\  grep -n FAIL "$1" >&2
        \\  echo "restore it, and do not run the two at once:" >&2
        \\  echo "  git checkout tools/removed/report.csv" >&2
        \\  exit 1
        \\fi
        ,
        "sh",
    });
    report_rows.addFileArg(b.path("tools/removed/report.csv"));
    report_rows.setCwd(b.path("."));
    verify_step.dependOn(&report_rows.step);

    // The banner, captured rather than inherited: `verify` has to be silent, and
    // an exact stdout match asserts more than the exit code alone, which is the
    // only thing a bare run step would check.
    const banner_run = b.addRunArtifact(exe);
    _ = banner_run.captureStdOut(.{});
    banner_run.expectStdOutEqual(b.fmt("{s} {s}\n", .{ lib_source.name(), lib_source.version }));
    verify_step.dependOn(&banner_run.step);
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

/// A step that fails unless the three tables `src/README.md` quotes are the
/// tables `zig build scale-profile` prints.
///
/// The tool's output is captured rather than inherited, for the same reason the
/// banner's is: `verify` is silent by contract, and a run step that prints the
/// whole profile on every green run trains everyone to scroll past it. The
/// README arrives as a plain file argument, so a failure diffs the two and names
/// the file that has to be re-copied.
///
/// The extraction is two `awk` rules and no more. A header line opens a table
/// and a blank line closes it, and the blank between two tables is emitted ahead
/// of the second so neither side can differ by a trailing newline. It keys on
/// the three header lines rather than on line numbers, so a row moving or a
/// column appearing reads as a diff rather than a misread.
fn addScaleTableCheck(b: *std.Build, exe: *std.Build.Step.Compile) *std.Build.Step.Run {
    const run = b.addRunArtifact(exe);
    run.addArg("scale-profile");
    const printed = run.captureStdOut(.{});

    const check = b.addSystemCommand(&.{
        "sh",
        "-c",
        \\tool=$1; readme=$2
        \\t=$(mktemp) || exit 1; r=$(mktemp) || exit 1
        \\awk '
        \\  /^ARITHMETIC, PROJECTED, GFLOP$/ {hdr=1}
        \\  /^BYTES, PROJECTED, GiB$/ {hdr=1}
        \\  /^VERDICTS, one line per deferred item, at 32 GiB of host memory$/ {hdr=1}
        \\  hdr {if (n++) print ""; on=1; hdr=0}
        \\  on && /^$/ {on=0; next}
        \\  on {print}
        \\' "$tool" > "$t"
        \\awk '
        \\  /<!-- scale-profile:begin -->/ {on=1; next}
        \\  /<!-- scale-profile:end -->/ {on=0; next}
        \\  on && /^```/ {next}
        \\  on {print}
        \\' "$readme" > "$r"
        \\if cmp -s "$t" "$r"; then rm -f "$t" "$r"; exit 0; fi
        \\echo "the scale tables in src/README.md are not what zig build scale-profile prints:" >&2
        \\echo "re-copy the block between <!-- scale-profile:begin --> and <!-- scale-profile:end -->" >&2
        \\diff -u "$t" "$r" >&2
        \\rm -f "$t" "$r"
        \\exit 1
        ,
        "sh",
    });
    check.addFileArg(printed);
    check.addFileArg(b.path("src/README.md"));
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
/// The only silent exit is a platform that is neither, and it is silent on
/// purpose. A line there would break the silence contract on a machine nobody
/// runs CI on, in exchange for telling a developer on that platform that this
/// gate does not cover their host. Two platforms is where the measurements were
/// taken and where the build runs; that is the whole of the claim.
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
        \\  # Neither platform this repository is built on. Silent is deliberate,
        \\  # and it is the one place in this file that is: a line here breaks
        \\  # `verify`'s silence contract, which CI asserts on every run, and a
        \\  # measurement that cannot be taken here must not become a CI failure
        \\  # on a platform nobody runs CI on. Darwin and Linux both measure.
        \\  rm -f "$t"; exit 0
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
/// Linux ships GNU `sha256sum`. So all three digests — the corpus, the loss curve
/// and the parity report — failed on Linux with `shasum: command not found`, on
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

/// A step that fails unless `tools/removed/report.csv` is the comparison the
/// repository committed, checked as far as the host it ran on allows.
///
/// Two checks, in this order, and the order is the design. The projection runs
/// everywhere and is the one that must not be weakened: it is the whole claim —
/// which tensors, which shapes, which gates, which verdicts, what argmax count —
/// with the two columns a different host owns taken out, so it is a statement
/// about the comparison rather than about the rounding of a float. The byte
/// digest runs only when the report's environment columns are the committed
/// ones, and is strictly stronger where it does: it catches the deltas moving
/// inside their gates, which the projection cannot see. A host that fails the
/// environment predicate has not got a weaker check, it has got the portable
/// one, and it is told that on stderr rather than left to assume.
///
/// Both live in one script because the predicate is a runtime fact about the
/// file, and the build graph has no conditionals: two steps cannot see each
/// other's result. That is also why the byte digest is spelled out here instead
/// of being a second `addDigestCheck` call — one command, one `exit`, and the
/// message on a red says which of the two fired, which two steps could not.
///
/// The sha256 tool selection is the same `sha256sum`-then-`shasum` dance
/// `addDigestCheck` spells out, for the same reason and with the same failure on
/// a host carrying neither. Duplicating three lines is the price of not editing
/// a function three other gates call; if this ever becomes the third copy,
/// hoist it to a comptime constant above both.
///
/// The projection keeps the header line, so a report that gained or lost a
/// column is a diff here rather than a silent shift of what `$8` means. Its
/// eight kept columns are positional (`$1..$6`, `$8`, `$9`), which is a
/// limitation rather than a choice: no field the oracle writes can hold a comma
/// — they are identifiers, small integers, a Python float `repr` of a gate, and
/// the literals `pass`, `FAIL`, `OK` — so `awk -F,` cannot mis-split one, and a
/// future column that could is a loud mismatch rather than a wrong answer.
fn addParityReportCheck(b: *std.Build) *std.Build.Step.Run {
    const check = b.addSystemCommand(&.{
        "sh",
        "-c",
        \\p=$(mktemp) || exit 1
        \\if command -v sha256sum >/dev/null 2>&1; then h() { sha256sum "$@"; };
        \\elif command -v shasum >/dev/null 2>&1; then h() { shasum -a 256 "$@"; };
        \\else echo "committed parity report: no sha256 tool on this host (looked for sha256sum and shasum)." >&2; exit 1; fi
        \\awk -F, 'NR == 1 { print; next } { print $1 "," $2 "," $3 "," $4 "," $5 "," $6 "," $8 "," $9 }' "$1" | tr -d '\r' > "$p" || { rm -f "$p"; exit 1; }
        \\want="$2"
        \\got=$(h "$p" | awk '{print $1}')
        \\if [ "$got" != "$want" ]; then
        \\  echo "committed parity report: the projection does not match, so this report is not" >&2
        \\  echo "the comparison that was committed, and no host's float rounding explains it." >&2
        \\  echo "A verdict, a gate, a row or the argmax count moved. sha256 over the eight" >&2
        \\  echo "kept columns of this file, then of the committed one:" >&2
        \\  echo "  expected  $want" >&2
        \\  echo "  actual    $got" >&2
        \\  echo "git diff tools/removed/report.csv shows which line moved. A report full of" >&2
        \\  echo "FAIL rows is sensitivity.sh's export rather than a comparison that passed:" >&2
        \\  echo "  git checkout tools/removed/report.csv" >&2
        \\  rm -f "$p"; exit 1
        \\fi
        \\rm -f "$p"
        \\# Every row's environment, not row 2's. This predicate decides whether
        \\# the byte digest runs at all, and it was read out of the file it is
        \\# checking, so editing one row's version columns made it disagree with
        \\# the committed record and switched the byte digest off -- an inflated
        \\# `max_abs_delta` then passed with every verdict still reading `pass`.
        \\# Found by breaking the gate, not by reading it. Rows that disagree
        \\# with each other are a malformed report, so this fails rather than
        \\# skips; a genuinely different host has every row agreeing, which is
        \\# the case the skip exists for.
        \\uniq=$(awk -F, 'NR > 1 { print $10 "," $11 "," $12 "," $13 "," $14 "," $15 }' "$1" | tr -d '\r' | sort -u)
        \\rows=$(printf '%s\n' "$uniq" | wc -l | tr -d ' ')
        \\if [ "$rows" -ne 1 ]; then
        \\  echo "committed parity report: the environment columns are not the same" >&2
        \\  echo "on every row ($rows distinct values), so this is not a report one" >&2
        \\  echo "oracle wrote. A row was edited to move the predicate below, which" >&2
        \\  echo "is the switch that decides whether the byte digest runs." >&2
        \\  printf '%s\n' "$uniq" | sed 's/^/  /' >&2
        \\  exit 1
        \\fi
        \\got=$uniq
        \\if [ "$got" != "$3" ]; then
        \\  echo "removed-digest: projection OK; byte digest NOT run, this host's oracle is not" >&2
        \\  echo "            the one the committed digest was taken over."
        \\  echo "  this report   $got" >&2
        \\  echo "  committed     $3" >&2
        \\  echo "The deltas here are unchecked against the committed ones, because they are" >&2
        \\  echo "the floats another host rounds its own way. The verdicts, gates, row set and" >&2
        \\  echo "argmax count were checked above, and those did match." >&2
        \\  exit 0
        \\fi
        \\if echo "$4  $1" | h -c - >/dev/null 2>&1; then
        \\  echo "removed-digest: projection OK; byte digest OK, environment matches the committed report."
        \\  exit 0
        \\fi
        \\echo "committed parity report: byte digest failed, and the environment columns are the" >&2
        \\echo "committed ones, so a version string does not account for it and neither does" >&2
        \\echo "the float difference between two hosts. Something else moved." >&2
        \\echo "  expected  $4  tools/removed/report.csv" >&2
        \\h "$1" >&2
        \\exit 1
        ,
        "sh",
    });
    check.setCwd(b.path("."));
    // The report by its path in the tree rather than as a build input, which is
    // what `addDigestCheck` does for the other two digests and is enough: a
    // system command with nothing captured has side effects, so the step runs on
    // every invocation and reads the file as it is now rather than as it was
    // when the manifest was written.
    check.addArg(removed_report_path);
    check.addArg(removed_report_projection_sha256);
    check.addArg(removed_report_environment);
    check.addArg(removed_report_sha256);
    return check;
}
