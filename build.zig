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
/// `zig build verify` checks it, and that proves exactly one thing: the curve in
/// the tree is the curve this repository documents. The other half — that the
/// run can produce those bytes — is `zig build train`, which writes a pending
/// file and only replaces this one on a byte-for-byte match.
const loss_csv_path = "outputs/loss.csv";
const loss_csv_sha256 = "f1dd54445064810c28002dcacaf23b4bc82bb1e6ecfa28f5ed91e7fa4518f792";

/// `tools/removed/report.csv`: the 156 tensor rows of the Llama
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
const removed_report_path = "tools/removed/report.csv";
const removed_report_sha256 = "c91abe2e924883b519ecfa537f015de1cf4d7e034fcbfd92828e21d139c8d30e";

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
    const parity_digest_step = b.step("removed-digest", "Check tools/removed/report.csv against the committed digest");
    parity_digest_step.dependOn(&addDigestCheck(b, "committed parity report", removed_report_path, removed_report_sha256).step);

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

    // `zig build scale-profile` prints the projections behind the deferred work.
    // The same Debug binary as `run`, because it runs no arithmetic of its own
    // over a tensor: it is Config arithmetic and a table, so there is nothing
    // here for ReleaseFast to make faster. No corpus dependency, so it works in
    // a clone that has not fetched one, and it writes nothing.
    const scale_run = b.addRunArtifact(exe);
    scale_run.addArg("scale-profile");
    const scale_step = b.step("scale-profile", "Print the projected cost of shapes this model cannot be run at");
    scale_step.dependOn(&scale_run.step);

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
    const verify_step = b.step("verify", "Check fmt, tests in Debug and ReleaseFast, the banner, the corpus digest, the committed loss curve, and the scale tables in src/README.md");

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

/// A step that fails unless one committed file still hashes to the digest named
/// beside its path at the top of this file.
///
/// `shasum -c` rather than a Zig hash here, and the reason is placement: as a
/// build step this can fail the way a build step is supposed to, with a non-zero
/// exit and its own message naming the file that did not match. Hashing in the
/// build script meant `std.process.exit`, which killed the build runner during
/// configuration and took `zig build --help` down with it.
///
/// The redirect is the other half. `verify` is silent on success by contract, and
/// `shasum -c` prints `<path>: OK` on the way past, so the success path is
/// swallowed and the failure path is not. The failure names both sides: the
/// expected digest this file was committed with, and the digest the bytes on disk
/// actually have, which is the one number that says whether the artifact drifted
/// or is simply not the one that was committed. `shasum` is present on macOS and
/// on the GitHub ubuntu runners this repository targets; a machine without it
/// fails the step loudly rather than skipping the check.
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
        "want='" ++ sha256 ++ "  " ++ path ++ "'; " ++
            "if echo \"$want\" | shasum -a 256 -c - >/dev/null 2>&1; then exit 0; " ++
            "else echo '" ++ label ++ " digest check failed:' >&2; " ++
            "echo \"expected  $want\" >&2; " ++
            "shasum -a 256 '" ++ path ++ "' >&2; exit 1; fi",
    });
    check.setCwd(b.path("."));
    return check;
}
