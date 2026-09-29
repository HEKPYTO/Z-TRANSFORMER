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
    const corpus = addCorpusDigest(b);

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
    // a developer runs and in ReleaseFast, the version banner, and the corpus
    // digest checked above before the graph existed. Silent when it passes,
    // because a gate that prints on every green run trains everyone to ignore
    // it. Nothing here writes anything outside the build cache.
    const verify_step = b.step("verify", "Check fmt, tests in Debug and ReleaseFast, the banner, and the corpus digest");

    // `b.graph.zig_exe` rather than `zig` off `PATH`, so the formatter that
    // decides whether the tree is formatted is the same compiler running the
    // build. `--check` exits non-zero and names the files it would rewrite, and
    // the cwd is the build root so `.` is the repo wherever the command was
    // typed from.
    const fmt = b.addSystemCommand(&.{ b.graph.zig_exe, "fmt", "--check", "." });
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

/// A step that fails unless the corpus still hashes to the digest
/// `data/README.md` documents.
///
/// `shasum -c` rather than a Zig hash here, and the reason is placement: as a
/// build step this can fail the way a build step is supposed to, with a non-zero
/// exit and its own message naming the file that did not match. Hashing in the
/// build script meant `std.process.exit`, which killed the build runner during
/// configuration and took `zig build --help` down with it.
///
/// The redirect is the other half. `verify` is silent on success by contract, and
/// `shasum -c` prints `<path>: OK` on the way past, so the success path is
/// swallowed and the failure path is not: a drifted corpus still prints the file
/// name and the mismatched checksum. `shasum` is present on macOS and on the
/// GitHub ubuntu runners this repository targets; a machine without it fails the
/// step loudly rather than skipping the check.
fn addCorpusDigest(b: *std.Build) *std.Build.Step.Run {
    const check = b.addSystemCommand(&.{
        "sh",
        "-c",
        "want='" ++ corpus_sha256 ++ "  " ++ corpus_path ++ "'; " ++
            "if echo \"$want\" | shasum -a 256 -c - >/dev/null 2>&1; then exit 0; " ++
            "else echo 'corpus digest check failed:' >&2; " ++
            "echo \"$want\" | shasum -a 256 -c - >&2; exit 1; fi",
    });
    check.setCwd(b.path("."));
    return check;
}
