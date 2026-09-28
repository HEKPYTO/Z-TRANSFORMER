const std = @import("std");
/// The library source, read for the one string `verify` asserts on. Taken from
/// the same file the binary is built from, so a version bump cannot leave a
/// stale expectation behind here.
const lib_source = @import("src/lib.zig");

/// The corpus and the digest `data/README.md` documents for it. A corpus that
/// changes silently turns every number derived from it into a lie.
const corpus_path = "data/tinyshakespeare.txt";
const corpus_sha256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed";

pub fn build(b: *std.Build) void {
    // Before the graph, not as one of its steps, because this is the one place
    // every entry point passes through. `zig build run -- train` reads the
    // corpus too, and a check only `verify` ran would leave that path reading a
    // drifted corpus unguarded.
    checkCorpusDigest(b);

    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

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
    const train_step = b.step("train", "Train and write outputs/loss.csv");
    train_step.dependOn(&train_run.step);

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
    release_tests.step.dependOn(&fmt.step);
    verify_step.dependOn(&fmt.step);
    verify_step.dependOn(&release_tests.step);

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

/// Fails the build unless the corpus still hashes to the digest
/// `data/README.md` documents.
///
/// The message names both digests, because "the check failed" does not say
/// which side moved, and the failure is a non-zero exit with that message and
/// nothing else: a stack trace here would bury the two lines that say what to
/// fix.
fn checkCorpusDigest(b: *std.Build) void {
    const bytes = std.Io.Dir.cwd().readFileAlloc(
        b.graph.io,
        b.path(corpus_path).getPath2(b, null),
        b.allocator,
        .unlimited,
    ) catch |err| {
        std.debug.print("\n{s} cannot be read: {s}\n", .{ corpus_path, @errorName(err) });
        std.process.exit(1);
    };
    var digest: [std.crypto.hash.sha2.Sha256.digest_length]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    const hex = std.fmt.bytesToHex(digest, .lower);
    if (std.mem.eql(u8, &hex, corpus_sha256)) return;
    std.debug.print(
        "\n{s} does not hash to the digest data/README.md documents.\n" ++
            "  expected  {s}\n  actual    {s}\n",
        .{ corpus_path, corpus_sha256, &hex },
    );
    std.process.exit(1);
}
