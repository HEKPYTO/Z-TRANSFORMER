const std = @import("std");
const lib = @import("ztransformer");

test "version is non-empty" {
    try std.testing.expect(lib.version.len > 0);
}

test "name returns the project name" {
    try std.testing.expectEqualStrings("Z-TRANSFORMER", lib.name());
}

comptime {
    _ = @import("tensor_test.zig");
    _ = @import("norm_test.zig");
    _ = @import("rope_test.zig");
    _ = @import("mlp_test.zig");
    _ = @import("attention_test.zig");
    _ = @import("loss_test.zig");
}

comptime {
    _ = @import("tokenizer_test.zig");
    _ = @import("data_test.zig");
    _ = @import("optim_test.zig");
    _ = @import("model_test.zig");
    _ = @import("autograd_test.zig");
    _ = @import("gradcheck_test.zig");
    _ = @import("scale_test.zig");
    _ = @import("attn_bench_test.zig");
}

comptime {
    _ = @import("train_test.zig");
}

comptime {
    _ = @import("removed_test.zig");
}

// `src/cuda/device.zig` lives under `src/cuda/` and holds its tests inline, which
// looks like it should be a `src/cuda/device_test.zig` and is deliberately not.
// The walk test above opens `src` non-recursively, so a test file under
// `src/cuda/` would be invisible to it AND unregistered, and the failure mode of
// that is the one this file exists to prevent: tests that never run, reported
// green. Importing the file itself from here is what actually builds its tests,
// and `src/` can reach `src/cuda/` -- the module-root restriction runs the other
// way, and is why the file imports nothing but `std`.
//
// Its tests are the pure size arithmetic only. The `extern fn` declarations are
// unreferenced there, so nothing is linked and the test binary needs no CUDA.
comptime {
    _ = @import("cuda/device.zig");
}

// The KV cache. In `src/` rather than `src/cuda/` because it is pure Zig with no
// device in it at all, and because the walk test above opens `src` -- a
// `src/kv_cache_test.zig` would be the right file by that test's own convention and
// is deliberately not used, for the same reason `cuda/device.zig` is imported
// directly: one mechanism, not two.
comptime {
    _ = @import("kv_cache.zig");
}

// Every `src/*_test.zig` is named by one of the blocks above, and a file that
// is not named there compiles to nothing: no test in it is ever built, and
// `zig build test` stays green. That is the failure this exists to catch.
//
// The needle is the whole `@import("...")` and not the bare file name, because
// the reference below is this file's own text and this file discusses itself in
// prose: it spells `src/cuda/device_test.zig` and `src/kv_cache_test.zig` in
// the comments above, to say why neither exists. A bare-name search found them
// there, so creating either one -- tests that never run, reported green, the
// exact failure this exists to catch -- left the walk passing and the build
// green. The quoted form also stops `a_test.zig` from being satisfied by
// `ba_test.zig`, which a substring search over the two of them cannot.
//
// The scope is test files, and saying so is the point: `src/cuda/norm_twin.zig`
// is 337 lines of Zig under `src/` that this walk cannot see and no build
// target compiles. It is not dead -- `src/cuda/run-norm.sh` builds it as its
// own module root -- but nothing in `zig build test` or CI checks it, which is
// why `AGENTS.md` says no CI step reaches it. (`zig build cuda-check` compiles
// `norm.cu` and `probe.cu` under the pinned toolchain, but not this file, which
// is a module root only for run-norm.sh, and it is not in `verify` either.")
//
// Zig 0.16 has no comptime filesystem: `std.fs.cwd` is gone with the rest of
// the pre-`std.Io` API, and every `std.Io.Dir` call needs an `Io` instance,
// which is a runtime value. A directory cannot be enumerated at comptime, so
// the enumeration is at runtime and the reference is this file's own source,
// embedded at comptime. A test rather than a compile error, which means a
// cached run can skip it; the file it is looking for is by definition not an
// input to the build, so a cache hit is possible and a developer who adds a
// test file may need to re-run before believing a green.
test "every test file in src is named by the collection above" {
    const source = @embedFile("tests.zig");
    const io = std.testing.io;
    const dir = try std.Io.Dir.cwd().openDir(io, "src", .{ .iterate = true });
    defer dir.close(io);

    var it = dir.iterate();
    var named: usize = 0;
    while (try it.next(io)) |entry| {
        if (!std.mem.endsWith(u8, entry.name, "_test.zig")) continue;
        named += 1;
        var needle: [96]u8 = undefined;
        const quoted = try std.fmt.bufPrint(&needle, "@import(\"{s}\")", .{entry.name});
        if (std.mem.indexOf(u8, source, quoted) == null) {
            std.debug.print(
                "\n{s} holds tests but no comptime block above names it, so none of them are built\n",
                .{entry.name},
            );
            return error.TestUnreferencedTestFile;
        }
    }
    // An empty listing means the walk found the wrong directory and every
    // assertion above would have passed vacuously.
    try std.testing.expect(named > 0);
}
