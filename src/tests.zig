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
}

comptime {
    _ = @import("train_test.zig");
}

comptime {
    _ = @import("removed_test.zig");
}

// Every `src/*_test.zig` is named by one of the blocks above, and a file that
// is not named there compiles to nothing: no test in it is ever built, and
// `zig build test` stays green. That is the failure this exists to catch.
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
        if (std.mem.indexOf(u8, source, entry.name) == null) {
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
