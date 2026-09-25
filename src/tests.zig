const std = @import("std");
const lib = @import("ztransformer");

test "version is non-empty" {
    try std.testing.expect(lib.version.len > 0);
}

test "name returns the project name" {
    try std.testing.expectEqualStrings("Z-TRANSFORMER", lib.name());
}
