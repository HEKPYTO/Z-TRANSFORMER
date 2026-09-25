const std = @import("std");
const attention = @import("attention.zig");

test "attention module is wired" {
    _ = attention;
    try std.testing.expect(true);
}
