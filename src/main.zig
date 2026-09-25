const std = @import("std");
const Io = std.Io;
const lib = @import("ztransformer");

pub fn main(init: std.process.Init) !void {
    var buffer: [128]u8 = undefined;
    var stdout: Io.File.Writer = .init(.stdout(), init.io, &buffer);
    try stdout.interface.print("{s} {s}\n", .{ lib.name(), lib.version });
    try stdout.interface.flush();
}
