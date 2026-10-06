const std = @import("std");
const model = @import("model.zig");
const checkpoint = @import("checkpoint.zig");

test "checkpoint round-trips byte-identical params" {
    const io = std.testing.io;
    const gpa = std.testing.allocator;
    const cfg = model.Config{ .n_layers = 1, .n_heads = 2, .n_kv_heads = 1, .head_dim = 4, .n_ctx = 16, .vocab_size = 32, .ffn_mult = 2 };
    var p = try model.initParams(gpa, cfg, 7);
    defer p.deinit();
    const path = ".zig-cache/checkpoint-roundtrip.bin";
    try std.Io.Dir.cwd().createDirPath(io, ".zig-cache");
    try checkpoint.save(io, path, cfg, p);
    var loaded = try checkpoint.load(gpa, io, path);
    defer loaded.params.deinit();
    try std.testing.expectEqual(cfg, loaded.cfg);
    // Every tensor, not a sample: save and load walk the same field order and
    // the test is what keeps them from desyncing one side at a time.
    try std.testing.expectEqualSlices(f32, p.tok_embed.data, loaded.params.tok_embed.data);
    try std.testing.expectEqualSlices(f32, p.final_norm.data, loaded.params.final_norm.data);
    for (p.layers, loaded.params.layers) |a, b| {
        try std.testing.expectEqualSlices(f32, a.attn_norm.data, b.attn_norm.data);
        try std.testing.expectEqualSlices(f32, a.wq.data, b.wq.data);
        try std.testing.expectEqualSlices(f32, a.wk.data, b.wk.data);
        try std.testing.expectEqualSlices(f32, a.wv.data, b.wv.data);
        try std.testing.expectEqualSlices(f32, a.wo.data, b.wo.data);
        try std.testing.expectEqualSlices(f32, a.mlp_norm.data, b.mlp_norm.data);
        try std.testing.expectEqualSlices(f32, a.w_gate.data, b.w_gate.data);
        try std.testing.expectEqualSlices(f32, a.w_up.data, b.w_up.data);
        try std.testing.expectEqualSlices(f32, a.w_down.data, b.w_down.data);
    }
}

test "checkpoint refuses a truncated file" {
    const io = std.testing.io;
    const gpa = std.testing.allocator;
    const cfg = model.Config{ .n_layers = 1, .n_heads = 2, .n_kv_heads = 1, .head_dim = 4, .n_ctx = 16, .vocab_size = 32, .ffn_mult = 2 };
    var p = try model.initParams(gpa, cfg, 7);
    defer p.deinit();
    const path = ".zig-cache/checkpoint-truncated.bin";
    try std.Io.Dir.cwd().createDirPath(io, ".zig-cache");
    try checkpoint.save(io, path, cfg, p);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .unlimited);
    defer gpa.free(bytes);
    const file = try std.Io.Dir.cwd().createFile(io, path, .{});
    defer file.close(io);
    var buffer: [4096]u8 = undefined;
    var out: std.Io.File.Writer = .init(file, io, &buffer);
    try out.interface.writeAll(bytes[0 .. bytes.len - 4]);
    try out.interface.flush();
    try std.testing.expectError(error.CheckpointShape, checkpoint.load(gpa, io, path));
}
