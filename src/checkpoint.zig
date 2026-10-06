//! Raw f32 checkpoint: one file, no format.
//!
//! Clause 1 (Llama-3 block): persists the weights `train.run` returns so a run
//! leaves a model and not only a curve. Both readers are ours, so the layout is
//! little-endian f32 straight out of memory behind a 64-byte header, the same
//! choice `src/cuda/norm_twin.zig` makes for its blobs: magic `ZTR1`, version 1,
//! then the seven `model.Config` fields as u64 LE, then every parameter in
//! `train.flatten` order (tok_embed, each layer's nine, final_norm).
//!
//! `load` allocates via `model.initParams` and overwrites, so shapes stay in one
//! place. A file whose byte count disagrees is refused rather than truncated.
const std = @import("std");
const Io = std.Io;
const model = @import("model.zig");

comptime {
    if (@import("builtin").cpu.arch.endian() != .little) {
        @compileError("checkpoint blob format is little-endian f32");
    }
}

pub const magic = "ZTR1";
pub const version: u32 = 1;
pub const header_len = 64;

pub const Loaded = struct {
    cfg: model.Config,
    params: model.Params,
};

fn paramCount(cfg: model.Config) usize {
    const d = model.dModel(cfg);
    const h = model.ffnDim(cfg);
    const kv = cfg.n_kv_heads * cfg.head_dim;
    return cfg.vocab_size * d + cfg.n_layers * (d + d * d + 2 * d * kv + d * d + d + 2 * d * h + h * d) + d;
}

pub fn save(io: Io, path: []const u8, cfg: model.Config, params: model.Params) !void {
    const file = try Io.Dir.cwd().createFile(io, path, .{});
    defer file.close(io);
    var buffer: [4096]u8 = undefined;
    var out: Io.File.Writer = .init(file, io, &buffer);
    const w = &out.interface;
    var header: [header_len]u8 = undefined;
    @memcpy(header[0..4], magic);
    std.mem.writeInt(u32, header[4..8], version, .little);
    const fields = [_]usize{ cfg.n_layers, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.n_ctx, cfg.vocab_size, cfg.ffn_mult };
    for (fields, 0..) |v, i| std.mem.writeInt(u64, header[8 + 8 * i ..][0..8], @intCast(v), .little);
    try w.writeAll(&header);
    try writeTensor(w, params.tok_embed);
    for (params.layers) |l| {
        try writeTensor(w, l.attn_norm);
        try writeTensor(w, l.wq);
        try writeTensor(w, l.wk);
        try writeTensor(w, l.wv);
        try writeTensor(w, l.wo);
        try writeTensor(w, l.mlp_norm);
        try writeTensor(w, l.w_gate);
        try writeTensor(w, l.w_up);
        try writeTensor(w, l.w_down);
    }
    try writeTensor(w, params.final_norm);
    try w.flush();
}

fn writeTensor(w: *std.Io.Writer, t: @import("tensor.zig").Tensor) !void {
    try w.writeAll(std.mem.sliceAsBytes(t.data));
}

pub fn load(allocator: std.mem.Allocator, io: Io, path: []const u8) !Loaded {
    const bytes = try Io.Dir.cwd().readFileAlloc(io, path, allocator, .unlimited);
    defer allocator.free(bytes);
    if (bytes.len < header_len) return error.CheckpointShape;
    if (!std.mem.eql(u8, bytes[0..4], magic)) return error.CheckpointShape;
    if (std.mem.readInt(u32, bytes[4..8], .little) != version) return error.CheckpointShape;
    var fields: [7]usize = undefined;
    for (0..7) |i| fields[i] = @intCast(std.mem.readInt(u64, bytes[8 + 8 * i ..][0..8], .little));
    const cfg: model.Config = .{
        .n_layers = fields[0],
        .n_heads = fields[1],
        .n_kv_heads = fields[2],
        .head_dim = fields[3],
        .n_ctx = fields[4],
        .vocab_size = fields[5],
        .ffn_mult = fields[6],
    };
    try model.validate(cfg);
    const want = header_len + paramCount(cfg) * 4;
    if (bytes.len != want) return error.CheckpointShape;
    var params = try model.initParams(allocator, cfg, 0);
    errdefer params.deinit();
    var off: usize = header_len;
    off = copyInto(params.tok_embed.data, bytes, off);
    for (params.layers) |*l| {
        off = copyInto(l.attn_norm.data, bytes, off);
        off = copyInto(l.wq.data, bytes, off);
        off = copyInto(l.wk.data, bytes, off);
        off = copyInto(l.wv.data, bytes, off);
        off = copyInto(l.wo.data, bytes, off);
        off = copyInto(l.mlp_norm.data, bytes, off);
        off = copyInto(l.w_gate.data, bytes, off);
        off = copyInto(l.w_up.data, bytes, off);
        off = copyInto(l.w_down.data, bytes, off);
    }
    off = copyInto(params.final_norm.data, bytes, off);
    return .{ .cfg = cfg, .params = params };
}

fn copyInto(dst: []f32, bytes: []const u8, off: usize) usize {
    @memcpy(std.mem.sliceAsBytes(dst), bytes[off..][0 .. dst.len * 4]);
    return off + dst.len * 4;
}
