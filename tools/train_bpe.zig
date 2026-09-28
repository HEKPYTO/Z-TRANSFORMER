//! Trains a byte-level BPE vocabulary from a corpus and writes it as JSON.
//!
//!     zig build-exe --dep tokenizer -Mroot=tools/train_bpe.zig \
//!         -Mtokenizer=src/tokenizer.zig -femit-bin=tools/train_bpe
//!     ./tools/train_bpe data/tinyshakespeare.txt outputs/vocab.json
//!
//! The `-M` form is how this tool reaches src/tokenizer.zig. A module root
//! cannot import a file outside its own directory, so a named module is the
//! only way to share the implementation instead of copying it into tools/.

const std = @import("std");
const Tokenizer = @import("tokenizer").Tokenizer;

/// `train` is O(len * n_merges): 200 merges is 59 s on the 1.1 MB corpus in a
/// debug build and 1000 is 259 s, so 20000 is a run measured in hours, not a
/// default worth handing someone who did not ask for it. Pass it explicitly
/// when a long vocabulary is the point.
const default_merges = 200;

pub fn main(init: std.process.Init) !void {
    // Two allocators, because the peak is set by what training throws away
    // between passes. The gpa owns the tokenizer: on an arena `deinit` is a
    // no-op, so each pass leaves its id buffers and the grown pair-count table
    // resident for the rest of the run, and a leak inside training has nowhere
    // to show up. Measured on data/tinyshakespeare.txt at 200 merges, the arena
    // peaks at 26.4 MiB and the gpa at 18.0 MiB. The arena keeps the two
    // allocations that live for the whole run and are wanted back in one step at
    // the end: the corpus and the encoded ids.
    const gpa = init.gpa;
    const scratch = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(gpa);
    defer gpa.free(args);
    if (args.len != 3 and args.len != 4) {
        std.debug.print("usage: {s} <corpus> [merges] <out.json>\n", .{args[0]});
        return error.BadArguments;
    }
    const corpus_path = args[1];
    // The merge count is optional and the output path is not, so a three
    // argument invocation is the default and a four argument one is the opt in.
    const n_merges: usize = if (args.len == 4) std.fmt.parseInt(usize, args[2], 10) catch {
        std.debug.print("merge count must be a non-negative integer, got '{s}'\n", .{args[2]});
        return error.BadMergeCount;
    } else default_merges;
    const out_path = if (args.len == 4) args[3] else args[2];
    // A sub path is resolved against the working directory even when it starts
    // with a slash, so an absolute output path lands somewhere other than where
    // it reads. Refusing it is the rule this directory already states: the tool
    // writes inside the repository.
    if (std.fs.path.isAbsolute(out_path)) {
        std.debug.print("output path must be inside the repository, got '{s}'\n", .{out_path});
        return error.AbsoluteOutputPath;
    }

    // No ceiling: the corpus is an operator-chosen local file, and refusing to
    // read the one they named is a worse failure than reading a large one.
    const corpus = try std.Io.Dir.cwd().readFileAlloc(io, corpus_path, scratch, .unlimited);
    var tk = try Tokenizer.init(gpa);
    defer tk.deinit();
    try tk.train(corpus, n_merges);

    const ids = try tk.encode(scratch, corpus);
    // outputs/ is the documented home of generated artifacts, and a tool that
    // only works on a checkout that already has the directory is a tool that
    // fails on a fresh clone.
    const dir = std.Io.Dir.cwd();
    if (std.fs.path.dirname(out_path)) |parent| try dir.createDirPath(io, parent);
    try tk.save(dir, io, out_path);
    try verifyRoundTrip(gpa, dir, io, out_path, corpus);

    var longest: usize = 0;
    for (tk.vocab.items) |token| longest = @max(longest, token.len);
    // stderr, so a caller can redirect the report and keep only the artifact.
    std.debug.print("corpus {d} bytes\n", .{corpus.len});
    std.debug.print("merges learned {d} of {d} requested, vocabulary {d} tokens, longest {d} bytes\n", .{
        tk.merges.items.len,
        n_merges,
        tk.vocab.items.len,
        longest,
    });
    std.debug.print("{d} tokens, {d:.2} bytes per token\n", .{
        ids.len,
        @as(f64, @floatFromInt(corpus.len)) / @as(f64, @floatFromInt(ids.len)),
    });
    std.debug.print("wrote {s}\n", .{out_path});
}

/// Re-reads the file just written and puts a slice of the corpus back together
/// from it. Training succeeding says nothing about the artifact: the merge list
/// is the file, and a file that loads into a vocabulary which cannot rebuild the
/// text it was trained on is not a vocabulary. 4 KiB is enough to cross token
/// boundaries and costs nothing next to the training that precedes it.
fn verifyRoundTrip(
    gpa: std.mem.Allocator,
    dir: std.Io.Dir,
    io: std.Io,
    out_path: []const u8,
    corpus: []const u8,
) !void {
    var loaded = try Tokenizer.load(gpa, dir, io, out_path);
    defer loaded.deinit();

    const sample = corpus[0..@min(corpus.len, 4096)];
    const ids = try loaded.encode(gpa, sample);
    defer gpa.free(ids);
    const back = try loaded.decode(gpa, ids);
    defer gpa.free(back);
    if (!std.mem.eql(u8, sample, back)) return error.RoundTripMismatch;
    std.debug.print("round trip ok: {d} bytes rebuilt from {d} tokens\n", .{ back.len, ids.len });
}
