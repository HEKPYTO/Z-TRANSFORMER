//! Trains a byte-level BPE vocabulary from a corpus and writes it as JSON.
//!
//!     zig build-exe --dep tokenizer -Mroot=tools/train_bpe.zig \
//!         -Mtokenizer=src/tokenizer.zig -femit-bin=train_bpe
//!     ./train_bpe data/tinyshakespeare.txt 200 outputs/vocab.json
//!
//! The `-M` form is how this tool reaches src/tokenizer.zig. A module root
//! cannot import a file outside its own directory, so a named module is the
//! only way to share the implementation instead of copying it into tools/.

const std = @import("std");
const Tokenizer = @import("tokenizer").Tokenizer;

pub fn main(init: std.process.Init) !void {
    // The arena owns everything for the length of the run, including the corpus
    // and the vocabulary, so there is no per-allocation teardown to get wrong.
    const gpa = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(gpa);
    if (args.len != 4) {
        std.debug.print("usage: {s} <corpus> <merges> <out.json>\n", .{args[0]});
        return error.BadArguments;
    }
    const corpus_path = args[1];
    const n_merges = std.fmt.parseInt(usize, args[2], 10) catch {
        std.debug.print("merge count must be a non-negative integer, got '{s}'\n", .{args[2]});
        return error.BadMergeCount;
    };
    const out_path = args[3];

    // No ceiling: the corpus is an operator-chosen local file, and refusing to
    // read the one they named is a worse failure than reading a large one.
    const corpus = try std.Io.Dir.cwd().readFileAlloc(io, corpus_path, gpa, .unlimited);
    var tk = try Tokenizer.init(gpa);
    try tk.train(corpus, n_merges);

    const ids = try tk.encode(gpa, corpus);
    try tk.save(std.Io.Dir.cwd(), io, out_path);

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
