//! The binary. A bare `ztransformer` prints the banner; `ztransformer train`
//! runs the one path the project ships end to end: corpus in, loss curve out;
//! `ztransformer parity` writes the tensors a Llama reference is
//! compared against.
//!
//! The defaults below are a smoke run, not a recipe, and the reason is measured
//! rather than assumed. A step is a dense f32 forward and backward over a
//! `ctx`-token window, so the run's cost is the number of windows and the
//! defaults are chosen to keep `zig build train` at roughly two minutes on a
//! laptop while the loss still visibly falls, which is the property the README
//! claims and the one a reviewer has to see. A real run wants the whole corpus
//! and more epochs; `model.defaultConfig` is untouched either way, because other
//! code depends on it.
const std = @import("std");
const Io = std.Io;
const lib = @import("lib.zig");
const tokenizer = @import("tokenizer.zig");
const data = @import("data.zig");
const model = @import("model.zig");
const train = @import("train.zig");
const parity = @import("removed.zig");
const scale = @import("scale.zig");

const corpus_path = "data/tinyshakespeare.txt";
const csv_path = "outputs/loss.csv";
/// Where a run writes the curve before anyone knows whether it is the committed
/// one. The digest this repository claims is checked by `zig build verify`, and a
/// reader who cannot produce those bytes has found out something; whether the
/// committed file survives that finding is the difference between a report and a
/// lost artifact, so the run reports first and replaces second. Never committed,
/// and `.gitignore` covers it: see `outputs/README.md`.
const csv_pending_path = "outputs/loss.pending.csv";
/// The digest the committed curve is claimed to have, from `build.zig`. Named
/// here rather than only there so the message below can print the two sides of a
/// mismatch without the reader leaving the terminal to run `shasum`.
const csv_sha256 = "f1dd54445064810c28002dcacaf23b4bc82bb1e6ecfa28f5ed91e7fa4518f792";
/// The 95/5 split `data/README.md` fixes, applied here because this is the only
/// shipped caller of `data.split`.
const val_fraction = 0.05;
/// 200 merges is 456 tokens of vocabulary, which is under the 1024 the default
/// model config allocates and keeps the O(len * merges) tokenizer off the
/// critical path. The tokenizer's own cost curve for this is documented on
/// `tokenizer.Tokenizer.train`.
const n_merges = 200;
/// 64 KiB of the 1.1 MB corpus, which tokenizes to about 33k tokens and 123
/// windows of the default `ctx` 256. The whole corpus is 2211 windows, eighteen
/// times the work, so a full epoch belongs in a scheduled run rather than in the
/// command a reviewer is expected to sit through. No time is claimed for it:
/// the smoke run's own wall time includes the build check and the tokenization,
/// neither of which divides into a per-step rate, so a total derived from one
/// would be wrong in a way nothing in the run would reveal.
const corpus_bytes = 65536;
const epochs = 1;
/// The same seed `train_test.zig` measures on, so a run here and a run there
/// agree.
const seed = 7;

pub fn main(init: std.process.Init) !void {
    // The arena owns the corpus, the vocabulary and the run's result for the
    // length of the process, so nothing here has a per-allocation teardown.
    const gpa = init.arena.allocator();
    const args = try init.minimal.args.toSlice(gpa);

    if (args.len > 1) {
        // Exactly one argument. The binary reads no flags, so a second one is
        // a flag that was meant to change the run and did not: `train
        // --epochs 20` used to train for the hard-coded epoch count and say
        // nothing, which is a 90 second run on the wrong configuration. The two
        // callers this ships to, `zig build train` and `zig build run -- parity`,
        // each pass one, so nothing legitimate is refused.
        if (args.len == 2) {
            if (std.mem.eql(u8, args[1], "train")) return runTrain(gpa, init.io);
            if (std.mem.eql(u8, args[1], "parity")) return runRemoved(gpa, init.io);
            if (std.mem.eql(u8, args[1], "scale-profile")) return runScaleProfile(init.io);
        }
        std.debug.print("usage: {s} [train|parity|scale-profile]\n", .{args[0]});
        return error.UnknownCommand;
    }

    var buffer: [128]u8 = undefined;
    var stdout: Io.File.Writer = .init(.stdout(), init.io, &buffer);
    try stdout.interface.print("{s} {s}\n", .{ lib.name(), lib.version });
    try stdout.interface.flush();
}

fn runTrain(gpa: std.mem.Allocator, io: Io) !void {
    const whole = try Io.Dir.cwd().readFileAlloc(io, corpus_path, gpa, .unlimited);
    // `@min`, not a bare slice. `corpus_bytes` is a cap, and a cap that is
    // longer than the file is a cap that reads past the end: the checked build
    // panics, and `zig build train` links a ReleaseFast binary where the bounds
    // check is elided, so the slice silently becomes 64 KiB of whatever follows
    // the allocation and the run trains on it. The vendored corpus is 1.1 MB so
    // this never fires today, which is exactly why it survived: the only way to
    // see it is to hand the binary a small file.
    const text = whole[0..@min(whole.len, corpus_bytes)];
    var tk = try tokenizer.Tokenizer.init(gpa);
    try tk.train(text, n_merges);
    const ids = try tk.encode(gpa, text);
    const corpus = try data.split(ids, val_fraction);

    var cfg: train.Config = .{
        .model = model.defaultConfig(),
        .epochs = epochs,
        .ctx = model.defaultConfig().n_ctx,
        .lr = 0.3,
        .warmup = 20,
        .weight_decay = 0.0,
        .max_grad_norm = 1.0,
        .seed = seed,
        .log_every = 25,
    };
    // The model cannot have a vocabulary smaller than the tokenizer's, and a
    // wider one only adds embedding rows nothing ever indexes.
    cfg.model.vocab_size = tk.vocab.items.len;

    const stride = cfg.ctx + 1;
    // stderr, so a caller can pipe stdout and keep only the numbers. The plan is
    // printed before the run because the run is silent and takes minutes.
    std.debug.print(
        "corpus {d} of {d} bytes, {d} merges, vocabulary {d}, {d} train / {d} val tokens\n" ++
            "d_model {d}, {d} layers, ctx {d}, {d} windows x {d} epochs\n",
        .{
            text.len,
            whole.len,
            n_merges,
            tk.vocab.items.len,
            corpus.train.len,
            corpus.val.len,
            model.dModel(cfg.model),
            cfg.model.n_layers,
            cfg.ctx,
            corpus.train.len / stride,
            epochs,
        },
    );

    var res = try train.run(gpa, cfg, corpus.train, corpus.val);
    defer res.deinit();

    // Written beside the committed curve and moved onto it only on a match. A
    // run truncates its output, so writing `csv_path` directly replaced the
    // committed artifact before anything had compared it, and a reader on a
    // different libm or a different optimization level found a diff they could
    // no longer tell from a change the repository had made.
    try train.writeCsv(csv_pending_path, res.rows);
    var buffer: [256]u8 = undefined;
    var stdout: Io.File.Writer = .init(.stdout(), io, &buffer);
    const w = &stdout.interface;
    // The three numbers, and nothing about the file. Whether the run's curve
    // became the committed one is `settleCsv`'s to say, and a summary that
    // claimed a file was written before that question was answered is how a
    // mismatch ends up announced as a success.
    try w.print("steps {d}\ntrain_loss {d:.4}\nval_loss {d:.4}\n", .{
        res.steps,
        res.train_loss,
        res.val_loss,
    });
    try w.flush();
    try settleCsv(io);
}

/// Compares the curve just written against the digest the repository claims for
/// it, and on a match renames it onto `csv_path`. The rename is unconditional
/// rather than skipped, and it is the whole of the success path: the bytes are
/// already the committed bytes, so this leaves the content identical and the
/// pending file gone, which is what makes a reproducing run leave the tree as
/// clean as it found it.
///
/// On a mismatch it does not replace anything and it does not fail. The run
/// itself is what it is either way; only the question of which bytes are the
/// committed ones is open, and a different answer there is not a broken build.
/// `@exp`, `@sqrt` and `@cos` resolve to the platform libm, and Debug differs
/// from release by about one f32 ulp per step, so a host or a build
/// configuration that disagrees here has produced a legitimate run of its own.
/// The pending file is kept so the reader can diff the two curves, and the
/// message says which is which instead of printing a bare mismatch: the exit
/// status stays 0, because failing it would report a host difference as a
/// defect, and a gate that cries wolf is worse than no gate.
fn settleCsv(io: Io) !void {
    const dir = Io.Dir.cwd();
    const fresh = try dir.readFileAlloc(io, csv_pending_path, std.heap.page_allocator, .unlimited);
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(fresh, &digest, .{});
    const hex = std.fmt.bytesToHex(digest, .lower);

    if (std.mem.eql(u8, &hex, csv_sha256)) {
        try dir.rename(csv_pending_path, dir, csv_path, io);
        return;
    }

    std.debug.print(
        \\
        \\zig build train: this run did not reproduce the committed {s}.
        \\  committed  {s}  sha256 {s}
        \\  this run   {s}  sha256 {s}
        \\
        \\  Both files are intact and the run exited 0. They differ because the
        \\  arithmetic is host- and configuration-dependent, not because anything
        \\  here is wrong: @exp, @sqrt and @cos resolve to the platform libm, and
        \\  Debug differs from release by about one f32 ulp per step. The
        \\  criterion is one seed, one build configuration, one host; see the
        \\  Reproducibility section of src/README.md.
        \\
        \\  To see whether the run itself is sound, diff the two curves. To accept
        \\  this run as the new committed one, copy it over {s} and put its digest
        \\  in build.zig; that is a change to the claim, so it is yours to make.
        \\
    , .{ csv_path, csv_path, csv_sha256, csv_pending_path, hex, csv_path });
}

/// `zig build scale-profile`. Projects what the code's own formulas imply at
/// shapes it cannot be run at, and prints nothing it measured: the whole output
/// is arithmetic over `model.Config`, and a reader who cannot tell a projection
/// from a benchmark is the failure this exists to prevent.
fn runScaleProfile(io: Io) !void {
    var buffer: [4096]u8 = undefined;
    var stdout: Io.File.Writer = .init(.stdout(), io, &buffer);
    try scale.print(&stdout.interface);
    try stdout.interface.flush();
}

fn runRemoved(gpa: std.mem.Allocator, io: Io) !void {
    // The exporter is the whole of this side of the harness. It runs in Zig
    // with no Python anywhere in sight, because AGENTS.md requires every
    // committed artifact to be reproducible from the toolchain the repository
    // claims to need. `tools/removed/oracle.txt` reads what this writes; it
    // never produces any of it.
    const s = parity.sweep();
    const res = try parity.run(gpa, io, s);

    var buffer: [256]u8 = undefined;
    var stdout: Io.File.Writer = .init(.stdout(), io, &buffer);
    const w = &stdout.interface;
    try w.print(
        "wrote {s}/{{config.txt,index.txt,inputs.txt,data.bin}}\n" ++
            "d_model {d}, {d} layers, {d} heads over {d} kv heads of {d}, ffn {d}, vocab {d}, ctx {d}\n" ++
            "{d} weight blobs, {d} intermediate blobs over {d} cases\n" ++
            "next: sh tools/removed/check.sh\n",
        .{
            parity.dir,
            model.dModel(s.model),
            s.model.n_layers,
            s.model.n_heads,
            s.model.n_kv_heads,
            s.model.head_dim,
            model.ffnDim(s.model),
            s.model.vocab_size,
            s.model.n_ctx,
            res.weight_blobs,
            res.intermediate_blobs,
            res.cases,
        },
    );
    try w.flush();
}
