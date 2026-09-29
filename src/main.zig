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
/// The digest the committed curve is claimed to have.
///
/// The one owner of this constant. `build.zig` reads it from here rather than
/// keeping a second copy: two copies of a digest that a reader is told to
/// update is a wedge with the tool's own recovery instructions on it, because
/// updating the one `verify` checks leaves `train` permanently unable to
/// promote. So when the claim changes, this is the line that changes, and the
/// `verify` digest check follows from it rather than needing a second edit.
pub const csv_sha256 = "f1dd54445064810c28002dcacaf23b4bc82bb1e6ecfa28f5ed91e7fa4518f792";
/// How a reader says "I have looked at the two curves and I know why they
/// differ" without promoting anything. A presence check, not a value check: any
/// value acknowledges, because the act is the reader's and the content of the
/// variable is not read by anything.
///
/// An environment variable rather than a flag because the binary reads no flags
/// and `zig build train` passes no arguments: adding one would mean a second
/// parsing path in a program whose whole argument contract is one word.
const accept_var = "ZTRANSFORMER_ACCEPT_LOSS_DIFFERENCE";
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
    // `init.arena` is permanent storage for the process and `init.gpa` is the
    // runtime's general purpose allocator, and the two are not interchangeable.
    // An arena's `free` is a no-op, so a caller that frees as it runs has to be
    // handed the gpa or it never gives the memory back. The corpus, the
    // vocabulary and the merges are read for the whole process and are the
    // arena's; a training step frees its cache, its logits and its gradient
    // buffers as it goes, so the run is the gpa's. It was the arena's once, and
    // the cost was that the peak was the sum of all 123 steps' working sets
    // rather than one step's.
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);

    if (args.len > 1) {
        // Exactly one argument. The binary reads no flags, so a second one is
        // a flag that was meant to change the run and did not: `train
        // --epochs 20` used to train for the hard-coded epoch count and say
        // nothing, which is a 90 second run on the wrong configuration. The two
        // callers this ships to, `zig build train` and `zig build run -- parity`,
        // each pass one, so nothing legitimate is refused.
        if (args.len == 2) {
            if (std.mem.eql(u8, args[1], "train")) return runTrain(arena, init.gpa, init.io, init.environ_map);
            if (std.mem.eql(u8, args[1], "parity")) return runRemoved(arena, init.io);
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

/// `arena` is `init.arena` and `gpa` is `init.gpa`, split for the reason
/// `main` gives. The run's `Result` is freed back into `gpa` by its own
/// `deinit` below, so the trained weights do not outlive the allocator that
/// produced them.
fn runTrain(
    arena: std.mem.Allocator,
    gpa: std.mem.Allocator,
    io: Io,
    environ: *std.process.Environ.Map,
) !void {
    const whole = try Io.Dir.cwd().readFileAlloc(io, corpus_path, arena, .unlimited);
    // `@min`, not a bare slice. `corpus_bytes` is a cap, and a cap that is
    // longer than the file is a cap that reads past the end: the checked build
    // panics, and `zig build train` links a ReleaseFast binary where the bounds
    // check is elided, so the slice silently becomes 64 KiB of whatever follows
    // the allocation and the run trains on it. The vendored corpus is 1.1 MB so
    // this never fires today, which is exactly why it survived: the only way to
    // see it is to hand the binary a small file.
    const text = whole[0..@min(whole.len, corpus_bytes)];
    var tk = try tokenizer.Tokenizer.init(arena);
    try tk.train(text, n_merges);
    const ids = try tk.encode(arena, text);
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
    try settleCsv(io, environ);
}

/// Compares the curve just written against the digest the repository claims for
/// it, and on a match renames it onto `csv_path`. The rename is unconditional
/// rather than skipped, and it is the whole of the success path: the bytes are
/// already the committed bytes, so this leaves the content identical and the
/// pending file gone, which is what makes a reproducing run leave the tree as
/// clean as it found it.
///
/// A digest mismatch exits non-zero, always, and promotion still requires the
/// bytes to be equal. The exit status no longer depends on a magnitude
/// threshold, and the reason is measured rather than preferred.
///
/// The threshold was built and then refuted by this repository's own two cases.
/// A budget of `steps * f32_epsilon * max_loss` — one f32 ulp per step, added,
/// which is what `src/README.md` documents — is 9.5e-5 on the committed curve.
/// `optim.AdamW.beta1` at 0.85 instead of 0.9 lands 1667x outside it, so that
/// case is caught. A Debug build of the identical source lands 189x outside it,
/// at step 74, and the run is fine. Worse, its disagreements are 39507, 6119,
/// -51706, 3219 and 9508 ulp at successive rows: not growing, not shrinking,
/// changing sign. That is trajectory chaos, where a one-ulp difference in an
/// early weight changes every weight after it, and it is unbounded in the step
/// count in a way no additive bound can express. A threshold between the two
/// cases would have to be 0.018 to 0.158 wide, which is a chosen constant
/// wearing a derivation's clothes, and it would move with the seed, the host's
/// libm, and the length of the run.
///
/// So the default is loud. A mismatch is a discrepancy the reader has not yet
/// explained, and this tool does not have the information to explain it: it can
/// name the size and the step, and it cannot say whether a size that big is a
/// broken build or a different machine. `ZTRANSFORMER_ACCEPT_LOSS_DIFFERENCE=1`
/// is how a reader who has looked says so, and until they do the run fails and
/// the committed curve is left alone. The variable does not promote anything: it
/// acknowledges a difference the reader has taken responsibility for, which is
/// a different act from changing the claim.
///
/// What survives from the threshold attempt is the measurement, because the
/// reader needs a number to judge the difference by and the number is the same
/// one either way: the largest disagreement, the step it is at, and the column.
/// Nothing here asserts a cause. A reader who has a reason can act on it.
///
/// The committed file is never replaced on a mismatch and the pending file is
/// never deleted, and the message says where the pending curve is and how to
/// diff it in both cases. Nothing here discards a curve a reader has not read.
fn settleCsv(io: Io, environ: *std.process.Environ.Map) !void {
    const dir = Io.Dir.cwd();
    const fresh = try dir.readFileAlloc(io, csv_pending_path, std.heap.page_allocator, .unlimited);
    const committed = try dir.readFileAlloc(io, csv_path, std.heap.page_allocator, .unlimited);
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(fresh, &digest, .{});
    const hex = std.fmt.bytesToHex(digest, .lower);

    if (std.mem.eql(u8, &hex, csv_sha256)) {
        try dir.rename(csv_pending_path, dir, csv_path, io);
        return;
    }

    // The measurement, when the two files are comparable at all. A curve of a
    // different shape has no cells to measure, and that is a louder failure than
    // a large disagreement, not a quieter one.
    const d = train.divergence(committed, fresh) catch |err| {
        std.debug.print(
            \\
            \\zig build train: this run's curve could not be compared with the committed one ({s}).
            \\
            \\  committed  {s}  sha256 {s}
            \\  this run   {s}  sha256 {s}
            \\
            \\  The two are not the same shape of file, so there is no cell-by-cell
            \\  comparison to report. The committed curve was not replaced and this run's
            \\  is at {s}, kept rather than deleted.
            \\
            \\  Compare them:  diff {s} {s}
            \\  To accept this run as the committed one, copy it over {s} and update
            \\  csv_sha256 in src/main.zig, which is where the digest is owned and
            \\  where build.zig reads it from. That is a change to the claim, so it is
            \\  yours to make.
            \\
        , .{
            @errorName(err),
            csv_path,
            csv_sha256,
            csv_pending_path,
            hex,
            csv_pending_path,
            csv_path,
            csv_pending_path,
            csv_path,
        });
        return err;
    };

    // Acknowledged, or not. `ZTRANSFORMER_ACCEPT_LOSS_DIFFERENCE=1` is a reader
    // saying they have looked at the two curves and know why they differ. It does
    // not promote: the committed file is still only replaced on a digest match,
    // because acknowledging a difference is not the same act as changing what
    // this repository claims, and conflating the two is what made the old message
    // assert a cause nobody had checked.
    if (environ.get(accept_var) != null) {
        std.debug.print(
            \\
            \\zig build train: this run did not reproduce the committed {s}, and you have
            \\  acknowledged the difference with {s}=1.
            \\
            \\  committed  {s}  sha256 {s}
            \\  this run   {s}  sha256 {s}
            \\
            \\  Largest disagreement  {d:.9} at step {d}, {s}: {d:.6} against {d:.6}
            \\  On a curve of {d} steps with a largest loss of {d:.6}
            \\
            \\  Exiting 0 because you said so. The committed curve was NOT replaced and
            \\  this run's is at {s}, kept. This message says the bytes differ and that
            \\  you accepted that; it does not claim to know why they differ.
            \\
            \\  Compare them:  diff {s} {s}
            \\
        , .{
            csv_path,           accept_var,       csv_path,   csv_sha256,
            csv_pending_path,   hex,              d.absolute, d.step,
            @tagName(d.column), d.committed,      d.fresh,    d.steps,
            d.max_loss,         csv_pending_path, csv_path,   csv_pending_path,
        });
        return;
    }

    std.debug.print(
        \\
        \\zig build train: this run did not reproduce the committed {s}. Failing, because
        \\  nothing has established that the difference is benign.
        \\
        \\  committed  {s}  sha256 {s}
        \\  this run   {s}  sha256 {s}
        \\
        \\  Largest disagreement  {d:.9} at step {d}, {s}: {d:.6} against {d:.6}
        \\  On a curve of {d} steps with a largest loss of {d:.6}
        \\
        \\  There are two reasons these bytes can differ, and this run cannot tell them
        \\  apart. Either something is wrong with the run — a gradient, a schedule, an
        \\  optimizer constant — or this is a different machine or a different build
        \\  configuration, where @exp, @sqrt and @cos resolve to a different libm and a
        \\  one-ulp difference in an early weight moves every weight after it. A wrong
        \\  optimizer constant lands about 0.16 out at step 24 here; a Debug build of the
        \\  identical source lands about 0.018 out at step 74. The gap between those is
        \\  8.8x, it moves with the seed, the libm and the run length, and no threshold
        \\  derived from it would be a fact about anything but this one host.
        \\
        \\  So: look at the curve before deciding.
        \\    compare  diff {s} {s}
        \\    broken   a large disagreement from step 1, growing, is a defect
        \\    host     a disagreement that wanders and changes sign is a different machine
        \\
        \\  If it is a host difference and you accept it, re-run with:
        \\    {s}=1 zig build train
        \\  which exits 0 and still does not replace the committed curve. Accepting a
        \\  difference is not the same as changing the claim; to do that, copy {s} over
        \\  {s} and update csv_sha256 in src/main.zig, which is where the digest is
        \\  owned and where build.zig reads it from.
        \\
        \\  This run's curve is at {s} and was not deleted.
        \\
    , .{
        csv_path,
        csv_path,
        csv_sha256,
        csv_pending_path,
        hex,
        d.absolute,
        d.step,
        @tagName(d.column),
        d.committed,
        d.fresh,
        d.steps,
        d.max_loss,
        csv_path,
        csv_pending_path,
        accept_var,
        csv_pending_path,
        csv_path,
        csv_pending_path,
    });
    return error.CurveDiffers;
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

/// `arena` is `init.arena`, and the name is the point: this function is handed
/// permanent storage and does not free as it goes. It was called `gpa` until the
/// training run's allocator was split, and a parameter whose name says one thing
/// while its value does another is how the run ended up on an arena in the first
/// place. The export is a single forward sweep over six cases, so it holds one
/// working set and the arena is the right allocator for it -- measured peak is
/// 34 MB, which is the sweep, not a sum over anything.
fn runRemoved(arena: std.mem.Allocator, io: Io) !void {
    // The exporter is the whole of this side of the harness. It runs in Zig
    // with no Python anywhere in sight, because AGENTS.md requires every
    // committed artifact to be reproducible from the toolchain the repository
    // claims to need. `tools/removed/oracle.txt` reads what this writes; it
    // never produces any of it.
    const s = parity.sweep();
    const res = try parity.run(arena, io, s);

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
