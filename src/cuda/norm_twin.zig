//! The CPU half of the CUDA RMSNorm harness: it generates the inputs, it runs
//! the real `norm.forward` over them, and it writes both the reference output
//! and its own timing. `src/cuda/norm.cu` then runs the GPU kernel over the same
//! input bytes and the two are compared.
//!
//! The reference is this file calling `norm.zig`, not a C copy of it: a second
//! implementation we wrote is not a reference and agreeing with it proves less
//! than it appears to, so the .cu deliberately has
//! no CPU implementation of RMSNorm at all: it can only fail by disagreeing with
//! the same code the CPU tests pin.
//!
//! It is not a build target and `build.zig` does not mention it. It runs the way
//! `tools/train_bpe.zig` does, as a root module over `src/` files brought in as
//! a second module, for the reason `tools/README.md` gives: a module root in
//! Zig 0.16 may not import a file outside its own directory, so a plain
//! `zig run src/cuda/norm_twin.zig` cannot reach `../norm.zig` at all.
//!
//!     zig build-exe -OReleaseFast -OReleaseFast --dep ztransformer \
//!       -Mroot=src/cuda/norm_twin.zig -Mztransformer=src/norm.zig \
//!       -femit-bin=<scratch>/norm_twin
//!     <scratch>/norm_twin <scratch>
//!
//! Output, all under the scratch directory, all little-endian f32 with no header,
//! because both readers are ours and a header would be one more thing to keep in
//! sync:
//!
//!   <tag>.x.bin  the [rows, cols] input
//!   <tag>.w.bin  the [1, cols] weight
//!   <tag>.y.bin  norm.forward's answer, the reference the kernel is graded against
//!   manifest.tsv one line per shape, described below
//!
//! The manifest exists so the shape table is written once. `norm.cu` reads it and
//! hardcodes no shape of its own, so adding a row to `shapes` below is the only
//! edit a new size needs and the two halves cannot disagree about what was run:
//!
//!   tag  rows  cols  kind  iters  cpu_us  alloc_us
//!
//! `iters` is 0 for a shape that is only checked for parity and non-zero for one
//! that is also timed. `cpu_us` is `norm.forward`'s own per-call time in
//! microseconds and `alloc_us` is what one `Tensor.init`/`deinit` pair at the
//! same shape costs, measured on its own. The subtraction is what the CPU
//! actually spends on the arithmetic, and a benchmark that charged the GPU a
//! resident output buffer while charging the CPU for one it had to ask for would
//! be measuring the allocator. `norm.cu` prints both.
//!
//! `cpu_us` is -1 and `alloc_us` is -1 for a shape that was not timed. Times
//! come from ReleaseFast because Debug leaves the loop unoptimised and a Debug
//! number would be a measurement of the build mode.

const std = @import("std");
const Io = std.Io;
const norm = @import("ztransformer");

// The Tensor type this module's `forward` takes, taken off the signature rather
// than imported.
//
// `src/norm.zig` is the module root, so `forward` is reachable from here and the
// `Tensor` it takes is not: a module exposes its root's public declarations, and
// the root's own imports are not part of that interface. Declaring
// `src/tensor.zig` as a second module instead fails the build outright, because
// norm.zig imports that same file and a file may belong to only one module. So
// the type is read off `forward`'s second parameter, which is the identical type
// from the identical module rather than a second copy that could drift from it.
// Parameter 0 is the allocator; parameter 1 is `x`.
const Tensor = @typeInfo(@TypeOf(norm.forward)).@"fn".params[1].type.?;

/// How a row of the input is filled. Both kinds are checked against the GPU and
/// they fail in opposite directions.
const Kind = enum {
    /// Standard normal per element. Roughly the scale of a residual stream, and
    /// sum_sq/cols lands near 1, so the normalisation is well conditioned.
    gauss,

    /// Every element the same value. This is the worst case for summing squares
    /// and the only kind that stresses the accumulator, because the partial sums
    /// are all positive and identical rather than cancelling: the rounding error
    /// grows with the row length instead of with its square root. It is the row
    /// `norm_test.zig` pins at 512, and it is here at 512 and at 4096, because
    /// "the reduction is more accurate than a serial one" is a claim that has to
    /// be measured against the case that punishes it hardest.
    constant,

    fn name(self: Kind) []const u8 {
        return @tagName(self);
    }
};

const Shape = struct {
    tag: []const u8,
    rows: usize,
    cols: usize,
    kind: Kind,
    /// Repetitions for the CPU timing, 0 for parity only. Chosen per row so each
    /// lands between roughly a tenth of a second and a second: fewer samples and
    /// a clock reading is a large fraction of one, more and the run drags.
    iters: usize,
};

/// The shapes, and the whole of what this harness covers.
///
/// `cols` of 128 is `d_model` at a full context window of 256, which is the
/// shipped model shape the benchmark is expected to lose at. 512 is `ffn_dim`,
/// the widest row this model actually builds. 4096 is neither, and is here to
/// find the crossover and to give the f32 accumulator its longest row.
const shapes = [_]Shape{
    // Parity only. 1x1 is the degenerate row, 1x4 is the hand computed one from
    // norm_test.zig, and 3x7 is a row length that is neither a multiple of the
    // block nor of the warp, which is where a strided loop goes wrong.
    .{ .tag = "one", .rows = 1, .cols = 1, .kind = .gauss, .iters = 0 },
    .{ .tag = "hand", .rows = 1, .cols = 4, .kind = .gauss, .iters = 0 },
    .{ .tag = "ragged", .rows = 3, .cols = 7, .kind = .gauss, .iters = 0 },
    // The shipped model shape and the widest row it builds, both kinds.
    .{ .tag = "model", .rows = 256, .cols = 128, .kind = .gauss, .iters = 0 },
    .{ .tag = "model_flat", .rows = 256, .cols = 128, .kind = .constant, .iters = 0 },
    .{ .tag = "ffn", .rows = 64, .cols = 512, .kind = .gauss, .iters = 0 },
    .{ .tag = "ffn_flat", .rows = 64, .cols = 512, .kind = .constant, .iters = 0 },
    // The long row: the parity case the f32 accumulator argument rests on, in
    // both kinds. At 4096 the constant row is where a serial sum is visibly wrong.
    .{ .tag = "wide", .rows = 64, .cols = 4096, .kind = .gauss, .iters = 0 },
    .{ .tag = "wide_flat", .rows = 64, .cols = 4096, .kind = .constant, .iters = 0 },

    // Timed, in rising element count, which is the order `norm.cu` walks to find
    // the crossover. The three small ones are there for that: the kernel has a
    // fixed cost of a few microseconds regardless of size, so the shape at which
    // it stops being worth running is below anything the model itself builds and
    // has to be measured rather than guessed.
    //
    // `ship` is the shipped model shape. `tall` through `r1024c2048` move one
    // axis at a time, and `big` is the large shape.
    .{ .tag = "micro", .rows = 32, .cols = 32, .kind = .gauss, .iters = 20000 },
    .{ .tag = "small", .rows = 32, .cols = 128, .kind = .gauss, .iters = 20000 },
    .{ .tag = "mid", .rows = 128, .cols = 128, .kind = .gauss, .iters = 10000 },
    .{ .tag = "ship", .rows = 256, .cols = 128, .kind = .gauss, .iters = 4000 },
    .{ .tag = "tall", .rows = 4096, .cols = 128, .kind = .gauss, .iters = 400 },
    .{ .tag = "r1024c512", .rows = 1024, .cols = 512, .kind = .gauss, .iters = 400 },
    .{ .tag = "r256c2048", .rows = 256, .cols = 2048, .kind = .gauss, .iters = 150 },
    .{ .tag = "r1024c2048", .rows = 1024, .cols = 2048, .kind = .gauss, .iters = 60 },
    .{ .tag = "big", .rows = 4096, .cols = 4096, .kind = .gauss, .iters = 12 },
};

/// What the `constant` rows are filled with: 1/3, written as a f32 division so it
/// is bit for bit the f32 `norm_test.zig` builds its 512-element row from. A
/// value with a binary expansion does not hide behind a rounded weight.
const flat_value: f32 = @as(f32, 1.0) / @as(f32, 3.0);

/// One PRNG seed for the whole table, so a run is reproducible from this file
/// alone and the .cu side needs no seed of its own.
const seed: u64 = 20260929;

// The blob format is little-endian f32 written straight out of memory, so this
// has to be asserted rather than assumed: on a big-endian target the files would
// carry in-memory order and the comparison would measure nothing.
comptime {
    if (@import("builtin").cpu.arch.endian() != .little) {
        @compileError("the harness blob format is little-endian f32, so the in-memory layout cannot be written directly");
    }
}

pub fn main(init: std.process.Init) !void {
    const scratch = init.arena.allocator();
    const args = try init.minimal.args.toSlice(scratch);

    if (args.len != 2) {
        std.debug.print("usage: {s} <scratch-dir>\n", .{args[0]});
        return error.Usage;
    }
    const dir = args[1];

    // The tensors go on the real allocator, not the arena. `norm.forward`
    // allocates its own output and the benchmark frees it between calls, so an
    // arena would grow by one output per iteration and the large shape would
    // measure the allocator rather than the op.
    const gpa = init.gpa;
    try Io.Dir.cwd().createDirPath(init.io, dir);

    var manifest: std.ArrayList(u8) = .empty;
    for (shapes) |shape| {
        var x = try Tensor.init(gpa, shape.rows, shape.cols);
        defer x.deinit();
        var w = try Tensor.init(gpa, 1, shape.cols);
        defer w.deinit();
        fillInput(&x, shape.kind);
        // A DIFFERENT seed from `fillInput`'s, and that is the whole point of the
        // parameter. `fillInput` draws from `seed` and consumes two randoms per
        // element; passing `seed` here too made `w.data[i]` bit-identical to
        // `x.data[i]` for every `i < cols` -- verified, not assumed -- so a kernel
        // reading `x_row[i]` where it meant `weight[i]` agreed with the CPU twin on
        // EVERY shape's first row, and on the whole of the 1x1 and 1x4 shapes. The
        // doc comment above `fillWeight` says a constant weight "would hide a kernel
        // that indexed the wrong element"; the equal-seed case hid it just as well.
        // Deriving the weight seed keeps the two streams provably distinct without a
        // second hand-maintained constant to fall out of step with this one.
        fillWeight(&w, seed ^ 0x9E3779B97F4A7C15, shape.cols);

        // `var` rather than `const` because `deinit` takes a mutable receiver,
        // and a const local would make defer a cast that discards const.
        var y = try norm.forward(gpa, x, w);
        defer y.deinit();

        try writeFloats(init.io, try std.fmt.allocPrint(scratch, "{s}/{s}.x.bin", .{ dir, shape.tag }), x.data);
        try writeFloats(init.io, try std.fmt.allocPrint(scratch, "{s}/{s}.w.bin", .{ dir, shape.tag }), w.data);
        try writeFloats(init.io, try std.fmt.allocPrint(scratch, "{s}/{s}.y.bin", .{ dir, shape.tag }), y.data);

        var cpu_us: f64 = -1.0;
        var alloc_us: f64 = -1.0;
        if (shape.iters != 0) {
            cpu_us = try timeCpu(init, gpa, x, w, shape.iters);
            alloc_us = try timeAlloc(init, gpa, shape.rows, shape.cols, shape.iters);
        }
        try manifest.appendSlice(scratch, try std.fmt.allocPrint(scratch, "{s} {d} {d} {s} {d} {d:.6} {d:.6}\n", .{
            shape.tag, shape.rows, shape.cols, shape.kind.name(), shape.iters, cpu_us, alloc_us,
        }));
        std.debug.print("twin: {s: <10} {d: >5}x{d:<5} {s: <7} iters {d: <5} cpu_us {d:>10.3} alloc_us {d:>8.3}\n", .{
            shape.tag, shape.rows, shape.cols, shape.kind.name(), shape.iters, cpu_us, alloc_us,
        });
    }
    try writeText(init.io, try std.fmt.allocPrint(scratch, "{s}/manifest.tsv", .{dir}), manifest.items);

    // Peak resident set, not the working set. The question the benchmark has to
    // answer is what the CPU path costs in memory, and an allocator is free to
    // hand back a page it kept, so the high water mark is the only number that
    // answers it. Read after every shape, so it covers the widest one.
    if (try peakRssBytes(init)) |bytes| {
        std.debug.print("twin: peak RSS {d:.1} MiB over {d} shapes\n", .{
            @as(f64, @floatFromInt(bytes)) / (1024.0 * 1024.0), shapes.len,
        });
    }
    std.debug.print("twin: OK\n", .{});
}

/// Fills `x` in place. One stream for the whole table rather than one per shape,
/// so the input of shape N does not depend on how many elements came before it.
fn fillInput(x: *Tensor, kind: Kind) void {
    var prng = std.Random.Xoshiro256.init(seed);
    for (x.data) |*v| {
        switch (kind) {
            .gauss => v.* = @floatCast(standardNormal(prng.random())),
            .constant => v.* = flat_value,
        }
    }
}

/// The weight is a standard normal even for a constant input. A weight of all
/// ones would leave the multiply untested, and a constant weight would hide a
/// kernel that indexed the wrong element.
fn fillWeight(w: *Tensor, s: u64, cols: usize) void {
    var prng = std.Random.Xoshiro256.init(s);
    for (0..cols) |i| w.data[i] = @floatCast(standardNormal(prng.random()));
}

/// Box-Muller. `1.0 - float()` rather than `float()` alone, because the argument
/// of a log has to be strictly positive and `float()` can return exactly zero.
fn standardNormal(rnd: std.Random) f64 {
    const open = 1.0 - rnd.float(f64);
    const angle = rnd.float(f64);
    return @sqrt(-2.0 * @log(open)) * @cos(2.0 * std.math.pi * angle);
}

/// Per-call microseconds for `norm.forward`, allocation included.
///
/// The allocation is inside the timed region because it is inside the function,
/// and measuring it is more honest than removing it: it is what the CPU path
/// actually pays. It is under a percent of the total at both ends of the table,
/// against tens of microseconds for the shipped shape and tens of milliseconds
/// for the large one, so no correction is applied and none is needed.
///
/// One warm-up call first, untimed, so the first-touch page faults on the input
/// and the allocator's first excursion into a large size class are not charged to
/// the sample. The clock is read once around the whole loop rather than once per
/// call, which keeps the clock's own cost out of a measurement this short.
fn timeCpu(init: std.process.Init, gpa: std.mem.Allocator, x: Tensor, w: Tensor, iters: usize) !f64 {
    var warm = try norm.forward(gpa, x, w);
    warm.deinit();

    const start = Io.Clock.awake.now(init.io);
    for (0..iters) |_| {
        var out = try norm.forward(gpa, x, w);
        out.deinit();
    }
    const ns = start.untilNow(init.io, .awake).toNanoseconds();
    return @as(f64, @floatFromInt(ns)) / (@as(f64, @floatFromInt(iters)) * @as(f64, std.time.ns_per_us));
}

/// Per-call microseconds for one `Tensor.init` plus `deinit` at the same shape,
/// measured exactly the way `timeCpu` measures `forward`.
///
/// This is subtracted from `cpu_us` and nothing else. `norm.forward` allocates
/// its output, so a bare CPU number carries one allocator round trip, while the
/// GPU path reuses buffers that already exist. Reporting the difference is what
/// keeps the comparison about the arithmetic rather than about the two hosts'
/// allocators, and both numbers are printed either way.
fn timeAlloc(init: std.process.Init, gpa: std.mem.Allocator, rows: usize, cols: usize, iters: usize) !f64 {
    var warm = try Tensor.init(gpa, rows, cols);
    warm.deinit();

    const start = Io.Clock.awake.now(init.io);
    for (0..iters) |_| {
        var t = try Tensor.init(gpa, rows, cols);
        t.deinit();
    }
    const ns = start.untilNow(init.io, .awake).toNanoseconds();
    return @as(f64, @floatFromInt(ns)) / (@as(f64, @floatFromInt(iters)) * @as(f64, std.time.ns_per_us));
}

/// `VmHWM` from /proc, in bytes, or null where there is no procfs. Null rather
/// than a guess: a benchmark that printed a fabricated memory number would be
/// worse than one that printed nothing.
fn peakRssBytes(init: std.process.Init) !?usize {
    // The root directory is opened and "/proc/self/status" is taken relative to
    // it, because `Io.Dir.cwd()` refuses an absolute path. Linux only, and null
    // everywhere else, which means "not measured" rather than a number.
    const root = Io.Dir.openDirAbsolute(init.io, "/", .{}) catch return null;
    defer root.close(init.io);
    var buffer: [8192]u8 = undefined;
    const status = root.readFile(init.io, "proc/self/status", &buffer) catch return null;
    var lines = std.mem.splitScalar(u8, status, '\n');
    while (lines.next()) |line| {
        if (!std.mem.startsWith(u8, line, "VmHWM:")) continue;
        const digits = std.mem.trim(u8, line["VmHWM:".len..], " \t");
        // The field is a count and a unit, "788 kB". Only the count is parsed,
        // because handing the unit to parseInt fails and this function returns
        // null on failure, which reads as "not measured" and not as an error.
        const end = std.mem.indexOfAny(u8, digits, " \t") orelse digits.len;
        const kib = std.fmt.parseInt(usize, digits[0..end], 10) catch return null;
        return kib * 1024;
    }
    return null;
}

fn writeFloats(io: Io, path: []const u8, data: []const f32) !void {
    const file = try Io.Dir.cwd().createFile(io, path, .{});
    defer file.close(io);
    var buffer: [4096]u8 = undefined;
    var out: Io.File.Writer = .init(file, io, &buffer);
    try out.interface.writeAll(std.mem.sliceAsBytes(data));
    try out.interface.flush();
}

fn writeText(io: Io, path: []const u8, bytes: []const u8) !void {
    const file = try Io.Dir.cwd().createFile(io, path, .{});
    defer file.close(io);
    var buffer: [4096]u8 = undefined;
    var out: Io.File.Writer = .init(file, io, &buffer);
    try out.interface.writeAll(bytes);
    try out.interface.flush();
}
