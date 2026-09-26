//! Seeded batching and the train/validation split over a token id stream.
//!
//! The module starts at token ids. It never sees text and never sees a
//! tokenizer, so the split and the batching are testable on their own.
//!
//! `Corpus` borrows the caller's array: both halves are subslices of the input,
//! so there is nothing for it to free and a `deinit` would be a no-op that lies
//! about ownership. `Batch` owns its two slices, because `Batcher` hands them
//! out one at a time and the caller has to return them.

const std = @import("std");

/// One training batch: `ctx` input ids and the `ctx` ids that follow them.
pub const Batch = struct {
    allocator: std.mem.Allocator,
    inputs: []u32,
    targets: []u32,

    pub fn deinit(self: *Batch) void {
        self.allocator.free(self.inputs);
        self.allocator.free(self.targets);
        self.* = undefined;
    }
};

/// The two halves of a token stream. Both borrow from the array passed to
/// `split`.
pub const Corpus = struct {
    train: []const u32,
    val: []const u32,
};

/// Cuts the last `val_fraction` of `tokens` off as validation.
///
/// The cut is positional and happens before any shuffle, as `data/README.md`
/// fixes it, so the same stream always splits the same way.
pub fn split(tokens: []const u32, val_fraction: f64) !Corpus {
    // Written as a negated range test so a NaN fraction is rejected too: NaN
    // fails every `>=` and `<=` it meets, and a ratio outside zero to one would
    // otherwise compute a cut outside the slice.
    if (!(val_fraction >= 0.0 and val_fraction <= 1.0)) return error.BadValFraction;
    // The only float in the module. The stream is an integer count and the
    // fraction is a configured ratio, so this one truncation is the whole
    // conversion, and it is a floor because the range check left it non-negative.
    const val_len: usize = @intFromFloat(@as(f64, @floatFromInt(tokens.len)) * val_fraction);
    const cut = tokens.len - val_len;
    return .{ .train = tokens[0..cut], .val = tokens[cut..] };
}

/// Walks a token stream as fixed-length windows, in a seeded order.
///
/// Windows are `ctx + 1` consecutive tokens, and they are disjoint: window `k`
/// starts at `k * (ctx + 1)`. A batch is the first `ctx` of those tokens with
/// the last `ctx` as targets. A trailing partial window is dropped rather than
/// padded, because padding would feed the trainer a token the corpus never
/// held.
///
/// The shuffle permutes the order of the windows. The tokens inside a window
/// keep their stream order, because a permutation applied inside a batch
/// destroys the sequence and trains the model on noise.
pub const Batcher = struct {
    allocator: std.mem.Allocator,
    tokens: []const u32,
    ctx: usize,
    /// `order[i]` is the window the i-th batch is drawn from.
    order: []usize,
    cursor: usize,
    /// Seeded once in `init` and then advanced by every `reset`, so the run's
    /// batch order is a function of the seed and the epoch and of nothing else.
    ///
    /// Xoshiro256++ named rather than reached through `std.Random.DefaultPrng`.
    /// That alias is documented as an implementation choice, so a Zig upgrade
    /// could reshuffle a training run's batches without a line here changing,
    /// and the batch order is what the loss curve and every benchmark built on
    /// it are measured from. Naming the engine makes the stream part of the
    /// recorded artifact.
    prng: std.Random.Xoshiro256,

    pub fn init(allocator: std.mem.Allocator, tokens: []const u32, ctx: usize, seed: u64) !Batcher {
        if (ctx == 0) return error.BadContext;
        const stride = ctx + 1;
        const count = if (tokens.len < stride) 0 else tokens.len / stride;
        const order = try allocator.alloc(usize, count);
        errdefer allocator.free(order);

        var self: Batcher = .{
            .allocator = allocator,
            .tokens = tokens,
            .ctx = ctx,
            .order = order,
            .cursor = 0,
            .prng = std.Random.Xoshiro256.init(seed),
        };
        self.reset();
        return self;
    }

    pub fn deinit(self: *Batcher) void {
        self.allocator.free(self.order);
        self.* = undefined;
    }

    /// The next batch, or null once the stream is exhausted.
    pub fn next(self: *Batcher) !?Batch {
        if (self.cursor == self.order.len) return null;
        const window = self.order[self.cursor];
        const start = window * (self.ctx + 1);

        const inputs = try self.allocator.alloc(u32, self.ctx);
        errdefer self.allocator.free(inputs);
        const targets = try self.allocator.alloc(u32, self.ctx);
        errdefer self.allocator.free(targets);

        // The target side reads one token further than the input side, so
        // together they cover start..start+ctx+1, which is the whole window.
        @memcpy(inputs, self.tokens[start .. start + self.ctx]);
        @memcpy(targets, self.tokens[start + 1 .. start + self.ctx + 1]);

        // Advanced only once both allocations have succeeded, so an allocation
        // failure leaves the walk where it was instead of dropping a batch.
        self.cursor += 1;
        return .{ .allocator = self.allocator, .inputs = inputs, .targets = targets };
    }

    /// Restarts the walk in a fresh permutation.
    ///
    /// The shuffle is drawn from the batcher's live PRNG, so every epoch gets a
    /// different order and two runs of one seed still agree epoch for epoch. It
    /// is not re-seeded here: a fresh PRNG per reset would replay the first
    /// epoch's order for every epoch, which is a shuffle that stops shuffling.
    pub fn reset(self: *Batcher) void {
        for (self.order, 0..) |*slot, i| slot.* = i;
        // The order is an index array walked front to back, so it never depends
        // on a hash table's iteration order.
        self.prng.random().shuffle(usize, self.order);
        self.cursor = 0;
    }
};
