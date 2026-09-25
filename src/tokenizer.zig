//! Byte-level BPE: 256 base byte tokens plus learned merges, trained and serialized in Zig.

const std = @import("std");

/// Token ids below this are the byte values themselves. Every possible input
/// byte therefore has a token, which is what makes encoding total: there is no
/// unknown case, no replacement character, and nothing for a caller to handle
/// beyond the output slice.
pub const byte_vocab_size = 256;

/// A learned merge. The index of a merge in the list is its rank, and the token
/// it produced is `byte_vocab_size + rank`, so rank and id are one number and
/// cannot disagree.
pub const Merge = struct {
    left: u32,
    right: u32,
};

/// The file is the merge list and nothing else. The 256 byte tokens are implied
/// by the layout, and a merged token is the concatenation of the two tokens it
/// came from, so both are rebuilt on load and a saved vocabulary cannot carry a
/// byte table that disagrees with its merges.
const Persisted = struct {
    merges: []const Merge,
};

/// 16 MiB. Orders of magnitude above any vocabulary a real corpus produces,
/// low enough that a corrupt or hostile file fails fast instead of taking the
/// memory with it.
const max_vocab_bytes = 1 << 24;

/// A pair is only worth a merge once it occurs twice. A pair seen once cannot
/// shorten anything that shows up again, so keeping one-off sequences out of
/// the vocabulary is what stops a requested merge count from being spent on
/// noise, and it is what lets training stop instead of padding.
const min_pair_count = 2;

/// Adjacent token pair, left in the high 32 bits, so comparing two of them as
/// plain integers is the tie break.
const Pair = u64;

const BestPair = struct {
    key: Pair,
    count: u32,
};

fn pairKey(left: u32, right: u32) Pair {
    return (@as(u64, left) << 32) | @as(u64, right);
}

pub const Tokenizer = struct {
    /// `vocab.items[id]` is the bytes that token id stands for. An ArrayList
    /// because training appends to it a merge at a time; nothing outside this
    /// module should care where the spare capacity is.
    vocab: std.ArrayList([]u8),
    /// `merges.items[rank]` is the pair that produced token
    /// `byte_vocab_size + rank`.
    merges: std.ArrayList(Merge),
    /// Rank per adjacent pair, so encoding asks one map lookup per pair instead
    /// of walking the merge list. Built by `init` and every `addMerge`, never
    /// read from the file, so it cannot go stale.
    pair_ranks: std.AutoHashMap(Pair, u32),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) !Tokenizer {
        var vocab: std.ArrayList([]u8) = .empty;
        errdefer {
            for (vocab.items) |token| allocator.free(token);
            vocab.deinit(allocator);
        }
        try vocab.ensureTotalCapacity(allocator, byte_vocab_size);
        for (0..byte_vocab_size) |b| {
            const one = try allocator.alloc(u8, 1);
            one[0] = @intCast(b);
            try vocab.append(allocator, one);
        }
        return .{
            .vocab = vocab,
            .merges = .empty,
            .pair_ranks = .init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Tokenizer) void {
        for (self.vocab.items) |token| self.allocator.free(token);
        self.vocab.deinit(self.allocator);
        self.merges.deinit(self.allocator);
        self.pair_ranks.deinit();
        self.* = undefined;
    }

    /// Learns at most `n_merges` merges from `text`, the most frequent pair
    /// first, and stops early once no pair occurs `min_pair_count` times.
    ///
    /// ponytail: rescans the whole sequence once per merge, O(len * n_merges).
    /// Measured in a debug build: 0.5 s on the 20k byte slice the tests use, 5 s
    /// on the 100k the trainer is verified on, 55 s on the whole 1.1 MB corpus
    /// at 200 merges. A linked list of positions with incremental pair counts is
    /// the upgrade when the merge count stops being small.
    pub fn train(self: *Tokenizer, text: []const u8, n_merges: usize) !void {
        const gpa = self.allocator;
        var ids: std.ArrayList(u32) = .empty;
        defer ids.deinit(gpa);
        try ids.ensureTotalCapacity(gpa, text.len);
        for (text) |b| try ids.append(gpa, b);

        var counts: std.AutoHashMap(Pair, u32) = .init(gpa);
        defer counts.deinit();
        var next: std.ArrayList(u32) = .empty;
        defer next.deinit(gpa);

        for (0..n_merges) |_| {
            counts.clearRetainingCapacity();
            const best = (try mostFrequentPair(ids.items, &counts)) orelse break;
            if (best.count < min_pair_count) break;

            const left: u32 = @truncate(best.key >> 32);
            const right: u32 = @truncate(best.key);
            const new_id: u32 = byte_vocab_size + @as(u32, @intCast(self.merges.items.len));
            try applyMerge(ids.items, left, right, new_id, &next, gpa);
            @memcpy(ids.items[0..next.items.len], next.items);
            ids.shrinkRetainingCapacity(next.items.len);
            try self.addMerge(left, right);
        }
    }

    /// The most frequent adjacent pair, or null when there is no pair at all.
    /// One pass, with the winner tracked as the counts rise. The comparison is
    /// (count, packed key) and that is a total order, so the winner never
    /// depends on the hash map's iteration order and two runs on the same text
    /// learn the same merges.
    fn mostFrequentPair(ids: []const u32, counts: *std.AutoHashMap(Pair, u32)) !?BestPair {
        if (ids.len < 2) return null;
        var best: BestPair = .{ .key = 0, .count = 0 };
        for (ids[0 .. ids.len - 1], 0..) |left, i| {
            const slot = try counts.getOrPut(pairKey(left, ids[i + 1]));
            if (!slot.found_existing) slot.value_ptr.* = 0;
            slot.value_ptr.* += 1;
            if (slot.value_ptr.* > best.count or
                (slot.value_ptr.* == best.count and slot.key_ptr.* < best.key))
            {
                best = .{ .key = slot.key_ptr.*, .count = slot.value_ptr.* };
            }
        }
        return best;
    }

    /// Replaces every non-overlapping occurrence of (left, right) with
    /// `new_id`, scanning left to right into `out`. The scan consumes two input
    /// tokens per hit, so "aaa" under the merge (a,a) becomes [A, a] and not
    /// [A, A]: the second pair sat inside the one that fired.
    fn applyMerge(
        ids: []const u32,
        left: u32,
        right: u32,
        new_id: u32,
        out: *std.ArrayList(u32),
        gpa: std.mem.Allocator,
    ) !void {
        out.clearRetainingCapacity();
        var i: usize = 0;
        while (i < ids.len) {
            if (i + 1 < ids.len and ids[i] == left and ids[i + 1] == right) {
                try out.append(gpa, new_id);
                i += 2;
            } else {
                try out.append(gpa, ids[i]);
                i += 1;
            }
        }
    }

    fn addMerge(self: *Tokenizer, left: u32, right: u32) !void {
        const gpa = self.allocator;
        const l = self.vocab.items[left];
        const r = self.vocab.items[right];
        const bytes = try gpa.alloc(u8, l.len + r.len);
        errdefer gpa.free(bytes);
        @memcpy(bytes[0..l.len], l);
        @memcpy(bytes[l.len..], r);

        const rank: u32 = @intCast(self.merges.items.len);
        try self.vocab.append(gpa, bytes);
        errdefer _ = self.vocab.pop();
        try self.merges.append(gpa, .{ .left = left, .right = right });
        errdefer _ = self.merges.pop();
        try self.pair_ranks.put(pairKey(left, right), rank);
    }

    /// Splits `text` into token ids, applying the lowest ranked applicable
    /// merge over and over until no merge applies.
    ///
    /// The loop ends because the rank is only reported when a real adjacent
    /// pair was found, and firing a merge turns two tokens into one: the
    /// sequence strictly shortens every pass and no merge can fire twice.
    ///
    /// ponytail: one scan per applied merge, O(len * n_merges), same ceiling as
    /// `train` and the same upgrade. It is not a pre-tokenizer pass: merges may
    /// cross any byte boundary, which is what keeps the encoder to a single
    /// pass over the text and keeps the merge list a property of the corpus
    /// alone.
    pub fn encode(self: *const Tokenizer, allocator: std.mem.Allocator, text: []const u8) ![]u32 {
        var ids: std.ArrayList(u32) = .empty;
        errdefer ids.deinit(allocator);
        try ids.ensureTotalCapacity(allocator, text.len);
        for (text) |b| try ids.append(allocator, b);

        var next: std.ArrayList(u32) = .empty;
        defer next.deinit(allocator);
        while (try self.applyLowestRank(&ids, &next, allocator)) {}
        return ids.toOwnedSlice(allocator);
    }

    fn applyLowestRank(
        self: *const Tokenizer,
        ids: *std.ArrayList(u32),
        out: *std.ArrayList(u32),
        gpa: std.mem.Allocator,
    ) !bool {
        const items = ids.items;
        if (items.len < 2) return false;
        var best: u32 = std.math.maxInt(u32);
        for (items[0 .. items.len - 1], 0..) |left, i| {
            const rank = self.pair_ranks.get(pairKey(left, items[i + 1])) orelse continue;
            if (rank < best) best = rank;
        }
        if (best == std.math.maxInt(u32)) return false;
        const merge = self.merges.items[best];
        try applyMerge(items, merge.left, merge.right, byte_vocab_size + best, out, gpa);
        @memcpy(items[0..out.items.len], out.items);
        ids.shrinkRetainingCapacity(out.items.len);
        return true;
    }

    /// The bytes behind `ids`, which is the exact inverse of `encode`. An id
    /// past the vocabulary is refused rather than truncated: ids arrive from
    /// outside this module, and folding one onto token 0 would corrupt the
    /// text silently.
    pub fn decode(self: *const Tokenizer, allocator: std.mem.Allocator, ids: []const u32) ![]u8 {
        var total: usize = 0;
        for (ids) |raw| {
            // Widening u32 to usize is lossless on every Zig target, which is
            // what keeps a corrupt id from wrapping down into the valid range
            // before the check sees it.
            const id: usize = raw;
            if (id >= self.vocab.items.len) return error.TokenOutOfRange;
            total += self.vocab.items[id].len;
        }

        const out = try allocator.alloc(u8, total);
        errdefer allocator.free(out);
        var at: usize = 0;
        for (ids) |raw| {
            const token = self.vocab.items[@intCast(raw)];
            @memcpy(out[at..][0..token.len], token);
            at += token.len;
        }
        return out;
    }

    /// Writes the merge list as JSON under `dir`. The directory is a parameter
    /// rather than an assumption about the process working directory, so a
    /// caller can hand over a temporary directory and have the file land there.
    pub fn save(self: *const Tokenizer, dir: std.Io.Dir, io: std.Io, sub_path: []const u8) !void {
        var buffer: std.Io.Writer.Allocating = .init(self.allocator);
        defer buffer.deinit();
        try std.json.Stringify.value(Persisted{ .merges = self.merges.items }, .{}, &buffer.writer);
        try dir.writeFile(io, .{ .sub_path = sub_path, .data = buffer.written() });
    }

    /// The file is untrusted input: a shape std.json does not recognise, a pair
    /// wider than the tokens that exist, or a file that is not there at all come
    /// back as errors, never as a half-built vocabulary.
    pub fn load(allocator: std.mem.Allocator, dir: std.Io.Dir, io: std.Io, sub_path: []const u8) !Tokenizer {
        const bytes = try dir.readFileAlloc(io, sub_path, allocator, .limited(max_vocab_bytes));
        defer allocator.free(bytes);
        const parsed = try std.json.parseFromSlice(Persisted, allocator, bytes, .{});
        defer parsed.deinit();

        var self = try Tokenizer.init(allocator);
        errdefer self.deinit();
        for (parsed.value.merges, 0..) |merge, rank| {
            // Only the 256 bytes and the `rank` merges before this one exist
            // when this merge is applied, so a larger id on either side is a
            // forward reference the file is not allowed to make.
            const known: u32 = byte_vocab_size + @as(u32, @intCast(rank));
            if (merge.left >= known or merge.right >= known) return error.InvalidMerge;
            try self.addMerge(merge.left, merge.right);
        }
        return self;
    }
};
