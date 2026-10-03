//! A key/value cache for autoregressive generation: per layer, the projected keys
//! and values for every position generated so far.
//!
//! WHAT THIS IS AND IS NOT. It is the storage and the one rule that governs it. It
//! is not a decode loop, and there is no generation code in this repository -- a
//! search for `generate`, `sample` or `decode` over `src/` finds only `scale.zig`'s
//! cost formula and `tokenizer.decode`, which turns ids back into text and is
//! unrelated. What is missing to make generation work is a caller that feeds one
//! token at a time and reads logits out; this file is the piece that has to exist
//! before that caller can, and it is deliberately not more.
//!
//! WHY A TRAINING STEP DOES NOT NEED IT. `data.Batcher` hands the model
//! `inputs[t]` and `targets[t+1]` for a whole window in one batch, so the full
//! sequence is always present and the forward recomputes every position's keys and
//! values from scratch. A cache is a generation feature, not a training one, and
//! making it a prerequisite for the CUDA wiring would have coupled two independent
//! pieces of work.
//!
//! THE ONE RULE, AND IT IS THE ONE THAT IS EASY TO GET WRONG. **The cache stores
//! `k` AFTER RoPE, never before.** `model.zig` rotates `q` and `k` and leaves `v`
//! alone, so at position 0 the cache receives `k_pos` and `v` as projected. Two
//! things follow, and both are silent failures:
//!
//!   1. Storing pre-rotation `k` and rotating again at decode time looks
//!      harmless, because RoPE is a rotation and rotations compose. They compose
//!      to `R(2t)`, not `R(t)`, so every cached key would be rotated to the wrong
//!      angle and nothing would crash.
//!   2. Rotating `v` as well, because it is sitting right there in the same place,
//!      is a plain wrong answer with no compensating error.
//!
//! So `append` takes the ALREADY-ROTATED key, and `appendRotated` exists only for
//! the case where a caller holds a raw projection and wants the rotation applied
//! here. There is no API that stores a pre-rotation key by accident, because
//! `append` does not know what it is being given and cannot check.
//!
//! SIZING. `scale.zig` already costs this: `kv_cache = n_layers *
//! (n_kv_heads * head_dim) * n_ctx * 2 * sizeof(f32)`, which at the shipped 4-layer
//! shape is 524288 bytes. A test below pins this file's own arithmetic to that
//! number, so the two cannot drift -- see the note on that test for how far the
//! check actually reaches.

const std = @import("std");
const rope = @import("rope.zig");
const tensor = @import("tensor.zig");

const Tensor = tensor.Tensor;

pub const Error = error{
    /// The cache already holds `n_ctx` positions, so there is nowhere to put
    /// another. Distinct from an allocation failure because the fix is to stop
    /// generating or to size the cache larger, not to retry.
    Full,
    /// A row or column count that cannot be this layer's keys or values.
    DimensionMismatch,
};

/// One layer's cache. `k` and `v` are both `[n_ctx, n_kv_heads * head_dim]`, which
/// is what `k` and `v` look like at the attention boundary.
pub const Layer = struct {
    k: Tensor,
    v: Tensor,
    len: usize,

    pub fn init(allocator: std.mem.Allocator, n_ctx: usize, width: usize) !Layer {
        if (n_ctx == 0 or width == 0) return Error.DimensionMismatch;
        return .{
            .k = try Tensor.init(allocator, n_ctx, width),
            .v = try Tensor.init(allocator, n_ctx, width),
            .len = 0,
        };
    }

    pub fn deinit(self: *Layer) void {
        self.k.deinit();
        self.v.deinit();
        self.len = 0;
    }

    /// Append one position's keys and values.
    ///
    /// `k` MUST already be rotated. See the file header for why that is not
    /// checkable here and what goes wrong otherwise.
    pub fn append(self: *Layer, k: []const f32, v: []const f32) Error!void {
        if (k.len != self.k.cols or v.len != self.v.cols) return Error.DimensionMismatch;
        if (self.len >= self.k.rows) return Error.Full;
        const r = self.len;
        @memcpy(self.k.row(r), k);
        @memcpy(self.v.row(r), v);
        self.len += 1;
    }

    /// Append one position from RAW projections, rotating the key here.
    ///
    /// This is the only way a caller can get rotation applied, and it is a separate
    /// function on purpose. `append` cannot tell a rotated key from an unrotated one
    /// -- both are `[width]` floats -- so offering one function that sometimes
    /// rotates and sometimes does not would make the mistake undetectable at the
    /// call site. Two named functions make it a choice.
    pub fn appendRotated(
        self: *Layer,
        k_raw: []const f32,
        v: []const f32,
        pos: usize,
        theta: f64,
        head_dim: usize,
        allocator: std.mem.Allocator,
    ) !void {
        if (k_raw.len != self.k.cols) return Error.DimensionMismatch;
        // One row in, one row out. `rope.forward` takes a whole tensor because the
        // training path rotates a whole sequence at once; here there is exactly one
        // position, and allocating per token is acceptable for a first decode loop
        // but is noted rather than hidden.
        var one = try Tensor.init(allocator, 1, k_raw.len);
        defer one.deinit();
        @memcpy(one.row(0), k_raw);
        var rotated = try rope.forward(allocator, one, pos, theta, head_dim);
        defer rotated.deinit();
        try self.append(rotated.rowConst(0), v);
    }

    /// How many positions a query at `pos` may attend to: every cached one.
    ///
    /// There is no mask to apply and no upper bound to clamp, which is the whole
    /// reason a cache is faster than recomputing. The caller passes `min(pos + 1,
    /// len)`-worth of keys; `attention.forwardWith` already handles a query whose
    /// row index differs from the key indices only if it is told where to start,
    /// which is NOT yet a parameter it has. See the file header's "what is missing".
    pub fn visible(self: *const Layer) usize {
        return self.len;
    }
};

/// Every layer's cache, plus the position the next token will occupy.
///
/// `len` is kept here as well as per layer because a generation loop advances one
/// position for all layers together, and deriving it from `layers[0].len` would
/// make a caller that had somehow desynchronised them silently produce a shorter
/// context rather than an error.
pub const Cache = struct {
    layers: []Layer,
    len: usize,

    pub fn init(
        allocator: std.mem.Allocator,
        n_layers: usize,
        n_ctx: usize,
        width: usize,
    ) !Cache {
        if (n_layers == 0) return Error.DimensionMismatch;
        const layers = try allocator.alloc(Layer, n_layers);
        // errdefer frees the ones already built, and `built` counts them -- the same
        // shape as the device holder's cleanup, for the same reason: a partial
        // construction must not leak.
        var built: usize = 0;
        errdefer {
            for (layers[0..built]) |*l| l.deinit();
            allocator.free(layers);
        }
        for (layers) |*l| {
            l.* = try Layer.init(allocator, n_ctx, width);
            built += 1;
        }
        return .{ .layers = layers, .len = 0 };
    }

    pub fn deinit(self: *Cache, allocator: std.mem.Allocator) void {
        for (self.layers) |*l| l.deinit();
        allocator.free(self.layers);
        self.layers = &.{};
        self.len = 0;
    }

    /// One position across every layer. `k` and `v` are the already-rotated key
    /// and the raw value, for each layer, in order.
    pub fn appendAll(self: *Cache, k: []const Tensor, v: []const Tensor) Error!void {
        if (k.len != self.layers.len or v.len != self.layers.len)
            return Error.DimensionMismatch;
        var i: usize = 0;
        // Not `errdefer`: a half-appended position is worse than a refused one, and
        // rolling back would need each layer's prior `len`. The loop below therefore
        // checks EVERY layer can take the write before ANY of them does.
        while (i < self.layers.len) : (i += 1) {
            const l = &self.layers[i];
            // `.data.len`, not `.cols`: a caller may hand a tensor whose declared
            // shape is anything, and `append` copies `data.len` floats, so that is the
            // number the bound has to be checked against. Tensor has no `len` field
            // at all -- it is { data, rows, cols, allocator } -- so the earlier
            // version of this line did not compile.
            if (k[i].data.len != l.k.cols or v[i].data.len != l.v.cols)
                return Error.DimensionMismatch;
            if (l.len >= l.k.rows) return Error.Full;
        }
        for (self.layers, 0..) |*l, li| try l.append(k[li].rowConst(0), v[li].rowConst(0));
        self.len += 1;
    }

    /// Whether another position would not fit.
    ///
    /// An empty `layers` is FULL rather than an error or a read: `deinit` empties
    /// the slice, and both this and `bytes` below used to index `layers[0]`
    /// straight after it, which Safe panics on and ReleaseFast reads past the end
    /// of. An empty cache has nowhere to put a position, so `true` is the answer
    /// that keeps a caller from asking one, and neither is `pub` API that any
    /// in-tree caller reaches after `deinit` -- which is exactly why it was
    /// undefended.
    pub fn full(self: *const Cache) bool {
        if (self.layers.len == 0) return true;
        return self.len >= self.layers[0].k.rows;
    }

    /// Bytes held, for the same reason `zig build`'s peak-RSS gate exists: a number
    /// that should match a documented figure, checked rather than assumed.
    ///
    /// Zero on a deinit'd cache, for the reason `full` gives.
    pub fn bytes(self: *const Cache) usize {
        if (self.layers.len == 0) return 0;
        const l = &self.layers[0];
        return self.layers.len * (l.k.data.len + l.v.data.len) * @sizeOf(f32);
    }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;

test "the shipped shape holds 524288 bytes, which is what scale.zig costs" {
    // A `Cache`, built at the shipped shape, asked what it holds. The earlier
    // version of this test built none and called no `bytes()`: it transcribed
    // `scale.zig`'s `kv_cache = l * kv * T * 2 * 4` term (`kv = n_kv_heads *
    // head_dim`) into three local constants and asserted that the arithmetic
    // worked. No edit to this file could fail it, which the comment at the time
    // admitted while the test's NAME and the file header's claim ("A test below
    // pins this file's own arithmetic to that number, so the two cannot drift")
    // said otherwise. What it can now reach is `Cache.bytes` on the real
    // allocation, at the shape `scale.zig` costs.
    //
    // Still not a call into `scale.zig`, so an edit to `scale.zig` alone still
    // passes this. The 524288 below is transcribed from it and is what a reader
    // budgets against; what this file can drift from is its own arithmetic, and
    // that is now the thing under test.
    var c = try Cache.init(testing.allocator, 4, 256, 2 * 32);
    defer c.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 524288), c.bytes());
}

test "a cache refuses to grow past n_ctx" {
    var c = try Cache.init(testing.allocator, 2, 3, 8);
    defer c.deinit(testing.allocator);

    var k1 = [_]f32{0} ** 8;
    var v1 = [_]f32{0} ** 8;
    var k2 = [_]f32{1} ** 8;
    var v2 = [_]f32{1} ** 8;

    // Two layers, so TWO tensors per side. An earlier version of this test built a
    // two-layer cache and passed `ks[0..1]`, and the one-layer slice was correctly
    // refused -- the test was wrong about its own fixture, not the code lenient.
    const ks = [_]Tensor{
        .{ .data = &k1, .rows = 1, .cols = 8, .allocator = testing.allocator },
        .{ .data = &k2, .rows = 1, .cols = 8, .allocator = testing.allocator },
    };
    const vs = [_]Tensor{
        .{ .data = &v1, .rows = 1, .cols = 8, .allocator = testing.allocator },
        .{ .data = &v2, .rows = 1, .cols = 8, .allocator = testing.allocator },
    };

    // A layer-count mismatch is refused, and refused before anything is written.
    try testing.expectError(Error.DimensionMismatch, c.appendAll(ks[0..1], vs[0..]));
    try testing.expectEqual(@as(usize, 0), c.len);

    // n_ctx is 3, so the third position fits and the fourth does not.
    try c.appendAll(&ks, &vs);
    try testing.expectEqual(@as(usize, 1), c.len);
    try c.appendAll(&ks, &vs);
    try testing.expectEqual(@as(usize, 2), c.len);
    // Every layer advanced together, which is the property `Cache.len` exists to hold.
    try testing.expectEqual(@as(usize, 2), c.layers[0].visible());
    try testing.expectEqual(@as(usize, 2), c.layers[1].visible());
    try c.appendAll(&ks, &vs);
    try testing.expectEqual(@as(usize, 3), c.len);
    try testing.expect(c.full());
    try testing.expectError(Error.Full, c.appendAll(&ks, &vs));
    try testing.expectEqual(@as(usize, 3), c.len); // a refusal must not advance
}

test "a wrong-width key is refused and leaves the cache untouched" {
    var c = try Cache.init(testing.allocator, 1, 4, 8);
    defer c.deinit(testing.allocator);
    var k = [_]f32{0} ** 8;
    var short = [_]f32{0} ** 4;
    const good = [_]Tensor{Tensor{ .data = &k, .rows = 1, .cols = 8, .allocator = testing.allocator }};
    const bad = [_]Tensor{Tensor{ .data = &short, .rows = 1, .cols = 4, .allocator = testing.allocator }};

    // First the key side, then the value side: both are width checks and both must
    // leave the cache untouched rather than half-appending a position.
    try testing.expectError(Error.DimensionMismatch, c.appendAll(&good, &bad));
    try testing.expectEqual(@as(usize, 0), c.len);
    try testing.expectError(Error.DimensionMismatch, c.appendAll(&bad, &good));
    try testing.expectEqual(@as(usize, 0), c.len);
}

test "appendRotated applies RoPE and a second call at the next position differs" {
    // The property that makes the file worth having: rotation is applied ONCE, at
    // the position the token actually occupies. Two positions of the same raw key
    // must produce different cached rows, because that is the entire effect of RoPE
    // and the reason a cache stores the rotated form at all.
    var l = try Layer.init(testing.allocator, 8, 4);
    defer l.deinit();
    var raw = [_]f32{ 1, 0, 0, 1 };
    var v = [_]f32{ 9, 9, 9, 9 };

    try l.appendRotated(&raw, &v, 0, 500000, 2, testing.allocator);
    try l.appendRotated(&raw, &v, 1, 500000, 2, testing.allocator);
    try testing.expectEqual(@as(usize, 2), l.len);

    const a0 = l.k.rowConst(0);
    const a1 = l.k.rowConst(1);
    // At pos 0 every angle is 0, so the rotation is the identity.
    try testing.expectApproxEqAbs(@as(f32, 1), a0[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0), a0[1], 1e-6);
    // At pos 1 the first frequency is 500000^0 = 1, so the angle is 1 radian and
    // the first column has moved. This is the assertion that would fail if the
    // cache stored pre-rotation keys, because then both rows would be identical.
    try testing.expect(a1[0] != a0[0] or a1[1] != a0[1]);
    // v is never rotated.
    try testing.expectApproxEqAbs(@as(f32, 9), l.v.rowConst(0)[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 9), l.v.rowConst(1)[0], 1e-6);
}

test "bytes matches n_layers * n_ctx * width * 2 * sizeof(f32)" {
    var c = try Cache.init(testing.allocator, 3, 16, 5);
    defer c.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 3 * 16 * 5 * 2 * @sizeOf(f32)), c.bytes());
}

test "a deinit'd cache answers full and bytes instead of indexing an empty slice" {
    // `deinit` sets `layers = &.{}`, and `full` and `bytes` both read `layers[0]`.
    // Called after it they indexed a zero-length slice: Safe panics on the bounds
    // check and ReleaseFast reads past the end, which is the release the test
    // binary also runs in. Neither has an in-tree caller that does this, which is
    // why the gap was open; both are `pub`, so a caller outside the tree can.
    // `testing.allocator` is what makes a leak in the half-built fixture fail too.
    var c = try Cache.init(testing.allocator, 2, 4, 8);
    c.deinit(testing.allocator);

    // Full, because a cache with no layers has nowhere to put a position. Zero
    // bytes, because it holds nothing.
    try testing.expect(c.full());
    try testing.expectEqual(@as(usize, 0), c.bytes());
    // And they are still answerable twice over, so nothing here caches a stale
    // answer or frees a second time.
    try testing.expect(c.full());
    try testing.expectEqual(@as(usize, 0), c.bytes());
}
