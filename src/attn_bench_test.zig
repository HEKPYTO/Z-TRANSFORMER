//! Tests for the attention benchmark's arithmetic, not for its timing.
//!
//! A clock cannot be asserted on, and the point of the three functions under test
//! is that they are pure: given a context length and the shipped head geometry
//! they return the bytes, the floor and the verdict a reader is shown. So these
//! pin the numbers the table prints, which is what a reader would otherwise have
//! to take on trust, and they pin the side the verdict is chosen on -- a verdict
//! that quietly flipped would make the whole instrument argue the opposite of
//! what it measures, and nothing else in the tree would notice.

const std = @import("std");
const attn_bench = @import("attn_bench.zig");
const attention = @import("attention.zig");

test "pcieBytes counts q, k, v in and the result out" {
    // 2 * T * n_heads * dim + 2 * T * n_kv_heads * dim, in f32.
    // 4 heads of 32 over 2 kv heads at T = 256:
    //   (2 * 256 * 4 * 32 + 2 * 256 * 2 * 32) * 4
    // = (65536 + 32768) * 4 = 393216 bytes = 0.375 MiB.
    const bytes = attn_bench.pcieBytes(256, 4, 2, 32);
    try std.testing.expectEqual(@as(u64, 393216), bytes);
    try std.testing.expectApproxEqAbs(@as(f64, 0.375), @as(f64, @floatFromInt(bytes)) / (1024.0 * 1024.0), 1e-9);

    // Every term scales with T, so doubling T doubles the bytes exactly.
    try std.testing.expectEqual(bytes * 2, attn_bench.pcieBytes(512, 4, 2, 32));
    try std.testing.expectEqual(bytes * 16, attn_bench.pcieBytes(4096, 4, 2, 32));
}

test "floorUs is the documented rate, not an assumed one" {
    // 393216 bytes at the 6.1 GB/s this project measured on its own card.
    try std.testing.expectApproxEqAbs(64.46, attn_bench.floorUs(393216), 0.01);
    try std.testing.expectApproxEqAbs(1031.39, attn_bench.floorUs(6291456), 0.01);
    // And it is the same bandwidth the constant names, not a second one.
    try std.testing.expectEqual(attn_bench.pcie_bytes_per_sec, 6.1e9);
}

test "verdict picks the side the ratio implies" {
    // cpu above the floor means the bus is not what stops a kernel, so the
    // kernel's own cost decides. That is the side every row of the shipped sweep
    // landed on, and it is the side that argues FOR writing the kernel.
    try std.testing.expectEqualStrings(
        "floor is below the cpu: the kernel's own cost decides",
        attn_bench.verdict(8344.03, 64.46),
    );
    // cpu under the floor means no kernel that must move those bytes can win, and
    // that direction has to be reachable or the table cannot report a crossing.
    try std.testing.expectEqualStrings(
        "cpu under the floor: no kernel of this shape can win",
        attn_bench.verdict(10.0, 64.46),
    );
    // Equality is not "under", so it takes the decisive branch rather than
    // claiming a win that is really a tie.
    try std.testing.expectEqualStrings(
        "floor is below the cpu: the kernel's own cost decides",
        attn_bench.verdict(64.46, 64.46),
    );
}

test "the head geometry the tool reports is the shipped one" {
    // LITERALS, and that is the whole point of the rewrite. `attention.defaultConfig`
    // is DEFINED as three fields copied out of `model.defaultConfig`
    // (`src/attention.zig:24-27`), so the earlier version of this test compared
    // a struct against the three fields it was built from and could not fail --
    // a value against itself, the identical mistake this file's own header
    // records for `rope_theta` in `src/rope_test.zig:4-22`. Nothing in the tree
    // pinned 4 / 2 / 32 anywhere: `model_test.zig` pins `n_layers`, `n_ctx` and
    // `n_heads % n_kv_heads == 0`, and `dModel` only pins the product. A silent
    // edit to the shipped split left the whole sweep measuring a shape nothing
    // runs, and this is the assertion that now says so.
    const cfg = attention.defaultConfig();
    try std.testing.expectEqual(@as(usize, 4), cfg.n_heads);
    try std.testing.expectEqual(@as(usize, 2), cfg.n_kv_heads);
    try std.testing.expectEqual(@as(usize, 32), cfg.head_dim);
    // GQA, not MHA: the floor is smaller than a full head count would give, and
    // that is the whole reason kv heads exist.
    try std.testing.expect(cfg.n_heads > cfg.n_kv_heads);
    // And the shape the bench builds its tensors from is the shape these three
    // fields make, which is what "the tool reports the shipped geometry" means:
    // `print` allocates `cfg.n_heads * cfg.head_dim` columns for q, so a split
    // that disagreed with its own arithmetic would be caught here too.
    try std.testing.expectEqual(@as(usize, 128), cfg.n_heads * cfg.head_dim);
    try std.testing.expectEqual(@as(usize, 64), cfg.n_kv_heads * cfg.head_dim);
}
