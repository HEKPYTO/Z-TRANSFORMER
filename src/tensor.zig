const std = @import("std");

pub const Tensor = struct {
    data: []f32,
    rows: usize,
    cols: usize,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, rows: usize, cols: usize) !Tensor {
        // The product is checked because a Tensor's shape is what its bounds
        // guards read. A wrap here would hand back a buffer shorter than
        // rows * cols while `rows` and `cols` still report the shapes asked
        // for, and every later guard would pass.
        const n = std.math.mul(usize, rows, cols) catch return error.DimensionOverflow;
        const data = try allocator.alloc(f32, n);
        @memset(data, 0);
        return .{ .data = data, .rows = rows, .cols = cols, .allocator = allocator };
    }

    pub fn deinit(self: *Tensor) void {
        self.allocator.free(self.data);
        self.data = &.{};
    }

    pub fn at(self: Tensor, r: usize, c: usize) f32 {
        return self.data[self.offset(r, c)];
    }

    pub fn set(self: *Tensor, r: usize, c: usize, v: f32) void {
        self.data[self.offset(r, c)] = v;
    }

    fn offset(self: Tensor, r: usize, c: usize) usize {
        if (r >= self.rows or c >= self.cols) @panic("tensor index out of range");
        return r * self.cols + c;
    }

    pub fn row(self: *Tensor, r: usize) []f32 {
        return self.data[self.rowStart(r)..][0..self.cols];
    }

    pub fn rowConst(self: Tensor, r: usize) []const f32 {
        return self.data[self.rowStart(r)..][0..self.cols];
    }

    fn rowStart(self: Tensor, r: usize) usize {
        if (r >= self.rows) @panic("tensor row out of range");
        return r * self.cols;
    }

    pub fn fill(self: *Tensor, v: f32) void {
        @memset(self.data, v);
    }
};

/// [m,k] @ [k,n] -> [m,n].
///
/// The i-k-j order keeps the reduction over k in one fixed sequence per output
/// element, so a run is bit-identical to the last one, and it walks both operand
/// rows and the output row as contiguous slices.
///
/// Eight output columns are held in registers across the whole k loop rather
/// than one being read, added to and written back on every k. Bit-exact, and
/// the argument is the same one `weightGrad` and `inputGrad` already rest on:
/// each `acc[u]` still sums k in ascending order, so it rounds to f32 exactly
/// where the memory-resident version did, and an f32 store followed by an f32
/// load is lossless, so the value sequence is identical element for element.
/// The lanes are never summed into one another.
///
/// Measured, on the Linux host rather than the Mac, because the Mac has a 21%
/// floor between `zig build bench` invocations and cannot see a 7% effect:
/// 1.7618 to 1.6346 CPU-seconds per step, **7.2%**, which is 18 times that
/// host's 0.41% floor. 1.7618 is the mean of two baseline invocations, 1.7654
/// and 1.7581; the 7.2% is against that mean and 7.4% against the slower one. The loss curve is byte-identical on both
/// hosts, which is the check that matters: the same argument that says the
/// lanes are never summed says the f32 rounding sequence is unchanged, and two
/// independent digests agreeing is what it was worth verifying for.
pub fn matmul(a: Tensor, b: Tensor) !Tensor {
    if (a.cols != b.rows) return error.DimensionMismatch;
    var out = try Tensor.init(a.allocator, a.rows, b.cols);

    const lanes = 8;
    for (0..a.rows) |i| {
        const a_row = a.rowConst(i);
        const out_row = out.row(i);
        var j: usize = 0;
        while (j + lanes <= b.cols) : (j += lanes) {
            var acc: [lanes]f32 = @splat(0);
            for (0..a.cols) |k| {
                const scale = a_row[k];
                const b_row = b.rowConst(k);
                inline for (0..lanes) |u| acc[u] += scale * b_row[j + u];
            }
            inline for (0..lanes) |u| out_row[j + u] = acc[u];
        }
        // The same cursor, so the tail is written once per row and never twice,
        // and the shapes the model builds -- `d_model` 128, `ffn_dim` 512,
        // `vocab_size` 456 -- are all multiples of eight anyway.
        while (j < b.cols) : (j += 1) {
            var acc: f32 = 0;
            for (0..a.cols) |k| acc += a_row[k] * b.rowConst(k)[j];
            out_row[j] = acc;
        }
    }
    return out;
}
