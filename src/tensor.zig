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
pub fn matmul(a: Tensor, b: Tensor) !Tensor {
    if (a.cols != b.rows) return error.DimensionMismatch;
    var out = try Tensor.init(a.allocator, a.rows, b.cols);

    for (0..a.rows) |i| {
        const a_row = a.rowConst(i);
        const out_row = out.row(i);
        for (0..a.cols) |k| {
            const scale = a_row[k];
            const b_row = b.rowConst(k);
            for (0..b.cols) |j| out_row[j] += scale * b_row[j];
        }
    }
    return out;
}
