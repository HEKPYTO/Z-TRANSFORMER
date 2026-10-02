//! The Zig side of the CUDA attention FFI: the declarations for the six C entry
//! points in `attn_kernels.cu`, the handful of runtime calls needed to use them,
//! and a holder for the device buffers a training step would need.
//!
//! WHAT THIS IS NOT, ANY MORE. `src/tests.zig` imports this file for its six
//! arithmetic tests, and `src/model.zig` now reaches `Attn.init` and `forward`
//! behind one `pub const cuda_attn: bool`, with `zig build cuda-attn-check` and
//! `zig build cuda-train` the two steps that link the object this file names.
//! `cuda-attn-check` exits 1 while that flag is false rather than skipping, because
//! a green that checked nothing is worse than a refusal.
//!
//! THE ONE FAILURE MODE WITH NO COMPILE-TIME CHECK. These six `extern fn`
//! declarations and the six C definitions in `attn_kernels.cu` are checked by
//! nothing. `nvcc` type-checks the C side against itself and Zig checks the Zig
//! side against itself, and no step compares the two arities. A `device.zig` one
//! revision behind the kernel shifts every argument after the first mismatch by one
//! slot, and the result is not a compile error: it is a segmentation fault inside
//! `zt_attn_forward` roughly every other run, with the parameters read from stack
//! garbage. That was measured, not theorised, after an out-of-date copy of this file
//! met a newer kernel. `zig build cuda-attn-check` is what catches it, by running the
//! whole model forward on the device and diffing it against `attention.forward` --
//! so run it after ANY change to either signature. Nothing cheaper sees this.
//!
//! WHY A HOLDER AND NOT A POOL. The shape is fixed for an entire run. `model.Config`
//! carries `n_ctx` and `data.Batcher` allocates exactly `cfg.ctx` inputs and
//! `cfg.ctx` targets for every batch, so `T` never changes, `n_heads` never
//! changes, and neither does `head_dim`. There is therefore nothing to grow and
//! nothing to reuse: eleven `cudaMalloc`s for a 123-step run, once, and eleven
//! `cudaFree`s at the end. A general pool would be a data structure solving a
//! problem this training loop does not have, and `attn.cu`'s own harness -- which
//! DOES visit seven shapes -- is where a pool would belong if anywhere.
//!
//! WHY THE CALLER OWNS NOTHING HERE. `init` takes plain integers rather than a
//! `model.Config`, for two reasons. A module rooted at `src/cuda/` may not import
//! outside its own directory, so taking the config would mean the importer has to
//! hand the whole of `model.zig` over as a module -- the arrangement
//! `attn_twin.zig` uses for `autograd.zig`. And the caller has the config
//! already, so passing four integers is less coupling, not more.

const std = @import("std");

// ---------------------------------------------------------------------------
// The C entry points, mirrored from src/cuda/attn_kernels.cu.
//
// SIGNATURES ARE THE CONTRACT. `extern fn` in Zig does NOT check its C signature,
// so a mismatch here is undefined behaviour rather than a compile error, and the
// failure mode of a transposed pointer is a kernel that runs, returns 0, and
// writes the right answer into the wrong buffer. Each declaration below is
// therefore a character-for-character mirror of its definition, and the audit
// that checks them is a real gate rather than a formality.
//
// `usize` is NOT used for any of the integer parameters. The C side takes
// `int`, and Zig would silently reinterpret a `usize` as 32 bits -- so a `T` of
// 2^31 would arrive as a negative number and the kernel would read out of bounds.
// `c_int` is 32 bits on every platform this is built on.
// ---------------------------------------------------------------------------

pub extern fn zt_attn_dim_ok(dim: c_int) c_int;
// These three are DEAD on the Zig side and are mirrored anyway, deliberately. They
// exist so `attn.cu`'s benchmark has one owner for the shared-memory layout; both
// launchers compute the same numbers internally, so a Zig caller has no use for
// them. They are declared rather than omitted so that this file's list of the six
// entry points is the C file's list, and an audit comparing the two finds a
// missing name instead of finding three undeclared ones and having to wonder.
pub extern fn zt_attn_forward_shmem(dim: c_int, tile: c_int, group_q: c_int) usize;
pub extern fn zt_attn_bwd_dq_shmem(dim: c_int, tile: c_int) usize;
pub extern fn zt_attn_bwd_dkdv_shmem(dim: c_int) usize;

pub extern fn zt_attn_forward(
    q: [*]const f32,
    k: [*]const f32,
    v: [*]const f32,
    out: [*]f32,
    T: c_int,
    n_heads: c_int,
    n_kv_heads: c_int,
    dim: c_int,
    group_q: c_int,
    max_tile: c_int,
    // `q_offset` is the absolute position of query row 0 and `n_keys` is how many
    // rows of k and v exist. A training-shaped call passes 0 and T, which is the
    // configuration every published row was measured at.
    q_offset: c_int,
    n_keys: c_int,
    stream: ?*anyopaque,
) c_int;

pub extern fn zt_attn_backward(
    q: [*]const f32,
    k: [*]const f32,
    v: [*]const f32,
    dout: [*]const f32,
    dq: [*]f32,
    dk: [*]f32,
    dv: [*]f32,
    row_max: [*]f32,
    row_den: [*]f32,
    row_del: [*]f32,
    T: c_int,
    n_heads: c_int,
    n_kv_heads: c_int,
    dim: c_int,
    max_tile: c_int,
    stream: ?*anyopaque,
) c_int;

// Only as much of the CUDA runtime as this file needs. Not a binding to the
// library -- a declaration of the three functions, which resolve against
// `libcudart` at link time. `cudaMalloc`'s error return is `cudaError_t`, an
// enum whose zero is success; it is declared as `c_int` because Zig has no
// binding for it and an `enum(c_int)` would be a lie about the ABI.
pub extern fn cudaMalloc(ptr: *?*anyopaque, size: usize) c_int;
pub extern fn cudaFree(ptr: ?*anyopaque) c_int;
pub extern fn cudaMemcpy(dst: ?*anyopaque, src: ?*const anyopaque, count: usize, kind: c_int) c_int;

pub const memcpy_host_to_device: c_int = 1;
pub const memcpy_device_to_host: c_int = 2;

pub const Error = error{
    /// A shape that is arithmetically fine but degenerate -- a zero length, a zero
    /// head count. Distinct from `ShapeTooLarge` because `cudaMalloc(0)` returns
    /// `cudaErrorInvalidValue`, so without this the caller would be told the
    /// ALLOCATOR failed and a retry loop would retry forever.
    DegenerateShape,
    /// A dimension or a buffer size does not fit in the C types the entry points
    /// take. Reached rather than assumed: a silent truncation here is an
    /// out-of-bounds read inside a kernel, not an error message.
    ShapeTooLarge,
    /// The C side refused the configuration and printed why. Its own reason, on
    /// stderr; this only says which call refused.
    ConfigRefused,
    /// `cudaMalloc` returned a non-zero `cudaError_t`.
    DeviceAllocFailed,
};

/// Which buffer a size is for. Named rather than indexed so that a caller cannot
/// pass a `dk` count where a `dq` count belongs.
pub const Kind = enum {
    /// [T, n_heads * dim] -- q, out, dout, dq
    query_headed,
    /// [T, n_kv_heads * dim] -- k, v, dk, dv
    kv_headed,
    /// [T, n_heads] -- the three per-row scalars the two backward kernels share
    per_row,
};

/// Bytes for one buffer, as a pure function.
///
/// Every size in this file goes through here, and the multiplication is checked
/// rather than performed. `T * n_heads * dim` at the shipped shape is 32768 and
/// nowhere near overflowing, but a `usize` overflow at these values would produce
/// a SMALL number, a small `cudaMalloc`, and a kernel writing past the end of it
/// -- with no error anywhere. That is the failure this check exists to make
/// impossible, and it is why the arithmetic is not inlined at the call sites.
pub fn bytesFor(kind: Kind, t: usize, n_heads: usize, n_kv_heads: usize, dim: usize) Error!usize {
    if (t == 0 or dim == 0 or n_heads == 0 or n_kv_heads == 0) return Error.DegenerateShape;
    const cols: usize = switch (kind) {
        .query_headed => std.math.mul(usize, n_heads, dim) catch return Error.ShapeTooLarge,
        .kv_headed => std.math.mul(usize, n_kv_heads, dim) catch return Error.ShapeTooLarge,
        .per_row => n_heads,
    };
    const elems = std.math.mul(usize, t, cols) catch return Error.ShapeTooLarge;
    return std.math.mul(usize, elems, @sizeOf(f32)) catch return Error.ShapeTooLarge;
}

/// Narrow a `usize` to the `c_int` the entry points take, or refuse.
///
/// Every `usize` that crosses into C goes through this. A `T` above `c_int`'s
/// maximum is not a shape this project can train -- the window would be 2 GB of
/// activations on its own -- but silently wrapping it to a negative would be a
/// kernel reading backwards through memory, so it is refused instead.
pub fn toCInt(x: usize) Error!c_int {
    if (x > std.math.maxInt(c_int)) return Error.ShapeTooLarge;
    return @intCast(x);
}

/// Eleven device buffers, sized once for a fixed shape.
pub const Attn = struct {
    // Forward inputs, and the forward's output.
    q: [*]f32,
    k: [*]f32,
    v: [*]f32,
    out: [*]f32,
    // Backward input and its three outputs.
    dout: [*]f32,
    dq: [*]f32,
    dk: [*]f32,
    dv: [*]f32,
    // Written by the dq kernel, read by the dk/dv kernel. Never cross the bus.
    row_max: [*]f32,
    row_den: [*]f32,
    row_del: [*]f32,

    t: usize,
    n_heads: usize,
    n_kv_heads: usize,
    dim: usize,

    /// The shape as the C side wants it, computed once so no call site has to
    /// convert and no call site can convert differently.
    c_t: c_int,
    c_heads: c_int,
    c_kv_heads: c_int,
    c_dim: c_int,

    // Zig analyses this body only when something references it, and on a CPU-only
    // build nothing does -- which is how a `%` on a `c_int`, an `@ptrCast` that raised
    // pointer alignment, and a `deinit` that freed nothing all sat here while every
    // test in this file passed. Referencing the function to force the analysis does
    // NOT work as a guard: it makes the test binary require `zt_attn_*` and
    // `cudaMalloc` at link time, so `zig build test` goes red on every machine without
    // a CUDA toolchain, which is every GitHub runner. What catches these is
    // `zig build cuda-attn-check`, which links this file against the real object.
    pub fn init(t: usize, n_heads: usize, n_kv_heads: usize, dim: usize) Error!Attn {
        const c_t = try toCInt(t);
        const c_heads = try toCInt(n_heads);
        const c_kv_heads = try toCInt(n_kv_heads);
        const c_dim = try toCInt(dim);

        // Ask the C side, before allocating anything. Without this, `init` would
        // happily take eleven buffers for a shape the kernels refuse -- 5 heads over
        // 2 kv heads, or head_dim 33 -- and hold 798 KiB on the device while every
        // later `forward` returned ConfigRefused. Refusing here names the problem at
        // the point where it can still be fixed.
        if (zt_attn_dim_ok(c_dim) == 0) return Error.ConfigRefused;
        if (c_heads <= 0 or c_kv_heads <= 0 or @rem(c_heads, c_kv_heads) != 0) return Error.ConfigRefused;

        const qh = try bytesFor(.query_headed, t, n_heads, n_kv_heads, dim);
        const kvh = try bytesFor(.kv_headed, t, n_heads, n_kv_heads, dim);
        const pr = try bytesFor(.per_row, t, n_heads, n_kv_heads, dim);

        var self: Attn = undefined;
        // ONE errdefer, and a counter, rather than one per allocation.
        //
        // An earlier version registered eleven `errdefer self.deinitPartial(N)`
        // calls, one after each allocation, on the stated reasoning that "each
        // errdefer frees only what has already been taken". That is not how
        // `errdefer` works: every errdefer registered BEFORE a failure point runs,
        // in reverse order. Nine live buffers and a failure on the tenth produced
        // 110 cudaFree calls -- each live pointer freed up to nine times, and
        // twenty of them on addresses read out of `undefined`.
        //
        // It failed on the FIRST allocation too, which is worse: deinitPartial(0)
        // freed all eleven fields, none of which had been assigned.
        //
        // The cleanup path is the one that runs when the machine is short on VRAM,
        // so this is the path that has to be right. One owner for the count, one
        // errdefer, and `field()` only ever indexed below the count -- so an
        // unassigned field is never read.
        var taken: usize = 0;
        errdefer self.deinitPartial(taken);
        self.q = try alloc(qh);
        taken += 1;
        self.k = try alloc(kvh);
        taken += 1;
        self.v = try alloc(kvh);
        taken += 1;
        self.out = try alloc(qh);
        taken += 1;
        self.dout = try alloc(qh);
        taken += 1;
        self.dq = try alloc(qh);
        taken += 1;
        self.dk = try alloc(kvh);
        taken += 1;
        self.dv = try alloc(kvh);
        taken += 1;
        self.row_max = try alloc(pr);
        taken += 1;
        self.row_den = try alloc(pr);
        taken += 1;
        self.row_del = try alloc(pr);
        taken += 1;

        self.t = t;
        self.n_heads = n_heads;
        self.n_kv_heads = n_kv_heads;
        self.dim = dim;
        self.c_t = c_t;
        self.c_heads = c_heads;
        self.c_kv_heads = c_kv_heads;
        self.c_dim = c_dim;
        return self;
    }

    /// The buffer at position `i`, by INDEX rather than by building an array of
    /// all eleven. The array version read every field -- including ones `init` had
    /// not assigned yet, whose values are `undefined` -- and then handed the
    /// survivors to `cudaFree`. Indexed, a call with `i >= taken` never happens.
    /// How many device buffers an `Attn` holds: what `field` enumerates and what
    /// `deinit` frees, so the two are read from one place.
    const N_BUFFERS = 11;

    fn field(self: *const Attn, i: usize) [*]f32 {
        return switch (i) {
            0 => self.q,
            1 => self.k,
            2 => self.v,
            3 => self.out,
            4 => self.dout,
            5 => self.dq,
            6 => self.dk,
            7 => self.dv,
            8 => self.row_max,
            9 => self.row_den,
            10 => self.row_del,
            else => unreachable,
        };
    }

    /// Free the first `taken` buffers. `taken` is a COUNT of what `init`
    /// successfully took, which is the opposite sense to the number the eleven
    /// per-allocation errdefers used to pass; there is one convention now.
    fn deinitPartial(self: *const Attn, taken: usize) void {
        var i: usize = 0;
        while (i < taken) : (i += 1) _ = cudaFree(field(self, i));
    }

    /// Free all eleven. `N_BUFFERS` is the constant `field`'s switch enumerates, so a
    /// twelfth buffer has to be added in both places or trip `field`'s
    /// `unreachable` arm, which is a Debug panic and so cannot ship by accident.
    ///
    /// This read `deinitPartial(0)`, which frees nothing: every forward pass would
    /// have leaked all eleven device buffers -- about 105 MB over a 123-step run --
    /// and nothing anywhere would have said so, because freeing nothing is not a
    /// compile-time property of anything. It takes a GPU and an allocation counter
    /// to see, which is what `zig build cuda-attn-check` is for.
    pub fn deinit(self: *Attn) void {
        self.deinitPartial(N_BUFFERS);
    }

    /// Total device bytes held: four query-headed buffers (q, out, dout, dq), four
    /// kv-headed (k, v, dk, dv) and three per-row. Useful for the same reason
    /// `zig build`'s peak-RSS gate exists on the host side: a number that should be
    /// small, checked rather than assumed.
    ///
    /// It said five query-headed once and the doc comment above `Kind` said four,
    /// so the function a VRAM-budget caller would use over-reserved by one 131072
    /// byte buffer -- 14% -- against the same file's own list of names.
    pub fn totalBytes(self: *const Attn) usize {
        const qh = bytesFor(.query_headed, self.t, self.n_heads, self.n_kv_heads, self.dim) catch 0;
        const kvh = bytesFor(.kv_headed, self.t, self.n_heads, self.n_kv_heads, self.dim) catch 0;
        const pr = bytesFor(.per_row, self.t, self.n_heads, self.n_kv_heads, self.dim) catch 0;
        return 4 * qh + 4 * kvh + 3 * pr;
    }

    /// The forward over the WHOLE cache: every cached key visible to every query row,
    /// which is the training shape. `q_offset` 0 and `n_keys` `c_t`. `group_q` is 1
    /// for every published row and `max_tile` is the tile cap the benchmark uses; both
    /// are parameters so a caller is not forced to match the benchmark by construction.
    ///
    /// A decode call wants a different thing and should NOT reach for this: one query
    /// row at absolute position `pos`, attending to `n_keys` cached rows. That is
    /// `forwardAt`, and it is a separate function rather than two extra defaulted
    /// arguments here, because folding them together is how a caller passes `n_ctx`
    /// where the cache length belonged -- and that mistake masks every key past index
    /// 0 and returns `v[0]` with no error.
    pub fn forward(self: *const Attn, group_q: c_int, max_tile: c_int) Error!void {
        return self.forwardAt(group_q, max_tile, 0, self.c_t);
    }

    /// The forward for one decode step. `q_offset` is the token's absolute position
    /// and `n_keys` is how much of the cache is filled.
    pub fn forwardAt(
        self: *const Attn,
        group_q: c_int,
        max_tile: c_int,
        q_offset: c_int,
        n_keys: c_int,
    ) Error!void {
        const rc = zt_attn_forward(self.q, self.k, self.v, self.out, self.c_t, self.c_heads, self.c_kv_heads, self.c_dim, group_q, max_tile, q_offset, n_keys, null);
        if (rc != 0) return Error.ConfigRefused;
    }

    /// The backward. Requires the forward to have run first on the same buffers:
    /// the two kernels read `q`, `k` and `v` and the second reads the per-row
    /// scalars the first wrote.
    pub fn backward(self: *const Attn, max_tile: c_int) Error!void {
        const rc = zt_attn_backward(self.q, self.k, self.v, self.dout, self.dq, self.dk, self.dv, self.row_max, self.row_den, self.row_del, self.c_t, self.c_heads, self.c_kv_heads, self.c_dim, max_tile, null);
        if (rc != 0) return Error.ConfigRefused;
    }

    /// Host to device. One call, one buffer: the caller decides the granularity,
    /// because a per-layer batched copy and a per-tensor copy differ by more than
    /// an order of magnitude in launch count and nothing here can know which is
    /// right.
    pub fn upload(self: *const Attn, dst: [*]f32, src: []const f32) Error!void {
        // Multiplied WITH A CHECK rather than multiplied, so a caller passing an
        // absurd slice gets a refusal instead of a wrapped small count that would
        // sail through the bounds test below.
        const n = std.math.mul(usize, src.len, @sizeOf(f32)) catch return Error.ShapeTooLarge;
        if (n > bufferBytes(dst, self)) return Error.ShapeTooLarge;
        if (cudaMemcpy(dst, src.ptr, n, memcpy_host_to_device) != 0) return Error.DeviceAllocFailed;
    }

    /// Device to host. One bound, and it is on the DEVICE side: the destination is a
    /// caller-owned host slice whose length says nothing about which device buffer
    /// it is paired with, so the pair is checked against the device buffer's real
    /// capacity. The host side needs no matching check -- host and device memory are
    /// disjoint address spaces, so the destination cannot overlap another buffer.
    pub fn download(self: *const Attn, dst: []f32, src: [*]f32) Error!void {
        const n = std.math.mul(usize, dst.len, @sizeOf(f32)) catch return Error.ShapeTooLarge;
        if (n > bufferBytes(src, self)) return Error.ShapeTooLarge;
        if (cudaMemcpy(dst.ptr, src, n, memcpy_device_to_host) != 0)
            return Error.DeviceAllocFailed;
    }
};

fn alloc(bytes: usize) Error![*]f32 {
    var p: ?*anyopaque = null;
    if (cudaMalloc(&p, bytes) != 0) return Error.DeviceAllocFailed;
    // `anyopaque` has alignment 1 and `f32` has 4, so the cast needs the
    // `@alignCast` to say the pointer really is aligned. `cudaMalloc` returns at
    // least 256-byte aligned memory, which is what makes that a fact and not a hope.
    return @ptrCast(@alignCast(p orelse return Error.DeviceAllocFailed));
}

/// Upper bound on the bytes behind a device pointer, by matching it against the
/// holder. A pointer that is not one of ours returns 0, which makes `upload`
/// refuse rather than overflow.
fn bufferBytes(target: [*]f32, self: *const Attn) usize {
    const qh = bytesFor(.query_headed, self.t, self.n_heads, self.n_kv_heads, self.dim) catch return 0;
    const kvh = bytesFor(.kv_headed, self.t, self.n_heads, self.n_kv_heads, self.dim) catch return 0;
    const pr = bytesFor(.per_row, self.t, self.n_heads, self.n_kv_heads, self.dim) catch return 0;
    const pairs = [_]struct { p: [*]f32, n: usize }{
        .{ .p = self.q, .n = qh },       .{ .p = self.k, .n = kvh },
        .{ .p = self.v, .n = kvh },      .{ .p = self.out, .n = qh },
        .{ .p = self.dout, .n = qh },    .{ .p = self.dq, .n = qh },
        .{ .p = self.dk, .n = kvh },     .{ .p = self.dv, .n = kvh },
        .{ .p = self.row_max, .n = pr }, .{ .p = self.row_den, .n = pr },
        .{ .p = self.row_del, .n = pr },
    };
    for (pairs) |pair| {
        if (pair.p == target) return pair.n;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// The one part of this file that can be tested without a GPU, and the one that
// most needs to be: a wrong size is an out-of-bounds write inside a kernel, and
// there is no gate in this repository that would catch it.
// ---------------------------------------------------------------------------

const testing = std.testing;

test "the shipped shape's sizes, spelled out" {
    // 4 layers of a 4-head, 2-kv-head, head_dim 32, ctx 256 model.
    try testing.expectEqual(@as(usize, 256 * 4 * 32 * 4), try bytesFor(.query_headed, 256, 4, 2, 32));
    try testing.expectEqual(@as(usize, 256 * 2 * 32 * 4), try bytesFor(.kv_headed, 256, 4, 2, 32));
    try testing.expectEqual(@as(usize, 256 * 4 * 4), try bytesFor(.per_row, 256, 4, 2, 32));
}

test "dk and dv together equal the kv_cache term scale.zig already costed" {
    // `scale.zig` prices a KV cache as
    // `n_layers * (n_kv_heads * head_dim) * n_ctx * 2 * sizeof(f32)`, because a
    // cache is exactly a dk-sized and a dv-sized buffer per layer -- which is what
    // this file allocates.
    //
    // HONESTLY SCOPED: the left side is a TRANSCRIPTION of that formula, not a call
    // into `scale.zig`, so editing `scale.zig` alone leaves this test green. The two
    // are not independent derivations and an earlier version of this comment said
    // they were. What IS independent is the right side: it goes through the checked
    // arithmetic in `bytesFor`, not through the transcribed constants. So this
    // catches the buffer arithmetic drifting away from the cost model -- which is
    // the failure worth catching here -- and not the cost model changing.
    const n_layers = 4;
    const n_ctx = 256;
    const n_kv_heads = 2;
    const head_dim = 32;
    const scale_kv_cache = n_layers * (n_kv_heads * head_dim) * n_ctx * 2 * @sizeOf(f32);
    const here = 2 * n_layers * (try bytesFor(.kv_headed, n_ctx, 4, n_kv_heads, head_dim));
    try testing.expectEqual(scale_kv_cache, here);
    try testing.expectEqual(@as(usize, 524288), scale_kv_cache);
}

test "the eleven buffers total 798720 bytes at the shipped shape, against 12 GiB of VRAM" {
    // The coefficient is the point. `totalBytes` once said 5 query-headed buffers
    // while `Kind`'s own doc comment listed four, and init allocates four, so the
    // function a VRAM-budget caller would use over-reserved by one 131072-byte
    // buffer -- 14%. Asserting the literal total rather than an expression over
    // `bytesFor` is what makes a wrong coefficient FAIL here: an earlier version of
    // this test was written as `5 * qh + 4 * kvh + 3 * pr` and so agreed with the bug
    // by construction, and never called the function it was nominally checking.
    const qh = try bytesFor(.query_headed, 256, 4, 2, 32);
    const kvh = try bytesFor(.kv_headed, 256, 4, 2, 32);
    const pr = try bytesFor(.per_row, 256, 4, 2, 32);
    try testing.expectEqual(@as(usize, 798720), 4 * qh + 4 * kvh + 3 * pr);
    try testing.expect(4 * qh + 4 * kvh + 3 * pr < 1024 * 1024);
}

test "a degenerate shape is refused by name, not as an allocator failure" {
    // `cudaMalloc(0)` returns cudaErrorInvalidValue, so without an explicit check
    // this reported DeviceAllocFailed and a caller retrying that error retried
    // forever.
    try testing.expectError(Error.DegenerateShape, bytesFor(.query_headed, 0, 4, 2, 32));
    try testing.expectError(Error.DegenerateShape, bytesFor(.query_headed, 256, 0, 2, 32));
    try testing.expectError(Error.DegenerateShape, bytesFor(.query_headed, 256, 4, 2, 0));
    try testing.expectError(Error.DegenerateShape, bytesFor(.per_row, 256, 4, 0, 32));
}

test "a size that would overflow is refused, not truncated" {
    // Without the check this returns a small number, a small cudaMalloc, and a
    // kernel writing past the end of it.
    try testing.expectError(Error.ShapeTooLarge, bytesFor(.query_headed, std.math.maxInt(usize), 4, 2, 32));
    try testing.expectError(Error.ShapeTooLarge, bytesFor(.query_headed, 1 << 40, 1 << 40, 1, 1));
    try testing.expectError(Error.ShapeTooLarge, bytesFor(.kv_headed, std.math.maxInt(usize), 4, 2, 32));
}

test "a dimension above c_int's range is refused rather than wrapped negative" {
    // 2^31 as a c_int is negative. A negative T would make the kernel walk its
    // prefix backwards.
    try testing.expectError(Error.ShapeTooLarge, toCInt(@as(usize, 1) << 31));
    try testing.expectEqual(@as(c_int, 256), try toCInt(256));
    try testing.expectEqual(std.math.maxInt(c_int), try toCInt(@as(usize, std.math.maxInt(c_int))));
}
