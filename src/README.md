# src

| Symbol | Kind | Signature | Purpose |
|---|---|---|---|
| `version` | const | `pub const version: []const u8` | Release string the binary reports. |
| `name` | fn | `pub fn name() []const u8` | Returns the project name, `Z-TRANSFORMER`. |
| `tensor.Tensor` | struct | `pub const Tensor` | Dense row-major `f32` matrix: `data: []f32` of length `rows * cols`, plus `rows`, `cols`, `allocator`. Owns its buffer. |
| `Tensor.init` | fn | `pub fn init(allocator, rows, cols) !Tensor` | Allocates and zeroes `rows * cols` elements. Propagates the allocation error. |
| `Tensor.deinit` | fn | `pub fn deinit(self: *Tensor) void` | Frees the buffer. Idempotent. |
| `Tensor.at` | fn | `pub fn at(self: Tensor, r, c) f32` | Reads one element. Panics if `r` or `c` is out of range. |
| `Tensor.set` | fn | `pub fn set(self: *Tensor, r, c, v: f32) void` | Writes one element. Panics if `r` or `c` is out of range. |
| `Tensor.row` | fn | `pub fn row(self: *Tensor, r) []f32` | Mutable view of row `r`, exactly `cols` elements. |
| `Tensor.rowConst` | fn | `pub fn rowConst(self: Tensor, r) []const f32` | Immutable view of row `r`, exactly `cols` elements. |
| `Tensor.fill` | fn | `pub fn fill(self: *Tensor, v: f32) void` | Writes `v` to every element. |
| `matmul` | fn | `pub fn matmul(a: Tensor, b: Tensor) !Tensor` | `[m,k] @ [k,n] -> [m,n]`. Returns `error.DimensionMismatch` when `a.cols != b.rows`. |
| `mlp.forward` | fn | `pub fn forward(allocator, x, w_gate, w_up, w_down: Tensor) !Tensor` | SwiGLU feed-forward. `x` is `[T, d]`, `w_gate` and `w_up` are `[d, h]`, `w_down` is `[h, d]`, out is `[T, d]`. Returns `error.DimensionMismatch` on any shape disagreement. Frees its intermediates on every path. |
| `mlp.silu` | fn | `pub fn silu(z: f32) f32` | `z * sigmoid(z)`, with the exponent negated below zero so the negative tail stays finite instead of collapsing to `-0.0`. |

`lib.zig` is the library root and the only public surface. `main.zig` is the
executable, and `tests.zig` holds the tests. Both import `lib.zig` as
`@import("ztransformer")`, so the module boundary is the same one the test suite
crosses. Later phases add numerics modules to `lib.zig`.

`tensor.zig` is imported by relative path, `@import("tensor.zig")`, from a
sibling file. It is 2D `f32` only, and deliberately has no strides, views into
other tensors, broadcasting, or transpose, so the index arithmetic in every
downstream op stays checkable. `at` and `set` panic on an out-of-range index
rather than read neighbouring memory; the contract keeps them total, so the
check is a precondition violation, not an error return.

`mlp.zig` is the Llama-3 feed-forward, `silu(gate) * up` projected down. It
takes no locks, spawns no threads, and leaves every reduction in the order
`matmul` fixes, so repeated runs are bit-identical. Its tests live in
`mlp_test.zig`.
