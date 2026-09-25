# src

| Symbol | Kind | Signature | Purpose |
|---|---|---|---|
| `version` | const | `pub const version: []const u8` | Release string the binary reports. |
| `name` | fn | `pub fn name() []const u8` | Returns the project name, `Z-TRANSFORMER`. |

`lib.zig` is the library root and the only public surface. `main.zig` is the
executable, and `tests.zig` holds the tests. Both import `lib.zig` as
`@import("ztransformer")`, so the module boundary is the same one the test suite
crosses. Later phases add numerics modules to `lib.zig`.
