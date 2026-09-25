const std = @import("std");
const lib = @import("ztransformer");

test "version is non-empty" {
    try std.testing.expect(lib.version.len > 0);
}

test "name returns the project name" {
    try std.testing.expectEqualStrings("Z-TRANSFORMER", lib.name());
}

comptime {
    _ = @import("tensor_test.zig");
    _ = @import("norm_test.zig");
    _ = @import("rope_test.zig");
    _ = @import("mlp_test.zig");
    _ = @import("attention_test.zig");
    _ = @import("loss_test.zig");
}

comptime {
    _ = @import("tokenizer_test.zig");
    _ = @import("data_test.zig");
    _ = @import("optim_test.zig");
    _ = @import("model_test.zig");
}
