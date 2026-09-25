# Z-TRANSFORMER

A transformer built from scratch in Zig and CUDA: RMSNorm, RoPE, grouped-query
attention, SwiGLU, and a fused IO-aware attention kernel. The goal is a
Llama-3-class block that matches a external reference within tolerance, and a
GPT-mini that trains end to end under a measured loss curve.

The project is under active construction. No benchmark number is published here
yet, because none has been measured.

Requires Zig 0.16.0. There is nothing to install and no package manager step.

```sh
zig build        # build the ztransformer binary
zig build run    # run it
zig build test   # run the tests
```

Source lives in `src/`. Generated benchmark and parity artifacts land in
`outputs/`. MIT licensed.
