# IR MVP (v0)

This document describes the minimal Intermediate Representation (IR) introduced in this MVP to enable simple kernel lowering without pulling in a heavyweight compiler stack.

Goals:
- Represent a tiny set of ops (MatMul, Relu, FusedMatMulRelu)
- Carry optional schedule hints (tile sizes, vectorization) for future tuning
- Serialize to/from JSON for transport across the Agent/API
- Lower to OpenCL C for execution (single-pass kernels)

## Data model

- Tensor
  - name: string
  - shape: int[]
  - dtype: string ("f32")
- Ops
  - MatMul: inputs A (M×K), B (K×N), output C (M×N)
  - Relu: input X, output Y (same shape)
  - FusedMatMulRelu: inputs A, B, output Y (M×N)
- Schedule (optional)
  - tile_m, tile_n, tile_k: ints
  - vectorize: int

Python API: `uhop.ir`

```python
from uhop.ir import Tensor, MatMul, Relu, FusedMatMulRelu, Schedule, ir_from_dict

mm = MatMul(A=Tensor("A", (64, 128)), B=Tensor("B", (128, 32)), schedule=Schedule(tile_m=8))
d = mm.to_dict()      # JSON-serializable dict
op = ir_from_dict(d)  # back to object
```

## Lowering to OpenCL

Module: `uhop.ir.opencl_lowering`

`lower_to_opencl(op)` returns a dict with keys:
- `language`: always `opencl` for the MVP
- `kernel_name`: symbol to launch
- `source`: generated OpenCL C
- `tile`: chosen tile size (normalized to 8,16,32 if provided)
- `vec`: chosen vectorization factor (1,2,4,8)

### MatMul / FusedMatMulRelu

Now lowered as a tiled kernel with local memory blocks:
- Workgroup local size: `(tile, tile)`
- Global size rounded up to tile multiples for M,N
- Inner loop over K split into blocks of `tile`; guarded loads into `__local` arrays
- Optional vectorized (float4) global→local load for B when `vec >= 4` and aligned; safe scalar fallback otherwise
- A simple `#pragma unroll VEC` hint used for loop unrolling

### Relu

Elementwise, still simple; returns metadata for consistency. Global size is rounded to a multiple of a fixed tile (64).

### Schedule Influence

From `Schedule`:
- `tile_m/tile_n/tile_k`: first non-null is used to pick the tile; normalized to common sizes
- `vectorize`: sets `vec`; guarded path for float4 loads when `vec=4`

Future additions: true vectorized accumulator math, per-axis tiling, unroll factors per axis, shared mem double-buffering.

## Agent integration

`agent.compile_kernel` now accepts IR descriptors via:

```json
{ "ir": { "type": "matmul", "A": {"name":"A","shape":[64,128],"dtype":"f32"}, "B": {"name":"B","shape":[128,32],"dtype":"f32"} } }
```

The Agent lowers IR to OpenCL and builds it. `validate` likewise accepts an `ir` block and will compile + execute on random inputs, comparing against NumPy references.

## Validation & Multi-shape Testing

The agent `validate` action supports:
```json
{ "ir": { ... }, "tolerance": 1e-4 }
```
Single shape inferred from IR tensor shapes.

For multiple shapes:
```json
{ "ir": { ... }, "shape_sets": [ { "A": [64,128], "B": [128,64] }, { "A": [96,64], "B": [64,32] } ] }
```
Returns `validated.multi` with pass/fail per shape.

## IR CLI Quickstart

```bash
# Lower with schedule hints
python -m uhop.cli_ir lower --file matmul_ir.json --tile 16 --vec 4 --out matmul.cl

# Build & get artifact (includes ir_key)
python -m uhop.cli_ir build --file matmul_ir.json --tile 16 --vec 4

# Validate multiple shapes
python -m uhop.cli_ir validate --file matmul_ir.json --tile 16 --vec 4 \
  --shape-set "A=64x128,B=128x64" --shape-set "A=96x64,B=64x32"

# Benchmark vectorization impact
python -m uhop.cli_ir bench --file matmul_ir.json --shape "A=256x512,B=512x256" --tile 16
```

## Tests

- IR roundtrip serialize/deserialize
- Lowered matmul correctness vs NumPy (small shapes)
- Lowered fused matmul+relu correctness vs separate matmul + relu
- Agent compile+validate with IR descriptor (single and multi-shape)

OpenCL-dependent tests auto-skip when PyOpenCL or a device is not available.

## Reusing built artifacts via IRKernelIndex

When compiling IR-based kernels, the agent records a persistent mapping from the normalized IR key (`ir_key`) and device name to the generated source hash, kernel name, and (optionally) the saved binary path. You can check for a cache hit and skip rebuilding like this:

```python
from uhop.ir.registry import IRKernelIndex, compute_ir_key

# 1) Compute a stable IR key (order/whitespace-insensitive)
ir_key = compute_ir_key(ir_desc)  # ir_desc is your IR as a Python dict

# 2) Look up for the current device
device_name = "AMD Radeon ..."  # match what your runtime reports
entry = IRKernelIndex().get(ir_key, device_name)

if entry:
  print("Reuse kernel:", entry.get("kernel_name"), "binary:", entry.get("binary_path"))
  # You can launch using the known kernel_name after loading the program,
  # or rely on your runtime to pick up the saved binary if supported.
else:
  # Build once (e.g., via agent compile_kernel or IR CLI). On success the
  # mapping is persisted and future runs on the same device can reuse it.
  pass
```

Notes:
- `binary_path` may be null if the OpenCL driver doesn't expose binaries; the mapping is still useful to detect that the exact source/options were already built.
- The agent attaches `ir_key` to the returned compile artifact for easy indexing from higher-level coordinators.
