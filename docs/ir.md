# UHOP Intermediate Representation (IR)

The UHOP IR is a minimal, extensible schema used to describe operator
implementations independent of any specific backend.  The schema is designed
to be serialized as canonical JSON so that artifacts (generated kernels,
autotune records, telemetry) can be keyed by a stable hash.

## Versioning

- Current version: `0.1.0`
- Version lives in `uhop/ir/ir.py` as `IR_VERSION`
- Serialized IR dictionaries include the `version` field for every op and
  derived artifacts use the version when computing cache keys
- Backwards-incompatible changes must bump `IR_VERSION`; consumers should offer
  migration hooks when older versions are encountered

## Core datatypes

### Tensor

| Field        | Type               | Notes                                                     |
|--------------|--------------------|-----------------------------------------------------------|
| `name`       | `str`              | Logical identifier used in prompts / debugging            |
| `shape`      | `Tuple[int, ...]`  | Concrete extents (no symbolic dims yet)                   |
| `dtype`      | `str`              | e.g. `"f32"`, `"f16"`, `"i32"`                           |
| `layout`     | `str`              | One of `row_major`, `col_major`, `nchw`, `nhwc`, …         |
| `strides`    | `Tuple[int, ...]`? | Optional explicit strides when non-contiguous             |
| `memory_space` | `str`            | `global`, `local`, `private`, `constant`                  |

### Schedule

Optional execution hints that backend lowerings can interpret.  All fields are
optional integers:

- `tile_m`, `tile_n`, `tile_k`
- `vectorize`
- `unroll`

### Ops

Currently supported ops (in `uhop/ir/ir.py`):

- `MatMul`
- `FusedMatMulRelu`
- `Relu`

Each op embeds the relevant tensors (`A`, `B`, `C`, …) and an optional
`schedule` block.  Ops expose `to_dict()` / `from_dict()` helpers and
`infer_output()` methods so callers do not have to rebuild shape logic.

## Stable hashing

IR dictionaries are serialized with `json.dumps(..., sort_keys=True)` and fed
into SHA-256 via `compute_stable_hash`.  This guarantees:

- Key order does not matter (dictionaries can be reordered safely)
- Numeric values are normalized (we stringify ints/floats before hashing)
- The resulting 64-character hex digest can be used anywhere a cache key is
  expected (kernel registry, dataset records, etc.)

Any semantic change — including schedule tweaks — produces a different hash.

## Lowering to OpenCL

`uhop/ir/opencl_lowering.py` implements the first backend lowering.  It accepts
an IR op instance and returns a dictionary with:

- `language`: currently always `"opencl"`
- `kernel_name`: canonical kernel symbol (`uhop_matmul`, `uhop_relu`, …)
- `source`: OpenCL C source text
- Optional metadata such as `tile`/`vec` extracted from the schedule

The lowering currently emits a correctness-first kernel (no tiling yet).  The
new OpenCL execution path in `uhop/backends/opencl/matmul.py` now accepts
an IR op and will:

1. Compute the IR hash key
2. Reuse the shared OpenCL program cache (`_get_program`)
3. Execute the lowered kernel with the provided host tensors
4. Persist the IR→binary mapping via `IRKernelIndex`

This wiring makes it possible for higher level tooling (CLI, agent, AI
pipeline) to run IR specs directly through the production backend without
re-implementing launch code.

## Usage examples

Minimal round-trip:

```python
from uhop.ir import MatMul, Tensor, Schedule
from uhop.ir.opencl_lowering import lower_to_opencl

op = MatMul(
    A=Tensor("A", (128, 256)),
    B=Tensor("B", (256, 64)),
    schedule=Schedule(tile_m=16, tile_n=16, vectorize=4),
)

ir_dict = op.to_dict()
ir_hash = compute_stable_hash(ir_dict)
opencl_kernel = lower_to_opencl(op)
```

Executing via the OpenCL backend:

```python
from uhop.backends.opencl.matmul import MatmulOp

result = MatmulOp().execute(host_A, host_B, ir=op)
print(result.impl)  # "ir"
print(result.output)
```

The same IR descriptor can later be stored, hashed, and replayed through the
AI generation pipeline or other backends as they come online.
