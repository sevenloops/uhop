# UHOP Protocol v0.1 (Draft)

Goals:
- Stable, versioned messages between UI/backend and local agents.
- Portable device capability reporting and kernel metadata exchange.
- Support remote compilation, benchmarking, validation, and cache inspection.

Transport:
- WebSocket primary (bi‑directional logs and requests), HTTP optional for simple GET/POST.
- All messages are JSON objects with top‑level fields: `type`, `id` (optional), and `v` (protocol version string).

Reference implementation helpers live in `uhop/policy.py` (selection) and `tests/test_protocol_handshake.py` (mock agent handshake). Schema validators and response builders are implemented in `uhop/protocol.py`.

Versioning:
- `v`: "0.1" for this draft. Backward‑compatible changes bump patch; breaking changes bump minor.

---

## Message types

### Agent hello
- Direction: Agent → Server
- Purpose: authenticate and advertise basic info

Example:
```json
{ "v": "0.1", "type": "hello", "agent": "uhop-agent", "version": "0.1.0", "token": "<optional>" }
```

### Request/Response envelope
- Direction: Server ↔ Agent
- Pattern: request has `id`; response echoes `id` and sets `ok` with `data` or `error`.

Request:
```json
{ "v": "0.1", "type": "request", "id": "abc123", "action": "info", "params": {} }
```

Response:
```json
{ "v": "0.1", "type": "response", "id": "abc123", "ok": true, "data": { /* action-specific */ } }
```

### Log streaming
```json
{ "v": "0.1", "type": "log", "level": "info", "line": "[opencl] tuned TILE=16 vec=4" }
```

---

## Actions

### info
- Return device capabilities and backend availability.

Data schema:
```json
{
  "kind": "cuda|opencl|opencl-cpu|metal|cpu",
  "vendor": "nvidia|amd|intel|apple|generic",
  "name": "GeForce RTX ...",
  "details": {
    "platform": "CUDA 12.2 / OpenCL 3.0 ...",
    "device_version": "<driver string>",
    "torch": "2.3.1",
    "opencl": "2024.2",
    "triton": "2.2.0"
  },
  "backends": {
    "torch": { "available": true, "accelerator": true },
    "triton": { "available": true },
    "opencl": { "available": true, "devices": ["AMD Radeon ..."], "default_index": 0 },
    "vulkan": { "available": false }
  }
}
```

### benchmark
- Compile (if needed) and time an op or kernel candidate, optionally autotuning a parameter grid.

Request params:
```json
{
  "op": "matmul",
  "backend": "opencl|triton|torch|vulkan",
  "shapes": { "A": [M,K], "B": [K,N] },
  "dtype": "float32",
  "schedule": {
    "tile": 16,
    "vec": 4,
    "local": [16,16],
    "flip_gws": false
  },
  "runs": 5,
  "validate": true
}
```

Response data:
```json
{
  "ok": true,
  "stats": { "mean_ms": 0.42, "std_ms": 0.02, "gflops": 350.1 },
  "validated": { "max_abs_err": 2.3e-6, "passed": true },
  "compiled": { "took_ms": 120, "cache_hit": true }
}
```

### compile_kernel
- Build and persist a kernel for a given IR or source+schedule. (Planned server-side action; current Node backend stubs validation only.)

Request params:
```json
{
  "descriptor": {
    "op": "matmul",
    "ir": null,
    "source": { "lang": "opencl", "text": "__kernel void ..." },
    "signature": { "inputs": [ ["A", ["M","K"], "f32"], ["B", ["K","N"], "f32"] ], "outputs": [["C", ["M","N"], "f32"]] },
    "layout": { "A": "row_major", "B": "row_major", "C": "row_major" },
    "schedule": { "tile": 16, "vec": 4, "local": [16,16] },
    "attrs": { "fused": false }
  }
}
```

Response data:
```json
{
  "artifact": {
    "id": "opencl:sha1:<source+opts>",
    "device": "AMD Radeon ...",
    "kernel_name": "uhop_matmul_t16_v4",
    "binary_path": "/home/.../.uhop_mvp_cache/ocl_<hash>.bin",
    "compiler_opts": "-D TILE=16 -D VEC=4",
    "ir_key": "sha1:<normalized_ir_json>",
    "metadata": {
      "driver": "amdgpu 23.10",
      "created_at": 1731206400
    }
  }
}
```

### validate
- Run a kernel and compare to a reference baseline within tolerance. (Planned; initial benchmarking exists for matmul.)

Params:
```json
{ "op": "matmul", "shapes": {"A": [M,K], "B": [K,N]}, "tolerance": 1e-4 }
```

IR path (single or multi-shape):
```json
{ "ir": { /* IR descriptor */ }, "tolerance": 1e-4 }
```

Multi-shape IR validation:
```json
{ "ir": { /* IR descriptor */ }, "shape_sets": [ {"A": [64,128], "B": [128,64]}, {"A": [96,64], "B": [64,32]} ], "tolerance": 1e-4 }
```

### cache_list / cache_set / cache_delete
- Introspect and update runtime selection cache and autotune store. (Cache schema validated in `protocol.py`.)

---

## Schemas

### DeviceCapability
```json
{
  "kind": "cuda|opencl|metal|cpu|vulkan",
  "vendor": "nvidia|amd|intel|apple|generic",
  "name": "string",
  "driver": { "version": "string", "extra": {} },
  "limits": {
    "max_workgroup": 1024,
    "local_mem_kb": 64,
    "warp_wave": 32,
    "shared_mem_kb": 100
  }
}
```

### KernelDescriptor
```json
{
  "op": "string",
  "signature": { "inputs": [[name, shape_syms, dtype]...], "outputs": [...] },
  "layout": { "<tensor>": "row_major|col_major|NCHW|NHWC|blocked(n)" },
  "schedule": { "tile": 16, "vec": 4, "local": [16,16,1], "unroll": 1, "prefetch": 0, "flip_gws": false },
  "resources": { "registers": null, "smem_bytes": null },
  "ir": null | { "dialect": "uhop-ir", "version": "0.1", "text": "..." },
  "source": { "lang": "opencl|cuda|triton|metal|vulkan", "text": "..." }
}
```

### AutotuneRecord
```json
{
  "key": "backend|op|kernel|device|shape",
  "best": { "tile": 16, "vec": 4, "local": [16,16] },
  "history": [ { "ts": 1731206400, "gflops": 350.1, "ms": 0.42 } ],
  "unstable": false,
  "last_validated": 1731206455
}
```

### CacheManifest
```json
{
  "selection": { "matmul|sig": { "backend": "opencl", "source": "tiled", "device_hint": "gpu|amd|...", "_cached_at": "2025-11-11T00:00:00Z" } },
  "binaries": [ { "backend": "opencl", "device": "AMD ...", "driver": "23.10", "path": "~/.uhop_mvp_cache/ocl_x.bin" } ]
}
```

---

## Security and isolation
- Timeouts, memory limits for sandboxed runs (Python/AI‑generated code).
- Do not trust agent inputs; validate schemas; sanitize file paths; explicit allow‑list of actions.

---

## Extensibility
- Additive fields are allowed under `metadata` and `attrs`.
- New backends add entries under `backends` capability with a minimal spec (available, versions, limits).
