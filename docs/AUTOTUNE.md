# Autotune metadata (OpenCL)

This document describes how UHOP records per-device and per-shape tuning data for OpenCL kernels.

## Key concepts

- Backend: `opencl`
- Op: `matmul`, `conv2d` (others in future)
- Kernel: logical implementation name (e.g., `matmul_tiled`, `conv2d_tiled`, `clblast/sgemm`)
- Device: OpenCL device name (string)
- Shape key: normalized shape string (e.g., `M512_K256_N512` for matmul)

## Storage

Autotune data is persisted to a JSON file (typically at `~/.uhop_mvp_cache/autotune.json`).

Each record key format:

```
backend|op|kernel|device_name|shape_key
```

Value is a dict with some of the following fields:

- `lsz`: Local work-group size, e.g., `[16, 16]` or `[16, 16, 1]`
- `tile`: Tiling parameter used for kernels (e.g., 8, 16, 32)
- `vec`: Vector width used (1, 2, 4, 8)
- `unstable`: If true, kernel/device combination was observed unstable and is avoided
- `history`: Rolling recent performance entries, each:
  - `ts`: ISO timestamp
  - `gflops`: Achieved throughput (float)
  - `ms`: Measured kernel time in milliseconds (float)

## Lifecycle

1. Candidate selection: Generate tile/vec/local-size candidates.
2. Pre-validation (optional): Small synthetic problem to catch indexing bugs.
3. Timing: Measure event time or wall time fallback.
4. Selection: Persist best `lsz` and parameters. Record a profiling entry in `history`.
5. Validation (optional, recommended): Compare output vs reference; if fails, mark `unstable` and fallback to a safer implementation.

## CLI & inspection

- List all cache entries:

```
uhop cache list
```

- Show specific key:

```
uhop cache show "opencl|matmul|matmul_tiled|<device>|M256_K256_N256"
```

- Clear CLBlast device-level `unstable` flags:

```
uhop cache autotune clear-clblast-unstable [device_substring] [--exact]
```

## Notes

- UHOP may periodically re-tune parameters if variance in `history` is high.
- Validation thresholds are configurable; see `ENV_VARS.md` for `UHOP_OPENCL_VALIDATE` and diagnostic toggles.
