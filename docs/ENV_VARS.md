# UHOP Environment Variables

This document lists supported `UHOP_*` environment variables, their purpose, defaults, and accepted values. Use these to tune backends, control validation, and enable diagnostics.

## Optimizer

- UHOP_BACKEND_PREFERENCE (string)
  - Comma-separated backend order (e.g., `opencl,torch,triton,cpu,numpy`).
  - Default: unset (uses built-in order per platform).
- UHOP_STRICT_VALIDATE (bool)
  - Tighten AI-kernel validation & checks. Default: `0`.
- UHOP_FORCE_BASELINE (bool)
  - Force NumPy baseline implementations; bypass optimizer. Default: `0`.
- UHOP_POLICY_MODE (string)
  - Backend policy mode: `order_probe` (pick first viable) or `benchmark` (time candidates, choose fastest). Default: `order_probe`.
- UHOP_POLICY_BENCH_ITERS (int)
  - Iterations per backend when `UHOP_POLICY_MODE=benchmark`. Default: `3`.
- UHOP_POLICY_BENCH_WARMUP (int)
  - Warmup runs per backend before timing in benchmark mode. Default: `1`.
- UHOP_POLICY_BENCH_EARLY (float)
  - Early-exit factor: if first timing of a candidate is slower than current winner \* factor, skip remaining iterations. Default: `2.5`.
- UHOP_CACHE_PER_SHAPE (bool)
  - Cache decisions per input shape/dtype signature. Default: `1` (enabled).
- UHOP_EXPLAIN_ON_CACHE (bool)
  - When a cache record is used, run a one-shot policy probe and log whether the cached backend matches the current suggestion. Default: `0`.

## Process Behavior

- UHOP_LOAD_DOTENV (bool)
  - Load `.env` files on startup. Set `0` to disable for stricter environments. Default: `1`.

## OpenCL

- UHOP_OPENCL_DEVICE_INDEX (int)
  - Select OpenCL device by flattened index. Default: `0`.
- UHOP_OPENCL_MATMUL_IMPL (string)
  - Matmul implementation: `naive|tiled|clblast`. Default: `naive`.
- UHOP_OPENCL_ENABLE_TILED (bool)
  - Permit tiled path when chosen. Default: `0`.
- UHOP_OPENCL_FORCE_NAIVE (bool)
  - Force naive matmul even if tiled requested. Default: `0`.
- UHOP_OPENCL_VEC_CANDIDATES (csv)
  - Candidate vector widths for kernels (e.g., `1,2,4`). Default: `1`.
- UHOP_OPENCL_VALIDATE (bool)
  - Validate tiled kernel outputs vs CPU. Default: `1`.
- UHOP_OPENCL_FLIP_GWS (bool)
  - Experimental flip of global work-size mapping. Default: `0`.
- UHOP_OPENCL_TILED_DIAG (bool)
  - Emit diagnostics during tiled tuning/validation. Default: `0`.
- UHOP_OPENCL_TILED_DIAG_STRICT (bool)
  - Assert raw vs backend tiled parity during diagnostics. Default: `0`.
- UHOP_OPENCL_NAIVE_VEC (int)
  - Override naive matmul vector width at fallback. Default: `1`.

## Experimental Backends

- UHOP_ENABLE_VULKAN_POC (bool)
  - Enable registration of the experimental Vulkan backend stub. If runtime dependencies are missing it is ignored. Default: `0`.

## IR Development / Diagnostics

- UHOP_IR_DUMP (bool)
  - When set, dump lowered IR sources (e.g. generated OpenCL) into the cache directory for inspection. Default: `0`. (planned; may be ignored in current build)
- UHOP_DISABLE_VEC_GUARD (bool)
  - Force-disable vectorized load path even when schedule requests `vec=4`; useful for A/B correctness diagnostics. Default: `0`. (planned; may be ignored in current build)

Notes:

- IR tiling and vectorization are controlled via `Schedule` objects or CLI flags (`--tile`, `--vec`) rather than environment variables.

## AI Codegen

- UHOP_OPENAI_MODEL (string)
  - Default OpenAI model for codegen. Default: `gpt-4o-mini`.
- UHOP_AI_DEBUG (bool)
  - Verbose codegen logging. Default: `0`.

## Logging & Agent

- UHOP_LOG_LEVEL (string)
  - One of `DEBUG, INFO, WARNING, ERROR`. Default: `INFO`.
- UHOP_AGENT_SERVER (string)
  - Local agent WebSocket endpoint. Default: `ws://localhost:8787/agent`.

## Sandbox

- UHOP_SB_CPU_SECS (int)
  - CPU time limit (seconds) for sandboxed Python runner. Default: derived from timeout.
- UHOP_SB_MEM_MB (int)
  - Memory limit (MiB) for sandboxed Python runner. Default: `1024`.
- UHOP_SB_BLOCK_NET (bool)
  - If set, disable network access in sandboxed Python runner. Best-effort via socket monkeypatch. Default: `0`.

Notes:

- Boolean accepts: `1,true,yes,on` (case-insensitive). Anything else is false.
- CSV values are split by comma and trimmed.
- Changes take effect at process start.
