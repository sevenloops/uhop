# Test pruning candidates (diagnostic/heavy)

These tests are valuable for diagnostics and development but should be gated or excluded from default CI runs unless explicitly enabled.

Environment toggles suggested below are already wired (or proposed) in the test files.

## OpenCL tiled debug diagnostics

- `tests/test_opencl_matmul_tiled.py`
  - Heavy debug kernels and per-tile dumps
  - Gated via `UHOP_RUN_DEBUG_TILED=1`
  - Keeps a small correctness check always-on (`test_tiled_validation_fallback_correct`)

## OpenCL performance sweeps

- `tests/test_opencl_tiles.py`
  - Perf/diagnostic sweep across tile sizes
  - Gated via `UHOP_RUN_PERF_TESTS=1`

## Protocol handshake and negative cases (Node + websocket)

- `tests/test_protocol_handshake.py`
- `tests/test_protocol_negative.py`
  - Skips when `node` or `websocket-client` is unavailable
  - Consider moving to a separate workflow job to reduce flakiness

## GPU vendor-specific

- `tests/test_nvidia_kernels.py` (if present)
  - Skip unless CUDA is available; avoid in CPU-only CI

---

CI Recommendation:

- Default job runs unit and small integration tests only.
- Add an optional job (or a nightly) enabling:
  - `UHOP_RUN_DEBUG_TILED=1` (OpenCL device present)
  - `UHOP_RUN_PERF_TESTS=1`
  - websocket + Node handshake tests
