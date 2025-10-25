# UHOP: Windows + AMD GPU Run Report (updated Oct 14, 2025)

This document captures the issues we hit, how we fixed them, and the concrete outputs/metrics from examples and tests on this system.

- Host OS: Windows
- Python: 3.12 (venv at `.venv/`)
- GPU: AMD Radeon (OpenCL platform: AMD Accelerated Parallel Processing, device: `gfx90c [GPU]`)

## Errors encountered and how we solved them

1) Pip install failed with deleteme errors (Windows)
- Error: `ERROR: Could not install packages due to an OSError: [WinError 2] ... 'C:\\Python312\\Scripts\\cpuinfo.exe' -> '...cpuinfo.exe.deleteme'`
- Cause: Installing into global Python Scripts on Windows; file locks/permissions.
- Fix:
  - Created local venv (`python -m venv .venv`) and activated it.
  - Upgraded packaging tools inside venv: `pip install -U pip setuptools wheel`.
  - Installed deps inside venv: `pip install -r requirements.txt --use-pep517`.

2) Missing module for tests: `uhop.kernels.numpy_kernels`
- Error during tests: import error from `tests/test_cpu_flow.py`.
- Fix: Added `uhop/kernels/numpy_kernels.py` with a simple `numpy_matmul` implementation.

3) PyTorch missing for examples
- Error: `ModuleNotFoundError: No module named 'torch'` when running training demos.
- Fix: Installed CPU-only torch in venv: `pip install torch --index-url https://download.pytorch.org/whl/cpu`.

4) Python path import issues for examples
- Error: `No module named 'uhop'` when running scripts with system Python.
- Fix: Used venv Python and/or `PYTHONPATH=.` to ensure repo is importable.

5) Autograd backward error in UHOP conv2d wrapper
- Error: `RuntimeError: element 0 of tensors does not require grad ...`.
- Cause: Using nested backward/grad on tensors that didn’t track grad.
- Fix: Implemented backward with `torch.nn.grad.conv2d_input` and `conv2d_weight` to compute gradients directly, no new graph.

6) OpenCL not selected / interactive context issues
- Problem: OpenCL context selection was interactive or not GPU-first.
- Fixes:
  - Explicitly select a GPU device programmatically (first GPU found) before fallback.
  - `examples/devtools/show_devices.py` added to display platforms/devices.

7) OpenCL kernel overhead and warnings
- Warnings: `RepeatedKernelRetrieval` due to repeated kernel lookups.
- Fix: Cache compiled program and use `cl.Kernel(prg, name)` once; reuse kernel objects.

8) OpenCL INVALID_WORK_GROUP_SIZE
- Error in persistent matmul demo when using BLOCK=32.
- Fix: Reduced workgroup tile to BLOCK=16 for safer defaults on AMD; added autotuner to try [8,16] and cache best per device.

9) Indentation/syntax regressions while iterating OpenCL backend
- Several `IndentationError` and unexpected indent issues while editing.
- Fix: Cleaned up `opencl_backend.py` indentation, kept all kernel launch logic inside function scope, and validated with runs.

- OpenAI SDK mismatch and env loading

  - Issue: Generator used legacy OpenAI SDK usage; API key not loaded when running via CLI.
  - Fixes:
    - Migrated generator to OpenAI Python SDK v1 (chat.completions).
    - CLI auto-loads .env (from CWD or repo root) to set OPENAI_API_KEY and related env.

- AI code extraction unreliable for OpenCL

  - Issue: Some completions wrapped code in ```opencl fences; our extractor didn’t recognize them.
  - Fix: Updated extractor to recognize ```opencl code blocks in addition to cuda/c/py.

- Smoke-test gaps and missing reproducibility

  - Issue: Only matmul had a simple smoke test; no manifest captured model/prompt/selection.
  - Fixes:
    - Added smoke tests for OpenCL ReLU (1D) and Conv2D (NCHW) with reference checks.
    - Wrote a manifest JSON (ai_\<op\>_manifest.json) logging: operation, target, model, prompt, created_at, per-candidate Linf/time, and the selected best.

- Optimizer didn’t reuse AI-generated kernels

  - Issue: After selecting a best candidate, subsequent runs didn’t benefit automatically.
  - Fix: CLI now auto-caches the best OpenCL matmul candidate into ~/.uhop_mvp_cache/index.json; optimizer recognizes backend=ai_opencl and runs the saved kernel.

## Backend and policy improvements

- GPU-first dispatch policy: Optimizer now prefers GPU backends by order:
  1) Triton (if available)
  2) OpenCL (AMD/Intel GPUs)
  3) Torch with accelerator (CUDA/MPS)
  4) Torch CPU fallback

- Torch conv2d fast path in autograd wrapper: Forward now uses `torch.nn.functional.conv2d` directly (fallback to naive NumPy only if torch path fails).

- OpenCL backend:
  - Tiled matmul with local memory (BLOCK tuned via autotuner per device).
  - Program and kernel object caching.
  - GPU device selection without interactive prompts.
  - ReLU kernel and a fused Conv2D+ReLU path via im2col + matmul + relu on OpenCL.
  - Fused Conv2D+ReLU CLI demo available; larger shapes recommended for clearer GPU wins.

## AI code generation pipeline (new)

- CLI: `python -m uhop.cli ai-generate <op> --target opencl --validate --smoke --samples N`
  - Supports matmul, relu, conv2d (OpenCL).
  - Compiles OpenCL sources and runs smoke tests with Linf/time reporting.
  - Writes a manifest `uhop/generated_kernels/ai_<op>_manifest.json` with prompt and selection.
  - For matmul: best candidate is auto-cached for the optimizer as backend=ai_opencl.

### Example manifest snippet

```json
{
  "operation": "matmul",
  "target": "opencl",
  "model": "gpt-4o-mini",
  "prompt": "Produce a single OpenCL C kernel ...",
  "created_at": "2025-10-14T15:02:43Z",
  "candidates": [
    {"path": ".../ai_matmul_cand1.cl", "linf": 9.5e-07, "time": 0.00565},
    {"path": ".../ai_matmul_cand2.cl", "linf": 8.1e-07, "time": 0.00352}
  ],
  "selected": ".../ai_matmul_cand2.cl"
}
```

## Example outputs (latest runs)

All runs used venv Python unless otherwise noted.


### 1) UHOP vs naive Python (matmul)

- Command: `PYTHONPATH=. .venv/Scripts/python.exe examples/benchmarks/compare_python_naive_vs_uhop.py`
- Output:
  - UHOP (optimized over naive): ~0.001 s median
  - Naive Python baseline: ~10.877 s median
  - Result: UHOP wins ✅ (orders of magnitude faster)


### 2) UHOP vs baseline (Conv2D parity)

- Command: `PYTHONPATH=. python examples/benchmarks/compare_uhop_vs_baseline.py` (ran earlier with system python; same result with venv)
- Outputs (two runs during iteration):
  - Before forward update: Baseline loss=1.270956, median ~0.0015 s; UHOP loss=1.270956, median ~0.0028 s
  - After forward uses torch.conv2d: Baseline loss=1.270956, median ~0.0015 s; UHOP loss=1.270956, median ~0.0023 s
- Result: Loss parity identical; UHOP wrapper adds small overhead on CPU.


### 3) Training demos (10 steps)

- UHOP training (`examples/training/train_cnn_uhop.py`) losses:
  - 28.425049, 28.576725, 26.715187, 27.565796, 27.236261,
    27.131676, 26.492235, 27.956526, 28.125742, 25.770393
- Baseline training (`examples/train_cnn_baseline.py`) losses:
  - ~1.356974, 1.337990, 1.306233, 1.303511, 1.310603,
    1.293979, 1.348835, 1.328439, 1.302255, 1.308596
- Note: Different initialization schemes originally caused loss scale differences. After aligning init (Kaiming uniform) and using torch conv2d forward, parity holds. The demo above shows values from the earlier run before full alignment.

### 4) GPU-preferred matmul benchmark (`examples/bench_matmul_gpu_pref.py`)

- Results across iterations during tuning:
  - Initial: UHOP ~0.4116 s, NumPy ~0.0088 s (NumPy faster)
  - After caching + BLOCK tweak: UHOP ~0.0501 s, NumPy ~0.0078 s
  - After further fixes: UHOP ~0.0228 s, NumPy ~0.0103 s
- Takeaway: CPU BLAS is strong for matmul; more tuning/persistent buffers are needed to consistently beat it for these sizes.

### 5) GPU ReLU benchmark (`examples/bench_relu_gpu_pref.py`)

- Output:
  - UHOP (OpenCL) ReLU: ~0.5267 s median
  - NumPy CPU ReLU: ~0.1453 s median
- Takeaway: Memory-bound; without persistent residency or fusing, CPU can win on this single-op scenario.

### 6) Device report (`examples/show_devices.py`)

- Output:
  - OpenCL platforms: 1
    - Platform 0: AMD Accelerated Parallel Processing
      - Device 0: gfx90c [GPU]

## Tests summary

- Latest (venv): `5 passed in 4.63s`
  - tests/test_cpu_flow.py (NumPy matmul correctness)
  - tests/test_matmul.py (decorated matmul correctness)
  - tests/test_pytorch_conv_grad.py (UHOP autograd backward matches torch)
  - tests/test_optimizer_backend_policy.py (policy surface + correctness)
  - any prior tests included in repository (as applicable)
- Earlier run (before adding more tests): `2 passed` (initial sanity)

## Recommendations / Next steps

- Optimizer wiring:
  - Extend ai_opencl cache support to ReLU and Conv2D, similar to matmul.
  - Persist selected kernel metadata per device/shape where applicable.
- Performance engineering:
  - Persistent residency: keep inputs on device; chain ops without host copies.
  - Broader autotuning space (BLOCK, vector width, unrolling) with per-shape caching.
  - Overlap transfers with compute via multiple queues.
- AI codegen:
  - Add more targets (CUDA/Triton) with smoke tests and manifests.
  - Generation manifest enhancements: include timings for validation runs and errors if any.
- Demos and docs:
  - Document the new ai-generate flow; add examples for relu/conv2d smoke tests.
  - Include tips on device selection (--ocl-device or UHOP_OPENCL_DEVICE_INDEX).

---

Prepared and verified during iterative development on this machine (Windows, AMD GPU via OpenCL).
