# UHop MVP (developer-integrated)

This repository is an MVP for UHOP — a runtime that:
- Detects hardware (CUDA vs CPU),
- Runs hand-written CUDA kernels (if PyCUDA available),
- Generates CUDA kernels via OpenAI (if configured),
- Benchmarks and caches the best implementation,
- Exposes a decorator `@hop.optimize("matmul")` for easy integration.

## Quickstart

1. Clone project.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   # or at minimum:
   pip install numpy openai
   # Optional for CUDA:
   pip install pycuda

## CLI

After installing this package in your environment, you get a `uhop` command:

- `uhop info` — print detected hardware and backend availability (Torch, Triton, OpenCL).
- `uhop info --json` — machine-readable JSON hardware info (includes backend availability).
- `uhop info --ocl-device 0` — override OpenCL GPU device selection by index (across all platforms).
- `uhop demo --size 192` — run a quick Naive Python vs UHOP-optimized matmul benchmark and show which wins.
- `uhop demo --iters 5 --ocl-device 0` — adjust iterations and explicitly choose an OpenCL GPU.

Environment override: set `UHOP_OPENCL_DEVICE_INDEX=<idx>` to select a default OpenCL device for the session.

## extra demos

- MatMul (UHOP vs naive):
   - `uhop demo --size 192`
   - Expect UHOP to win decisively vs naive Python; increase size for more stress.
- Fused Conv2D+ReLU (OpenCL fused kernel):
   - `python -m uhop.cli demo-conv2d-relu --h 128 --w 128 --c-in 3 --c-out 32 --k 3 --stride 1 --padding 1`
   - Use `--ocl-device` to select a GPU, and try larger shapes (e.g. `--h 224 --w 224 --c-in 16 --c-out 32`) for stronger GPU gains.

See `docs/RUN_REPORT.md` for a summary of errors encountered and solutions applied during development, plus sample benchmark outputs.
