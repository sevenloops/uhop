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
