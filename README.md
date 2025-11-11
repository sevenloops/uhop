# UHOP — Universal Hardware Optimization Protocol

[![Deploy Frontend to GitHub Pages](https://github.com/sevenloops/uhop/actions/workflows/deploy-frontend-pages.yml/badge.svg)](https://github.com/sevenloops/uhop/actions/workflows/deploy-frontend-pages.yml)

Live demo: [uhop.dev](https://uhop.dev)

UHOP is an open hardware optimization platform that unifies GPU acceleration across CUDA, ROCm/HIP, Metal, OpenCL, and future architectures. It detects your machine, dispatches to the best backend, can generate kernels with AI, validates them, and caches the fastest path for reuse — so developers can write simple code and run fast everywhere.

Key capabilities today:

- Automatic backend detection: Torch (CUDA/MPS/CPU), OpenCL (GPU/CPU), Triton (Linux), CPU fallback
- Drop‑in acceleration via `@uhop.optimize("op")` decorator (e.g., matmul)
- AI kernel generation (OpenAI) for OpenCL/CUDA/Python/Triton with validation/smoke tests
- On‑disk caching of selected kernels/implementations per device
- Friendly CLI for hardware info, demos, AI codegen, and cache tools
- Optional Local Agent so the web portal can run on your hardware
- Policy modes (`order_probe`, `benchmark`) with explain tooling (`uhop policy explain`) for transparent backend choice
- KPI snapshot + CLI (`python -m uhop.cli_kpi --show`) including per‑backend selection counts, OpenCL matmul GFLOPS, and Conv2D stage timings (im2col / GEMM / copy + chunk stats)
- Minimal IR (MatMul, Relu, FusedMatMulRelu) with OpenCL lowering via Agent (`docs/IR_MVP.md`)
- Torch shim (`uhop.torch_shim`) for transparent accelerated `matmul` / `relu`
- Vulkan backend PoC stub (disabled by default; see `docs/BACKEND_VULKAN_POC.md`)

Vision: a universal, community-driven runtime optimizer that makes high‑performance computing approachable, portable, and fun — across vendors and form factors.

Planned (see `issues/`): multi‑backend benchmarking/policies, correctness suites, distributed training loops for AI‑generated kernels, richer dashboard, and tighter framework integrations (PyTorch/JAX).

---

## Architecture

![UHOP Architecture diagram](docs/architecture.svg)

The platform has four layers working together:

1. Frontend (Vite + React) — live controls, real‑time logs, and benchmarks
2. Backend (Node/Express + ws) — routes jobs to your Local Agent or server runtime
3. Local Agent (Python) — runs UHOP operations on your machine securely
4. UHOP Core (Python) — backends, optimizer, AI codegen/validation, caching

See also: `docs/architecture.svg` (source image) for sharing in blogs/slides.

At a glance, the request flow prefers the Local Agent when connected, and falls back to server‑side execution when not.

---

## Getting Started

Prereqs

- Python 3.10+
- OS: Windows, macOS, or Linux
- Drivers/toolchains as applicable: CUDA (NVIDIA), OpenCL runtime (AMD/Intel/NVIDIA), Apple MPS (macOS)
- Optional: `OPENAI_API_KEY` for AI codegen

Install

```bash
git clone https://github.com/sevenloops/uhop.git
cd uhop
pip install -e .            # install CLI `uhop`
# optional extras
pip install -e .[dev]       # tests & notebooks
pip install -e .[amd]       # ROCm Python tools
pip install -e .[nvidia]    # CuPy for CUDA
```

Verify your setup

```bash
uhop info
uhop info --json
```

Run a demo

```bash
# Matmul: naive Python vs UHOP‑optimized
uhop demo --size 192 --iters 3

# Fused Conv2D+ReLU (OpenCL). Choose device if multiple are present:
uhop demo-conv2d-relu --h 128 --w 128 --c-in 3 --c-out 32 --k 3 --stride 1 --padding 1
uhop demo-conv2d-relu --ocl-device 0
```

Try OpenCL elementwise add vs naive

```bash
python examples/opencl/compare_elementwise_add_opencl_vs_naive.py --size 2000000
```

Integrate in your code

```python
from uhop import optimize

@optimize("matmul")
def my_matmul(a, b):
    # write the simplest correct version — UHOP will dispatch/accelerate
    import numpy as np
    return np.array(a) @ np.array(b)
```

Torch shim usage (optional):

```python
import torch
from uhop.torch_shim import matmul, relu

a = torch.randn(128, 64, device='cuda' if torch.cuda.is_available() else 'cpu')
b = torch.randn(64, 32, device=a.device)
y = matmul(a, b)   # UHOP policy selects backend
z = relu(y)
```

Enable Vulkan PoC backend (stub):

```bash
export UHOP_ENABLE_VULKAN_POC=1
uhop backends list
```

IR CLI (optional):

```bash
# Lower IR JSON to OpenCL C, injecting schedule hints
python -m uhop.cli_ir lower --file path/to/ir.json --tile 16 --vec 4 --out kernel.cl

# Build via agent (returns artifact with device, kernel_name, ir_key)
python -m uhop.cli_ir build --file path/to/ir.json --tile 16 --vec 4

# Validate one or more shapes
python -m uhop.cli_ir validate --file path/to/ir.json --tile 16 --vec 4 \
        --shape-set "A=64x128,B=128x64" \
        --shape-set "A=96x64,B=64x32"

# Quick benchmark: vec=1 vs vec=4 when N is divisible by 4
python -m uhop.cli_ir bench --file path/to/ir.json --shape "A=256x512,B=512x256" --tile 16
```

Notes:

- IR lowering uses schedule hints for tiling (`tile_m`) and a simple guarded vectorization knob (`vectorize`).
- Agent validation accepts `{ "ir": { ... }, "shape_sets": [...] }` for multi-shape testing.

Environment knobs

- See `docs/ENV_VARS.md` for the full catalog, types, and defaults.
  See also `docs/AUTOTUNE.md` for how tuning metadata is stored and how to inspect it.
  Tip: `uhop config list` prints current effective config (incl. defaults).
  Common ones: - `UHOP_OPENCL_DEVICE_INDEX=<idx>` — default OpenCL device override - `UHOP_STRICT_VALIDATE=1` — tighten AI‑kernel validation during codegen - `UHOP_BACKEND_PREFERENCE=opencl,torch,triton,cpu,numpy` — override optimizer backend order (comma‑separated). Examples: `opencl,torch` to force OpenCL first; `torch,cpu` to prefer Torch; `numpy` to force baseline. - `UHOP_OPENCL_MATMUL_IMPL=naive|tiled|clblast` — pick matmul impl (tiled with validation is recommended where available) - `UHOP_OPENCL_CONV_IMPL=auto|tiled|im2col_gemm` — choose Conv2D implementation. `auto` prefers im2col+GEMM on larger shapes (CLBlast required) - `UHOP_OPENCL_VEC_CANDIDATES="1,2,4"` — compile-time vector widths candidates for OpenCL kernels - `UHOP_EXPLAIN_ON_CACHE=1` — when set, every cached backend use triggers a one-shot policy probe; divergences (cached vs current preference-chain suggestion) are logged.

        ### Policy Introspection (New)

        Use the policy explain tool to see how UHOP chooses a backend for a given op + shapes and (optionally) benchmark each candidate:

        ```bash
        uhop policy explain matmul --arg-shape 128x256 --arg-shape 256x64 --iters 3 --warmup 1 --stats
        uhop policy explain conv2d --arg-shape 1x3x64x64 --arg-shape 8x3x3x3 --iters 2 --stats
        uhop policy explain relu --arg-shape 1048576 --iters 5 --stats --json
        ```

        Output shows:

        - Preference order (from `UHOP_BACKEND_PREFERENCE` or default)
        - Each candidate: success/skip, latency (median ms), optional stats (mean/min/max/std)
        - Selected backend and reason (`order_probe` or `forced_baseline`)
        - Per-shape cache key + cached record (if any)
        - Comparison line: cached vs suggested backend, highlighting divergence

        JSON output additionally includes:

        - `env`: relevant environment flags (`UHOP_BACKEND_PREFERENCE`, `UHOP_FORCE_BASELINE`, `UHOP_STRICT_VALIDATE`)
        - `params`: warmup/iteration/stat collection parameters
        - Full stats block (if `--stats` used)

        Set `UHOP_EXPLAIN_ON_CACHE=1` to log a lightweight explain probe when an existing cached backend is reused during normal optimized function calls. This helps identify stale or suboptimal cached choices early.

        #### Policy Modes

        You can choose how UHOP selects a backend:

        - `order_probe` (default): take the first backend in preference order that runs successfully.
        - `benchmark`: time each viable backend (warmup + iterations) and pick the fastest median latency. Configure with env vars:
                - `UHOP_POLICY_MODE=benchmark`
                - `UHOP_POLICY_BENCH_WARMUP=1` (warmup runs per backend)
                - `UHOP_POLICY_BENCH_ITERS=3` (timed iterations)
                - `UHOP_POLICY_BENCH_EARLY=2.5` (early exit factor for skipping slow candidates quickly)

        CLI override example (without changing env):

        ```bash
        uhop policy explain matmul --arg-shape 512x512 --arg-shape 512x512 --iters 3 --warmup 1 --stats --mode benchmark
        ```

        Cache entries now record `policy=order_probe` or `policy=benchmark` for traceability.

        ### KPI Snapshot & CLI (New)

        Capture a point‑in‑time performance + decision summary:

        ```bash
        python -m uhop.cli_kpi --show
        ```

        Output includes:
        - Backend selection counts (distribution of cached decisions)
        - OpenCL matmul rows: shape, kernel variant, latency, GFLOPS
        - OpenCL Conv2D rows: total latency, stage timings (im2col, GEMM, copy), chunking (whether chunked + chunk count), variance, retune suggestion flag

        These metrics feed future visualization and retune heuristics.

---

## Backend maturity

- Most optimized backend today: OpenCL (GPU) — broad op coverage with tuned/tiled kernels and a growing autotuning surface.
- In progress: CUDA (via Torch and AI CUDA), Apple MPS, and ROCm/HIP backends — parity work and optimizations are active.
- Coming later: CPU-optimized paths beyond Torch CPU, and Vulkan/other GPU APIs. Contributions are welcome to accelerate these paths.

---

## CLBlast integration (optional)

UHOP can use CLBlast for GEMM when available, enabling a BLAS-backed matmul and an im2col+GEMM path for Conv2D on OpenCL devices.

- Requirements: CLBlast shared library installed on your system (DLL/SO/Dylib) - Windows: clblast.dll (MSYS2/conda or vendor package) - Linux: libclblast.so (APT/Yum/Pacman or conda-forge) - macOS: libclblast.dylib (Homebrew/conda-forge)
- Discovery: We auto-detect via system library paths. You can set `CLBLAST_LIBRARY` to an absolute path if needed.
- Controls: - `UHOP_OPENCL_MATMUL_IMPL=clblast` — use CLBlast GEMM for matmul - `UHOP_OPENCL_CONV_IMPL=im2col_gemm` — use im2col+GEMM for Conv2D (per-batch im2col OpenCL kernel + CLBlast GEMM)

Notes:

- If CLBlast is not found, UHOP falls back to tiled kernels with a one-line warning.
- On Windows, some CLBlast builds and OpenCL driver stacks have calling‑convention mismatches or event‑handling quirks when called via ctypes. UHOP now: - Loads the DLL with WinDLL (stdcall) on Windows and CDLL elsewhere - Passes an explicit cl_event\* out‑parameter (non‑NULL) to avoid NULL‑dereference bugs - Uses exact c_size_t/c_float/c_void_p arg types and row‑major leading dimensions (lda=k, ldb=n, ldc=n)
  If you still see an access‑violation in CLBlastSgemm, set `UHOP_OPENCL_CONV_IMPL=tiled` (default) and/or `UHOP_OPENCL_MATMUL_IMPL=tiled` to continue with stable tiled kernels.
- For best results, ensure your OpenCL ICD/runtime is installed and the CLBlast library matches your platform and driver stack. If multiple ICDs are present, try switching GPU vendors (AMD vs Intel) or updating drivers.

---

## AI Kernel Generation (optional)

```bash
# Generate OpenCL matmul, validate build, run smoke test
python -m uhop.cli ai-generate matmul --target opencl --validate --smoke

# Generate fused Conv2D+ReLU and benchmark vs current fused backend
python -m uhop.cli ai-generate-fused --stride 1 --padding 1
```

---

## Minimal Web API (optional)

Expose a local HTTP API for demos/automation:

```bash
uhop web-api --host 0.0.0.0 --port 5824
# or
python -m uhop.web_api --host 0.0.0.0 --port 5824
```

Endpoints

- GET `/health`
- GET `/info`
- POST `/demo/matmul` with `{ "size": 256, "iters": 3 }`

Docker

```bash
docker build -t uhop-demo-api -f api.Dockerfile .
docker run --rm -p 5824:5824 uhop-demo-api
```

---

## Contributing

We’re building UHOP as a friendly, long‑term open platform. All experience levels welcome — and we especially invite:

- GPU engineers (CUDA/ROCm/Metal/OpenCL)
- Compiler/runtime developers (Triton/MLIR/TVM)
- ML engineers and researchers (kernels, validation, datasets)
- Frontend devs (Vite/React/Tailwind, data viz)

Start here:

- Read `CONTRIBUTING.md` for local setup, tests, and PR tips
- Run `./contributing.sh setup` and `./contributing.sh test`
- Explore `issues/` for scoped design notes and milestones

Expectations:

- Keep public APIs stable; update docs/tests with behavior changes
- Aim for reproducible steps and minimal dependencies
- Small, focused PRs with clear titles (Conventional Commits encouraged)

---

## Roadmap & Timeline

Roadmap is now tracked in phases (see `docs/ROADMAP.md` for full detail & live progress). High‑level snapshot:

| Phase               | Goal                                             | Core Outcomes                                                                             | Status                                      |
| ------------------- | ------------------------------------------------ | ----------------------------------------------------------------------------------------- | ------------------------------------------- |
| 0. Foundation       | Decorator, hardware detect, cache, CLI, IR seed  | Implement stable baseline & persistence                                                   | ✅ (ongoing hardening)                      |
| 1. MVP Policy       | Multi‑backend selection & benchmarking modes     | Order probe + benchmark policy, explain tooling, KPI snapshot                             | 🚧 (policy explain + benchmark mode landed) |
| 2. Perf Parity      | Close gap vs vendor libs (CUDA / CLBlast)        | Compute‑side vectorization, better tiling, GEMM integration, autotune variance heuristics | Planned                                     |
| 3. AI Autotune Loop | AI kernel generate → validate → profile → retain | Iterative refinement, instability flags, schedule search                                  | Planned                                     |
| 4. IR Expansion     | More ops & schedules (conv, fuse, reduce)        | Multi‑op lowering, schedule strategies, artifact reuse metrics                            | Planned                                     |
| 5. Ecosystem & UX   | Dashboard, framework shims, docs polish          | Frontend progress cards, PyTorch/JAX training loop integration                            | Planned                                     |

Forward‑looking (vision): Vulkan/oneAPI experimentation, distributed kernel optimization, self‑improving training loop, broad fused op coverage, stable protocol v1.0.

Legacy issue files are being consolidated; prefer new phase‑style issues (e.g. `01-foundation.md`, `02-mvp-policy.md`). Older duplicates like `01-implement-runtime-decorator.md` & `01-pre-mvp-implement-runtime-decorator.md` will be archived.

See `docs/ROADMAP.md` for milestone checklists and progress % updates.

### Current Performance Snapshot (Representative)

Recent comparative runs (NumPy vs Torch CPU vs UHOP OpenCL vs Torch CUDA) show current OpenCL tiled kernels still trail optimized BLAS / CUDA on larger shapes:

| Shape (M,K,N)  | NumPy (ms) | Torch CPU (ms) | UHOP OpenCL (ms) | Torch CUDA (ms) |
| -------------- | ---------- | -------------- | ---------------- | --------------- |
| 256x512x256    | ~0.45      | ~6.0           | ~7.4             | ~0.05           |
| 512x512x512    | ~1.4       | ~4.0           | ~10.5            | ~0.10           |
| 1024x1024x1024 | ~9.9       | ~9.9           | ~21.2            | ~0.50           |

Vectorized B loads (float4) alone did not yield speedups on this test hardware; upcoming work will focus on compute‑side vectorization, improved memory reuse, subgroup ops, and CLBlast GEMM fallback paths.

Use `python -m uhop.cli_kpi --show` to view live KPI snapshot (selection counts, GFLOPS rows, Conv2D stage profiling) and `uhop policy explain ...` to inspect backend choices.

### Policy Explain Quick Examples

```bash
uhop policy explain matmul --arg-shape 512x512 --arg-shape 512x512 --iters 3 --warmup 1 --stats --mode benchmark
uhop policy explain relu --arg-shape 1048576 --iters 5 --stats --json
```

### Beginner Frontend Note

The frontend (Vite + React + Tailwind) currently exposes agent connection, log streaming, and basic benchmark views. Early contributors can add:

- Progress cards sourced from `python -m uhop.cli_kpi --show --json`
- Simplified "Try a Matmul" panel auto‑detecting device & showing policy selection
- A phase roadmap widget fed from `docs/ROADMAP.md`

See `frontend/README.md` for build commands; consider adding onboarding tooltips and a minimal settings drawer for env overrides in a future PR.

---

## Good First Issues

Jump in with these approachable starters:

- Improve OpenCL/kernel templates and add simple correctness tests
- Add a CUDA/HIP example parity with the OpenCL elementwise add
- Enhance `uhop info --json` fields (driver versions, memory footprints)
- Add README snippets for Windows/Mac specific setup tips
- Polish the frontend build or add a minimal dashboard card
- Optimize CI/CD workflow and docs for PRs and promotions (badges, faster CI, templates) — see [issues/15-ci-cd-workflow-docs-promo.md](issues/15-ci-cd-workflow-docs-promo.md)

Or pick one from the tracked proposals above in `issues/` and comment to claim.

---

## Testing

Run the test suite (GPU‑dependent tests skip automatically):

```bash
pytest -q

# OpenCL IR tests only (skipped if pyopencl unavailable)
pytest -q tests/test_ir_opencl_lowering.py

# Agent IR compile/validate tests
pytest -q tests/test_agent_ir_compile_validate.py
```

Targeted runs:

```bash
pytest -q tests/test_matmul.py
pytest -q -k "opencl or cuda or hip or metal"
```

---

## License

MIT © UHOP Systems

---

Tags: gpu, compiler, rocm, cuda, opencl, metal, hpc, mlops, deep-learning, open-hardware
