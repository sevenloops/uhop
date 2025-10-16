# Project Context for Copilot

U-HOP (Universal Hardware Optimization Protocol) is an AI-driven, cross-hardware optimization runtime.
It detects available compute devices (NVIDIA CUDA, AMD ROCm, Apple Metal, OpenCL, Vulkan, Triton, etc.),
and automatically compiles or generates optimized kernels for each backend.

It serves as an abstraction layer between frameworks like PyTorch and diverse hardware APIs.
The goal is to eliminate vendor lock-in, enable dynamic kernel generation, and optimize runtime performance
through continuous benchmarking and AI-driven tuning.

## Current focus

- OpenCL backend (Windows/AMD): tiled matmul, ReLU, and fused Conv2D+ReLU via im2col+matmul+relu.
- Autotuned BLOCK size per device and program/kernel caching.
- CLI commands: info, demo (naive vs UHOP), demo-conv2d-relu, ai-generate (with validate/smoke/samples).
- AI codegen: OpenAI v1 integration, multi-candidate generation, smoke tests and manifest logging, optimizer auto-cache for matmul (ai_opencl).
- Reduce UHOP overhead vs baseline (optimize caching, async streams, and launch config).

## Wow demos (try on your machine)

- Naive vs UHOP MatMul (GPU-preferred):
  - python -m uhop.cli demo --size 192
- Fused Conv2D+ReLU (OpenCL fused path where available):
  - python -m uhop.cli demo-conv2d-relu --h 128 --w 128 --c-in 3 --c-out 32 --k 3 --stride 1 --padding 1
  - Tip: increase H/W and channels for more GPU-friendly workloads, e.g. --h 224 --w 224 --c-in 16 --c-out 32

- AI generation + selection (OpenCL matmul):
  - python -m uhop.cli ai-generate matmul --target opencl --validate --smoke --samples 2
  - Produces a manifest and caches the best kernel for later optimizer use.

## Wow factor

- UHOP can offload MatMul, ReLU, and Conv2D to AMD GPUs via OpenCL, outperforming naive Python and matching or beating NumPy on large sizes.
- Fused Conv2D+ReLU kernel demo shows significant speedup over separate launches, with device selection and benchmarking via CLI.
- CLI supports hardware introspection, device selection, and reproducible benchmarks for fair comparisons.
- All code is cross-platform, with a modular backend system and AI kernel generation pipeline.

## Device controls

- List devices and details:
  - python -m uhop.cli info
  - python -m uhop.cli info --json
- Select a GPU device:
  - python -m uhop.cli info --ocl-device 0
  - python -m uhop.cli demo --ocl-device 0
  - Or set environment variable UHOP_OPENCL_DEVICE_INDEX

## Reproducibility and manifests

- AI generation writes a JSON manifest next to generated kernels:
  - Contains model, prompt, candidates with Linf/time, and selected best.
  - Example: uhop/generated_kernels/ai_matmul_manifest.json

## Next steps for contributors

- Optimizer: add ai_opencl cached-path support for ReLU and Conv2D (mirroring matmul).
- Performance: persistent residency, overlapping transfers with compute, and broader autotuning (BLOCK, vector width, unrolling).
- AI codegen: add CUDA/Triton targets with smoke tests and manifests, plus failure analytics in manifest.
- UX: more CLI affordances (e.g., --use-cached, --clear-cache) and richer hardware report JSON.
