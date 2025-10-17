# Hardware detection refinement

Stage: Pre-MVP

Labels: system, backend

## Description
Improve detection for CUDA, MPS, OpenCL, and CPU fallback. Provide graceful errors and detailed device info.

## Goals

- Robust detection of Torch CUDA/MPS availability and preferred device
- Enumerate OpenCL platforms/devices with human-readable summaries
- Graceful fallbacks when backends unavailable

## Acceptance Criteria

- `uhop info --json` returns structured fields for torch/opencl/triton
- CLI prints vendor, device names, memory, and UHOP preferred device
