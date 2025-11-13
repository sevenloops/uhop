# Issue 21: Core Kernel Baselines & Performance Targets

## Summary
Deliver hand-tuned baseline kernels per backend so UHOP owns critical operations rather than proxying through third-party libraries. Establish performance targets against vendor stacks to anchor future AI/codegen improvements.

## Deliverables

- [ ] Implement a reference matmul kernel per backend (OpenCL, CUDA if available, CPU fallback) under `uhop/backends/<backend>/kernels/` with clear scheduling decisions and comments.
- [ ] Add a benchmark harness `python -m uhop.benchmarks.core_ops --op matmul` that compares UHOP kernels to cuBLAS/CLBlast/PyTorch where available.
- [ ] Configure CI to run the benchmark in baseline mode and capture GFLOP metrics (skip when hardware unavailable).
- [ ] Write `docs/core_kernels.md` summarizing design choices, perf numbers, and target gaps vs. vendor libraries.

## Acceptance Criteria

- Benchmarks show UHOP kernels within 5-10x of vendor libraries on at least one supported GPU (document the delta and hardware).
- Kernels integrate with the optimizer path (can be selected/cached like other implementations).
- Regression tests validate correctness across representative shapes and dtypes.

## Definition of Done

- Baseline kernel implementations merged, benchmarks executable locally, and documentation published.
- Follow-up issues filed for extending baselines to conv2d and transformer primitives once matmul is complete.

## Notes / Dependencies

- May rely on optional CUDA toolchain; document setup and skip gracefully in CI.
- Use this issue to codify coding standards for hand-tuned kernels (naming, comments, profiling notes).
