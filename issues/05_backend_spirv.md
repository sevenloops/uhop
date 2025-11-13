# Backend: IR→SPIR-V (SYCL/oneAPI/Vulkan) via MLIR

**Labels:** backend

## Summary

Add SPIR-V lowering using MLIR or SYCL; target Intel Xe / Vulkan-capable devices for matmul.

## Tasks

- [ ] Evaluate MLIR vs direct SYCL pipeline feasibility
- [ ] Implement IR→SPIR-V lowering (matmul v1)
- [ ] Integrate oneAPI or Vulkan execution path
- [ ] Device probing & capability gating
- [ ] Tests and on-device validation

## Definition of Done

- [ ] SPIR-V backend compiles & runs matmul on at least one device

## Dependencies

- Depends on: Core IR (issues/01_core_ir.md)
- Relates to: Validation, Benchmarking

<!-- ISSUE-SOURCE: issues/05_backend_spirv.md -->
