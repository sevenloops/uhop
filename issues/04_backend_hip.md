# Backend: HIP (ROCm) support for matmul and conv2d

**Labels:** backend

## Summary

Implement HIP backend by translating IR→HIP; validate on AMD GPUs and compare parity with OpenCL.

## Tasks

- [ ] Implement IR→HIP lowering (matmul v1)
- [ ] Implement IR→HIP lowering (conv2d v1)
- [ ] Toolchain integration and environment probes
- [ ] Fallback/escalation policy (HIP→OpenCL→CPU)
- [ ] Unit tests and on-device smoke tests (AMD)
- [ ] Throughput comparison vs OpenCL; log results

## Definition of Done

- [ ] HIP backend runs matmul/conv2d correctly
- [ ] Parity report collected on AMD GPU(s)

## Dependencies

- Depends on: Core IR (issues/01_core_ir.md)
- Relates to: Benchmarking & Dataset

<!-- ISSUE-SOURCE: issues/04_backend_hip.md -->