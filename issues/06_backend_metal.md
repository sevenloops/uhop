# Backend: Metal threadgroup-aware matmul (FP16/FP32)

**Labels:** backend

## Summary

Implement a Metal backend with threadgroup memory; validate correctness and stability for FP16/FP32 on M-series Macs.

## Tasks

- [ ] IR→Metal lowering (matmul) with threadgroup memory usage
- [ ] Build integration and device probing
- [ ] FP16 correctness tests and tolerance envelopes
- [ ] Minimal performance benchmarks vs Accelerate

## Definition of Done

- [ ] Metal backend matmul passes correctness tests and runs locally

## Dependencies

- Depends on: Core IR (issues/01_core_ir.md)
- Relates to: Validation, Benchmarking

<!-- ISSUE-SOURCE: issues/06_backend_metal.md -->