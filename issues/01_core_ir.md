# Core IR: Define UHOP Intermediate Representation and OpenCL Lowering (Matmul v1)

**Labels:** core, uhop-core, docs, P1

## Summary
Define a minimal UHOP IR capturing op type (matmul, conv2d, elementwise), shapes, layout/strides, tiling, vector width, and memory spaces. Implement IR→OpenCL lowering for matmul and document schema/versioning.

## Rationale
A stable IR is the foundation for multi-backend lowering, AI kernel generation prompts, and reproducible dataset records.

## Tasks
- [ ] Design IR schema (Python dataclasses / Pydantic) for ops (phase 1: matmul)
- [ ] Fields: op_type, dims, layout/strides, tile sizes, vector width, memory spaces
- [ ] Implement JSON serialization + stable hash (sha256 of canonical JSON)
- [ ] Add version field and migration placeholder
- [ ] Implement IR→OpenCL lowering (matmul) generating kernel source
- [ ] Unit tests: serialization, hashing, lowering correctness vs NumPy
- [ ] Add `docs/ir.md` with examples and versioning policy
- [ ] Integrate IR path into existing matmul execution (feature flag)

## Definition of Done
- IR schema documented and committed (`docs/ir.md`)
- Matmul IR lowers to OpenCL and passes correctness tests for representative shapes (square, rectangular, odd)
- Hash stable across runs for identical specs
- CI tests cover serialization + lowering

## Dependencies
- Enables: AI Generation Pipeline, HIP/SPIR-V/Metal backends
- No hard dependency; standalone start

<!-- ISSUE-SOURCE: issues/01_core_ir.md -->
