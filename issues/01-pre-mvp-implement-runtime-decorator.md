# Implement @hop.optimize runtime decorator

Stage: Pre-MVP

Labels: core, runtime, in-progress

## Description
Wrap Python ops with a runtime decorator that detects hardware and dispatches to the best available backend kernel.

## Goals

- Decorator `@optimize(op_name)` usable on plain Python functions
- Preserve torch tensors and device when present (prefer CUDA/MPS over CPU)
- Integrate cache lookups and store decisions/metadata
- Validate correctness with dtype-aware tolerances

## Acceptance Criteria

- Example function runs faster on GPU/MPS/OpenCL when available
- Cache populated with backend decision and metadata
- Passes validation on randomized cases
