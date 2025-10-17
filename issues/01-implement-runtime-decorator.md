# Implement @hop.optimize runtime decorator

Wrap Python ops with a runtime decorator that detects hardware and dispatches to the best available backend kernel.

Labels: core, runtime, in-progress

## Goals

- Decorator `@optimize(op_name)` usable on plain Python functions
- Preserve torch tensors and device when present (prefer CUDA/MPS over CPU)
- Integrate cache lookups and store decisions/metadata
- Validate correctness with dtype-aware tolerances

## Acceptance Criteria

- Example function runs faster on GPU/MPS/OpenCL when available
- Cache populated with backend decision and metadata
- Passes validation on randomized cases

## References

- `uhop/optimizer.py`
- `uhop/validation.py`# Implement @hop.optimize runtime decorator

Wrap Python ops with a runtime decorator that detects hardware and dispatches to the best available backend kernel.

Labels: core, runtime, in-progress

## Goals

- Decorator `@optimize(op_name)` usable on plain Python functions
- Preserve torch tensors and device when present (CUDA/MPS > CPU)
- Integrate cache lookups and store decisions/metadata
- Validate correctness (dtype-aware tolerances)

## Acceptance Criteria

- Example function runs faster on GPU/MPS/OpenCL when available
- Cache populated with backend decision and metadata
- Passes validation on randomized cases
name: "Implement @hop.optimize runtime decorator"
labels: [core, runtime, in-progress]
# Implement @hop.optimize runtime decorator

## Summary
Wrap Python ops with a runtime decorator that detects hardware and dispatches to the best available backend kernel.

## Goals

- Decorator `@optimize(op_name)` usable on plain Python functions
- Preserve torch tensors and device when present (CUDA/MPS > CPU)
- Integrate cache lookups and store decisions/metadata
- Validate correctness (dtype-aware tolerances)

## Acceptance Criteria

- Example function runs faster on GPU/MPS/OpenCL when available
- Cache populated with backend decision and metadata
- Passes validation on randomized cases

## Notes

- Reuse `uhop.optimize` if already implemented; align naming if @hop vs @optimize.
