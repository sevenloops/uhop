# Issue 18: Unified Buffer Abstraction (MVP)

## Summary

Introduce a shared buffer abstraction that wraps backend-specific allocations and keeps track of data residency. This is the first step toward cross-platform unified memory while remaining minimal enough to ship quickly.

## Deliverables

- [ ] Define `UnifiedBuffer` class (`uhop/memory/unified.py`) encapsulating size, dtype, and backend handles with methods `acquire(backend)`, `mark_dirty(backend)`, and `release()`.
- [ ] Implement backend adapters for OpenCL and CPU baseline that plug into `UnifiedBuffer` (other backends fall back to simple copies for now).
- [ ] Integrate the buffer abstraction into at least one execution path (matmul) so the runtime can reuse allocations across repeated calls without manual copies.
- [ ] Add documentation to the Unified Memory pillar in `docs/PRODUCTION_VISION.md` outlining the abstraction and extension hooks.

## Acceptance Criteria

- Running `python examples/test_ai_compilation.py` with unified buffers enabled shows no regressions and reduces redundant host↔device copies in debug logs.
- `UnifiedBuffer` tracks the current owner backend; requesting a different backend triggers `copy_to` semantics exactly once per write.
- Unit tests cover buffer lifecycle, dirty-bit tracking, and adapter error handling when a backend is unavailable.

## Definition of Done

- Unified buffer abstraction merged, integration point landed, and docs/tests updated.
- Environment toggle (e.g., `UHOP_UNIFIED_BUFFER=1`) documented and default behavior decided.
- Follow-up issues filed for CUDA/Metal adapters, prefetch hints, and memory pooling.

## Notes / Dependencies

- Reuse existing buffer pool logic where possible to avoid code duplication.
- Keep the API conservative; future work can extend to asynchronous transfers and zero-copy paths.
