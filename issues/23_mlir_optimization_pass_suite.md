# Issue 23: MLIR Optimization Pass Suite

## Summary

Add core MLIR passes for tiling, vectorization, shared-memory staging, and register-pressure awareness so the new compiler pipeline can compete with vendor kernels.

## Deliverables

- [ ] Implement tiling + fusion pass (`uhop/compiler/mlir/passes/tiling.py`) configurable by hardware manifest values (tile sizes, vector width).
- [ ] Add shared-memory promotion and prefetch pass (`passes/memory.py`) that uses the manifest to stage hot tensors.
- [ ] Integrate a register-pressure estimator/liveness analysis to avoid spill-heavy schedules; fail gracefully with actionable error messages.
- [ ] Wire passes into the matmul pipeline (Issue 12) and expose CLI flags to toggle each stage for debugging.
- [ ] Document pass architecture and extension points in `docs/PRODUCTION_VISION.md` and `docs/ir.md`.

## Acceptance Criteria

- Running the MLIR matmul pipeline with passes enabled produces kernels that outperform the naive version by ≥2x on at least one GPU.
- Pass pipeline unit tests cover tiling factor selection, shared-memory usage, and spill avoidance heuristics.
- Compilation logs summarize applied passes and decisions (tile sizes, vector width, memory promotion).

## Definition of Done

- Passes merged, tests green, docs updated, and CI confirms the pipeline remains functional when passes are toggled on/off.
- Follow-up issues created for additional ops (conv, attention) once matmul validates the approach.

## Notes / Dependencies

- Depends on Issues 12 and 14 for MLIR pipeline and hardware manifest.
- Coordinate with Issue 13 predictor to feed pass choices into performance modeling.
