# Phase 03 — Performance Parity

Close the gap vs vendor libraries on common shapes.

Matmul:
- [ ] Compute-side vectorization and unrolling
- [ ] Smarter tiling (register blocking, double-buffering)
- [ ] Subgroup/MMA-like intrinsics where available
- [ ] CLBlast GEMM fallback with stability guards

Conv2D:
- [ ] im2col tuning (local memory and launch config)
- [ ] GEMM chunking heuristics, reduced copies

Autotune:
- [ ] Per-shape params + variance tracking
- [ ] `retune_suggested` based on drift

Bench harness:
- [ ] Consistent metrics capture into KPI snapshot
