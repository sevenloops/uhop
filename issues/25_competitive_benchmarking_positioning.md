# Issue 25: Competitive Benchmarking & Positioning

## Summary
Document how UHOP compares to Triton, MLIR/IREE, oneAPI, and vendor libraries through quantitative benchmarks and qualitative migration guides so contributors and users understand where we stand and where we are heading.

## Deliverables

- [ ] Define benchmark suite (`docs/benchmarks/competitive_suite.md`) covering matmul, conv2d, attention blocks across representative shapes.
- [ ] Implement scripts under `scripts/benchmarks/` that run UHOP kernels vs Triton vs vendor libraries (cuBLAS, cuDNN, CLBlast) where licenses permit; capture results to JSON.
- [ ] Produce summary report (`docs/competitive_positioning.md`) with charts, methodology, and roadmap-driven action items.
- [ ] Add migration guidance section describing when to choose UHOP vs existing solutions and what improvements are planned.

## Acceptance Criteria

- Benchmarks run on at least one NVIDIA and one non-NVIDIA GPU; limitations clearly documented.
- Report highlights where UHOP leads/lags by ≥X% and references relevant issues (12–24) that address the gaps.
- Documentation linked from README and `docs/PRODUCTION_VISION.md` so contributors can easily find performance context.

## Definition of Done

- Benchmark scripts, raw results, and docs checked in with reproducibility instructions.
- Follow-up issues created for underperforming areas identified in the report.

## Notes / Dependencies

- Coordinate with legal/compliance on redistribution of vendor benchmark results if needed.
- Leverage Issue 21 baseline kernels to ensure we measure UHOP-native performance, not just third-party fallbacks.
