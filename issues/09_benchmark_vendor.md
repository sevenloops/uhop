# Benchmark: Replace naive baselines with vendor libs (cuBLAS/rocBLAS/oneDNN)

**Labels:** benchmark

## Summary

Integrate vendor libraries for realistic comparisons; output latency, GFLOPS, bandwidth, and occupancy (when available).

## Tasks

- [ ] Wrap cuBLAS, rocBLAS, oneDNN references
- [ ] Unified benchmark harness configuration (sizes, dtypes)
- [ ] Structured JSON reports (per backend/device)
- [ ] Occupancy/bandwidth collection (Nsight/rocprof)
- [ ] CI smoke benchmark subset

## Definition of Done

- [ ] Reports include UHOP vs vendor libs per op/device; artifacts stored

## Dependencies

- Relates to: Agent profiling (issues/10_agent_profiling.md), Dataset

<!-- ISSUE-SOURCE: issues/09_benchmark_vendor.md -->
