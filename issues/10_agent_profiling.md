# Agent: Collect profiler metrics (SM util, DRAM BW, warp/occupancy)

**Labels:** benchmark, infra

## Summary

Extend local agent to capture profiling metrics and store them alongside benchmark outputs.

## Tasks

- [ ] Integrate Nsight/rocprof or lightweight counters
- [ ] Collect SM utilization, DRAM bandwidth, warp efficiency
- [ ] Store metrics under `/benchmarks/artifacts` with run metadata
- [ ] Include in structured JSON output
- [ ] Portal integration hook (future dependency)

## Definition of Done

- [ ] Profiling metrics captured and linked to benchmark entries

## Dependencies

- Depends on: Benchmark vendor baselines (issues/09_benchmark_vendor.md)
- Relates to: Portal dashboard (issues/12_portal_ui.md)

<!-- ISSUE-SOURCE: issues/10_agent_profiling.md -->