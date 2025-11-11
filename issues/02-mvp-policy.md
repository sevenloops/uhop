# Phase 02 — MVP Policy

Goal: Solid backend selection with transparency and light tuning.

- [x] Preference-order probe mode
- [x] Benchmark mode with warmup/iters/early-exit
- [x] CLI explain (order, candidates, stats, selected, reason)
- [x] Cache decision trace and latencies
- [ ] Frontend “policy cards” sourced from CLI JSON
- [ ] Config surface docs + examples

Acceptance:
- `uhop policy explain` works for matmul/relu/conv2d with shapes
- KPI shows backend selection counts across runs
- CI executes explain in smoke mode on CPU
