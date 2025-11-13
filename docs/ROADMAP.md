# UHOP Roadmap (Phased)

This roadmap tracks near-term milestones and progress. It complements the README and replaces scattered issue documents. For the long-range production vision, see [`docs/PRODUCTION_VISION.md`](PRODUCTION_VISION.md).

Legend: ✅ Done • 🚧 In progress • ⏳ Planned

## Phase 0 — Foundation (✅ ongoing hardening)

- [x] Runtime decorator with backend optimizer and per-shape cache
- [x] Hardware detection and capability introspection
- [x] Basic CLI (info, demos, cache tools)
- [x] Metrics snapshot (`python -m uhop.cli_kpi --show`)
- [x] Minimal IR (MatMul/Relu/Fused) + OpenCL lowering + agent compile/validate
- [x] Torch shim for matmul/relu
- [x] Vulkan backend PoC (env-gated)

## Phase 1 — MVP Policy (🚧)

- [x] Preference order selection (order_probe)
- [x] Benchmark mode selection with warmup/iters/early-exit
- [x] Policy explain CLI showing candidates, stats, and reason
- [x] Cache trace of decisions and latencies
- [x] KPI surfacing for selection counts and perf rows
- [ ] Frontend “progress cards” sourced from KPI snapshot
- [ ] Dashboard basics (agent status, logs, quick benchmarks)

## Phase 2 — Performance Parity (⏳)

- [ ] Compute-side vectorization and improved tiling (see Issue 23)
- [ ] Double-buffering and better local memory reuse (Issue 23)
- [ ] Subgroup/MMA-style paths where available (Issues 14, 23)
- [ ] CLBlast GEMM fallback (matmul) and im2col+GEMM (conv2d) with guards
- [ ] Autotune variance heuristics (retune_suggested + thresholds)
- [ ] Core kernel baselines benchmarked vs vendor libraries (Issue 21)
- [ ] Evolutionary autotuning driver delivering >15% speedups over grid search (Issue 22)

## Phase 3 — AI Autotune Loop (⏳)

- [ ] AI kernel generation → validation → profile → retain best
- [ ] Instability flags and automatic fallback paths
- [ ] Schedule search space definitions per op
- [ ] MLIR pass suite (tiling, memory promotion, register heuristics) integrated (Issue 23)
- [ ] Unified buffer abstraction powering cross-backend memory reuse (Issue 18)
- [ ] Framework integration + observability for production deployments (Issue 24)
- [ ] Competitive benchmarking dataset published (Issue 25)
- [ ] Sandbox enforcement + deterministic build hashing enabled by default (Issue 15)
- [ ] Release checklist adopted for production drops (Issue 19)

## Phase 4 — IR Expansion (⏳)

- [ ] More ops (conv, reductions) + fusions
- [ ] Schedule strategies (tiling/vectorization templates)
- [ ] Artifact reuse metrics (ir_key × device registry)

## Phase 5 — Ecosystem & UX (⏳)

- [ ] Framework shims (PyTorch/JAX training loop integration)
- [ ] Frontend polish: onboarding, timeline, settings
- [ ] Docs cleanup and beginner walkthroughs

---

## Performance Status Snapshot

Representative matmul results (from recent local test hardware):

| Shape (M,K,N)  | NumPy (ms) | Torch CPU (ms) | UHOP OpenCL (ms) | Torch CUDA (ms) |
| -------------- | ---------- | -------------- | ---------------- | --------------- |
| 256x512x256    | ~0.45      | ~6.0           | ~7.4             | ~0.05           |
| 512x512x512    | ~1.4       | ~4.0           | ~10.5            | ~0.10           |
| 1024x1024x1024 | ~9.9       | ~9.9           | ~21.2            | ~0.50           |

Notes:

- Guarded float4 vector loads on B did not improve on this stack; focus shifts to compute-side vectorization and memory scheduling.
- Compile reuse via IRKernelIndex reduced rebuilds from seconds to sub-second on repeat.

Use `python -m uhop.cli_kpi --show` for your machine’s snapshot.

---

## Issue Consolidation Guidance

We’re moving from many pre-MVP issue files to a compact set per phase:

- 01-foundation.md
- 02-mvp-policy.md
- 03-perf-parity.md
- 04-ai-autotune.md
- 05-ir-expansion.md
- 06-ecosystem-ux.md
- 07-production-readiness.md (tracks Issues 15, 19, 24)
- 08-compiler-and-performance.md (tracks Issues 12–14, 21–23, 25)

Older overlapping files (e.g., `01-implement-runtime-decorator.md`, `01-pre-mvp-*`) will be archived. New work should be filed against the phase files above with checklists and acceptance criteria.
