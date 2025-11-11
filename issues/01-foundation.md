# Phase 01 — Foundation

Scope: Baseline features, stable APIs, initial docs.

- [x] Decorator + optimizer + cache
- [x] Hardware detection & info
- [x] CLI basics (info, demos)
- [x] Minimal IR + OpenCL lowering + agent compile/validate
- [x] Torch shim (matmul/relu)
- [x] KPI snapshot utility
- [x] Vulkan PoC (env-gated)
- [ ] Harden logging and config UX (`uhop config list`)
- [ ] Archive legacy overlapping issues

Risks: driver variance, Windows packaging, optional CLBlast.

Deliverable: README and docs aligned; CI green on CPU-only; gated GPU tests.
