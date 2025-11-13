# Issue 15: Kernel Sandbox & Deterministic Builds

## Summary

Ship the first enterprise-readiness slice by introducing a sandboxed execution path for AI-generated kernels and ensuring compiled artifacts are reproducible. This reduces security risk and gives downstream teams confidence that kernels are safe to run.

## Deliverables

- [ ] `uhop/security/sandbox.py` that executes compiled kernels in a restricted subprocess with timeouts and resource limits (use `subprocess` + seccomp or OS-level constraints where possible).
- [ ] Integration hook so the AI pipeline routes kernels through the sandbox prior to caching; failures fall back to the baseline implementation.
- [ ] Deterministic build check that stores a content hash alongside cached kernels and refuses to reuse entries when source + parameters diverge.
- [ ] Documentation updates: new section in `docs/PRODUCTION_VISION.md` (Enterprise Readiness pillar) and a troubleshooting paragraph in `CONTRIBUTING.md`.

## Acceptance Criteria

- Running `UHOP_ENFORCE_SANDBOX=1 python examples/test_ai_compilation.py` executes all generated kernels via the sandbox and passes existing assertions.
- Re-running the AI pipeline with identical inputs results in identical kernel hashes; mismatches emit a clear warning and discard the cache entry.
- Unit/integration tests cover sandbox timeouts, fallback behavior, and hash mismatches; CI remains green without requiring privileged permissions.

## Definition of Done

- Sandbox implementation merged with documentation, tests, and deterministic hashing support.
- Feature flag default determined (documented if opt-in/out) and referenced in release notes.
- Follow-up issues queued for advanced verification (static analysis, signing) and observability hooks.

## Notes / Dependencies

- Coordinate with infra owners to ensure sandbox approach works on Linux, macOS, and Windows where feasible.
- Keep deterministic hashing logic consistent with IR hashing to avoid redundant cache keys.
