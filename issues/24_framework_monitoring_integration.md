# Issue 24: Framework Integration & Observability

## Summary
Bridge UHOP into mainstream ML stacks and add production-grade observability so kernels can be deployed in real training/inference environments.

## Deliverables

- [ ] Implement PyTorch custom operator bindings for matmul/relu using the new baseline kernels (Issue 21) with automatic fallback to UHOP optimizer paths.
- [ ] Provide ONNX Runtime or TensorFlow integration hooks (choose one to start) with configuration docs.
- [ ] Add structured logging/metrics export (OpenTelemetry or Prometheus) capturing kernel selection, latency, and sandbox status.
- [ ] Update dashboard/frontend to visualize new metrics (backend usage, failures, latency percentiles).
- [ ] Document integration steps and monitoring expectations in `docs/PRODUCTION_VISION.md` (Enterprise Readiness pillar) and a new `docs/integration/pytorch.md` guide.

## Acceptance Criteria

- Sample training script (`examples/pytorch/train_mnist_uhop.py`) runs end-to-end using UHOP kernels and falls back cleanly when unavailable.
- Metrics exporter runs by default (configurable) and emits data consumable by common observability stacks.
- Dashboard shows at least latency distributions, kernel success/failure counts, and sandbox enforcement state.

## Definition of Done

- Framework bindings, metrics, and dashboard updates merged with tests and documentation; CI includes a smoke test for the bindings (with hardware skips as needed).
- Follow-up issues opened for TensorFlow/ONNX parity if the initial integration targets only one.

## Notes / Dependencies

- Depends on Issue 21 (baseline kernels) and Issue 15 (sandbox) to expose meaningful metrics.
- Coordinate with frontend maintainers to align dashboard UX with existing components.
