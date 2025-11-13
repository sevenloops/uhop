# Issue 13: Performance Dataset & Baseline Predictor

## Summary
Lay the groundwork for learned performance modeling by capturing kernel telemetry into a structured dataset and training a simple predictor that estimates matmul latency. This supplies training data and APIs that later models can extend.

## Deliverables

- [ ] Introduce `uhop/perfdata/collector.py` that records kernel shape, backend, tuning parameters, and measured latency into a local Parquet (or JSONL) file.
- [ ] Add a CLI entry point `python -m uhop.perfdata.collect --op matmul` that runs a seeded suite of matmul workloads and generates at least 200 labeled samples.
- [ ] Implement `uhop/perfdata/predictor.py` with a lightweight model (e.g., gradient boosted trees via scikit-learn) that trains on the collected dataset and exposes `predict_latency(features)`.
- [ ] Documentation section under `docs/PRODUCTION_VISION.md` (Performance Modeling pillar) describing how to collect data and run the predictor.

## Acceptance Criteria

- Dataset schema includes: operation, shape, backend name, tuning config hash, device summary, elapsed microseconds.
- `python -m uhop.perfdata.predict --op matmul --shape 256 256 256` prints a predicted latency and confidence interval without executing a kernel.
- Unit tests cover feature extraction and predictor serialization; CI runs tests without requiring GPUs (fall back to mocked data when hardware absent).

## Definition of Done

- Telemetry collection workflow checked in with tests and docs.
- Baseline predictor artifact (e.g., saved model file) produced deterministically from collected data and ignored via `.gitignore` when generated locally.
- Follow-up issues identified for multi-op support and integration with autotuner selection.

## Notes / Dependencies

- Reuse existing benchmarking utilities where possible to avoid duplicating kernel invocation code.
- If scikit-learn is too heavy for runtime dependencies, gate imports behind the CLI and document installation requirements.
