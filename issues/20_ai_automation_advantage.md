# Issue 20: Automation Metrics & Learning Loop Telemetry

## Summary

Instrument the AI pipeline so we can quantify automation leverage: how many kernels are generated, validated, promoted, and how performance improves over manual baselines. These metrics power the automation flywheel and justify investments.

## Deliverables

- [ ] Add instrumentation to the AI pipeline (`uhop/ai_codegen/pipeline.py`) that records per-run stats: number of kernel variants generated, validation pass/fail counts, autotune samples evaluated, and wall-clock time saved vs. baseline.
- [ ] Persist metrics to a structured log (`logs/automation_metrics.jsonl`) with one record per optimization session.
- [ ] Implement a reporting script `python -m uhop.analytics.automation_report --since 7d` that aggregates metrics (totals, averages, success rates) and prints a short summary table.
- [ ] Documentation update in `docs/PRODUCTION_VISION.md` (Automation & Learning Flywheel pillar) describing how to enable metrics and interpret the report.

## Acceptance Criteria

- Running the AI generation demo with metrics enabled produces a log file with at least the fields: `operation`, `variants_generated`, `validated`, `promoted`, `baseline_latency_ms`, `optimized_latency_ms`, `wallclock_seconds`, `timestamp`.
- Report script supports `--since` filtering and displays trends (e.g., success rate percentage, average speedup) without requiring external dependencies besides standard library + `tabulate` (if needed).
- Unit tests cover metric emission (with temporary directories) and the reporting aggregator; CI passes without writing persistent files outside temp dirs.

## Definition of Done

- Automation metrics instrumentation merged with tests and docs.
- Sample report output captured in documentation; follow-up issues filed for dashboard integration or pushing metrics to external sinks.
- Metrics collection has minimal overhead (<5% wall-clock impact) verified and noted in PR description.

## Notes / Dependencies

- Ensure logging respects user privacy; avoid capturing raw prompts or hardware identifiers beyond high-level model names.
- Hook into the autotuner job orchestrator (Issue 17) when available to enrich metrics with search depth information.
