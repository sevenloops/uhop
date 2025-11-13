# Issue 17: Autotuning Job Orchestrator Skeleton

## Summary
Stand up the scaffolding for scalable autotuning by defining the job format, persistence schema, and a minimal parallel executor that can fan out tuning jobs across local processes. This gives us the backbone required for future evolutionary search and cloud execution.

## Deliverables

- [ ] Define a `TuningJob` dataclass (`uhop/autotune/job.py`) capturing operation, shape, backend, parameter dictionary, and priority score.
- [ ] Introduce a lightweight job queue backed by SQLite or JSONL (`uhop/autotune/queue.py`) with enqueue/dequeue APIs and rudimentary retry tracking.
- [ ] Implement a parallel executor using Python’s `concurrent.futures` that can evaluate jobs across available devices/processes, reporting results through callbacks.
- [ ] CLI entry point `python -m uhop.autotune.run --op matmul --candidates configs/matmul_candidates.json` that enqueues jobs and runs the executor until the queue is empty.

## Acceptance Criteria

- Job records persisted by the queue include start/end timestamps and measured latency along with exit status.
- Executor handles at least 100 jobs using multi-process parallelism on a developer workstation without deadlocks.
- Unit tests cover queue persistence, executor retry behavior, and deterministic ordering when priorities tie.

## Definition of Done

- Autotuning job infrastructure merged with tests and CLI wiring.
- Documentation paragraph added to `docs/PRODUCTION_VISION.md` (Scalable Autotuning pillar) referencing the new tooling.
- Backlog issues filed for evolutionary sampling, transfer learning, and remote worker support.

## Notes / Dependencies

- Keep dependencies standard library-only for now; avoid coupling to specific cloud providers.
- Reuse existing benchmarking harness for job evaluation to ensure metrics align with current autotuning code.