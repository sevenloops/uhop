# Issue 22: Evolutionary & Multi-Objective Autotuning

## Summary
Build on the job orchestrator to support evolutionary search and multi-objective optimization (latency, throughput, power) so tuning scales beyond naive grid searches.

## Deliverables

- [ ] Implement candidate encoding (`uhop/autotune/candidate.py`) that captures parameter sets and mutation/crossover logic.
- [ ] Add an evolutionary driver `uhop/autotune/evolution.py` that consumes the job queue, evolves populations, and honors constraints (timeouts, resource usage).
- [ ] Incorporate a surrogate model (linking to Issue 13 predictor) to prune low-value candidates before execution.
- [ ] Expose CLI controls for objectives and weights: `python -m uhop.autotune.run --objectives latency throughput power --weights 0.6 0.3 0.1`.

## Acceptance Criteria

- Evolutionary tuner finds configurations that outperform the baseline grid search on at least one benchmark by >15% latency reduction.
- Multi-objective mode produces Pareto-optimal frontiers logged to disk (JSON) for inspection.
- Unit/integration tests cover mutation/crossover correctness, surrogate pruning efficacy, and CLI argument validation.

## Definition of Done

- Evolutionary autotuner integrated with the existing orchestrator and documented in `docs/PRODUCTION_VISION.md` (Scalable Autotuning pillar).
- Follow-up issues filed for GPU cluster execution and transfer learning across operations/hardware families.

## Notes / Dependencies

- Relies on Issue 17 (job orchestrator) and Issue 13 (performance predictor).
- Design API to accommodate future reinforcement learning or Bayesian strategies without breaking changes.
