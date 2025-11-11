# Autotune: Bayesian/learned search and speedup regressor

**Labels:** ai-generation, uhop-ai, benchmark

## Summary

Extend the autotuner with model-based search to select tile/vec/unroll and build a lightweight regressor to predict speedup, reducing trials.

## Tasks

- [ ] Define parameter space per backend (tile/vec/unroll)
- [ ] Integrate Bayesian Optimization or TPE for ≤10 trial budgets
- [ ] Train a lightweight regressor on dataset attempts
- [ ] Use model to prioritize candidates pre-compile
- [ ] Persist tuned configs keyed by hardware fingerprint
- [ ] Tests: convergence to >90% optimal in ≤10 trials on fixtures

## Definition of Done

- [ ] Autotuner reaches >90% of best measured config within 10 trials
- [ ] Tuned configs saved and reused by hardware fingerprint

## Dependencies

- Depends on: Performance Dataset (issues/11_dataset_telemetry.md)
- Relates to: AI Gen pipeline (issues/02_ai_generation_pipeline.md)

<!-- ISSUE-SOURCE: issues/03_autotuning_cost_model.md -->