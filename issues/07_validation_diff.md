# Validation: Differential harness (NumPy/Torch reference, randomized shapes)

**Labels:** validation, uhop-validation, P1

## Summary

Create a harness comparing AI-generated kernels vs trusted baselines with randomized shapes, deterministic seeds, and tolerance rules (FP16/FP32).

## Tasks

- [ ] Randomized shape generator (odd/prime, tiny, large)
- [ ] Deterministic seeds and test IDs for reproduction
- [ ] Reference computation (NumPy/Torch) comparison
- [ ] Relative/absolute tolerance envelopes by dtype
- [ ] CI integration for nightly runs

## Definition of Done

- [ ] >99% validation pass rate on 100 randomized seeds for matmul
- [ ] Failing cases auto-logged with artifacts

## Dependencies

- Relates to: AI Gen pipeline (issues/02_ai_generation_pipeline.md), Dataset

<!-- ISSUE-SOURCE: issues/07_validation_diff.md -->