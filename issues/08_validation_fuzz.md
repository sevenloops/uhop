# Validation: Fuzzing/stress (param perturbations, edge shapes)

**Labels:** validation, uhop-validation

## Summary

Fuzz tile/vec/unroll parameters and exercise extreme shapes; capture failures for retraining and regression tracking.

## Tasks

- [ ] Fuzzer for parameter perturbations (in/out-of-range)
- [ ] Edge shape suites (1xN, primes, non-multiples of tile, very large)
- [ ] Failure trace capture (kernel, logs, seed, device)
- [ ] Nightly CI suite with reporting and zero silent failures

## Definition of Done

- [ ] Fuzz & stress suite runs nightly with no silent failures

## Dependencies

- Depends on: Differential Harness (issues/07_validation_diff.md)

<!-- ISSUE-SOURCE: issues/08_validation_fuzz.md -->