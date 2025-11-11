# Infra: Unified performance dataset (JSONL/SQLite)

**Labels:** infra

## Summary

Create a performance dataset capturing (op, shape, backend, kernel variant, metrics, correctness) and expose a simple query API.

## Tasks

- [ ] Choose storage (SQLite or JSONL) and define schema
- [ ] CRUD utilities and query API (by device/spec/time)
- [ ] Write-through from autotune and AI-gen pipeline
- [ ] Data retention and versioning policy
- [ ] Unit tests for schema and queries

## Definition of Done

- [ ] Dataset grows automatically as kernels are generated/tuned
- [ ] Queryable for modeling and dashboards

## Dependencies

- Relates to: AI generation, Autotuning, Portal UI

<!-- ISSUE-SOURCE: issues/11_dataset_telemetry.md -->