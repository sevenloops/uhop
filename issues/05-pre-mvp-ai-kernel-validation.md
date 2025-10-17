# AI kernel validation

Stage: Pre-MVP

Labels: ai, validation

## Description
Validate AI-generated kernels by comparing outputs against reference implementations with dtype-aware tolerances.

## Goals

- `validate_callable` utility with randomized and edge-case inputs
- Strict mode toggle via `--strict-validate` or env var
- Legacy wrapper for backward compatibility

## Acceptance Criteria

- Unit tests for pass/fail and strict behavior
- Validation gated AI kernel adoption in optimizer
