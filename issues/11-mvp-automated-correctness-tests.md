# Automated correctness tests

Stage: MVP

Labels: testing, ci

## Description
Pytest suite comparing kernel outputs with reference NumPy/Torch. Include a few randomized sizes and edge cases.

## Goals

- Tests for matmul, relu, conv2d basics
- CI config to run tests on CPU-only

## Acceptance Criteria

- All tests pass on CI and locally
- Clear failure messages with max error stats
