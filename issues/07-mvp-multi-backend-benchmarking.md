# Add multi-backend benchmarking

Stage: MVP

Labels: benchmark, mvp

## Description
Compare CPU, CUDA, MPS, OpenCL backends for selected ops and cache the best performer per shape/dtype.

## Goals

- Benchmark harness with repeatable runs
- Cache winner with device+shape keying

## Acceptance Criteria

- Report shows per-backend times and selected best
- Optimizer respects cached winner on subsequent runs
