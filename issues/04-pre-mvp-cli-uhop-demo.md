# CLI command: uhop demo

Stage: Pre-MVP

Labels: cli, demo

## Description
Add a demo that runs a small matmul benchmark comparing naive Python vs UHOP-optimized path.

## Goals

- `uhop demo --size N --iters K` with median timing
- Warm-up to trigger backend selection/caching
- Parseable output lines for UHOP/naive timings

## Acceptance Criteria

- Demo shows speedup on GPUs/accelerators where available
- Works on CPU-only with reasonable output
