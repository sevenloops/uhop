# AI Gen: End-to-end closed-loop kernel generation for matmul

**Labels:** ai-generation, uhop-ai, infra, P1

## Summary

Implement a closed-loop system: spec extraction → AI prompt → static analysis → compile → micro-test → profile → feedback → archive. Support multiple providers (OpenAI, DeepSeek) and persist full metadata.

## Tasks

- [ ] Extract spec from IR + hardware fingerprint
- [ ] Prompt composer including prior tuned params and constraints
- [ ] Multi-provider abstraction (OpenAI, DeepSeek) with config keys
- [ ] Static analyzer for resource hints (registers/local mem), simple heuristics
- [ ] Compile and run micro-tests on randomized shapes vs reference
- [ ] Profile timing, GFLOPS, bandwidth; capture logs
- [ ] Corrective prompting on failures/slow kernels; retry with constraints
- [ ] Persist dataset records (spec, prompts, kernel, metrics, correctness, hardware)
- [ ] CLI to run the loop for matmul

## Definition of Done

- [ ] One-click loop runs and logs all attempts/metrics
- [ ] Failures retried with corrective prompts until cutoff
- [ ] Dataset artifacts queryable (by device/spec)

## Dependencies

- Depends on: Core IR (issues/01_core_ir.md)
- Relates to: Validation Harness, Dataset & Telemetry

<!-- ISSUE-SOURCE: issues/02_ai_generation_pipeline.md -->