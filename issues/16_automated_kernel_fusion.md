# Issue 16: Graph Capture & Fusion Candidate Report

## Summary

Introduce a lightweight graph capture and analysis pass that surfaces profitable fusion candidates before we attempt AI-generated fused kernels. This establishes the groundwork for future automatic fusion by making opportunities visible and quantifiable.

## Deliverables

- [ ] Add optional graph capture to the runtime decorator so that calling `@optimize` functions with `UHOP_CAPTURE_GRAPH=1` records op sequences and tensor metadata.
- [ ] Implement `uhop/fusion/analyzer.py` that scans captured graphs for predefined patterns (matmul+bias+relu, conv+bn+activation, elementwise chains) and emits a ranked report.
- [ ] CLI tool `python -m uhop.fusion.report --trace traces/*.json` that prints the top candidates with estimated memory bandwidth savings.
- [ ] Documentation snippet under the Fusion pillar in `docs/PRODUCTION_VISION.md` explaining capture, analysis, and how to interpret the report.

## Acceptance Criteria

- Captured graph traces are stored as JSON with op type, tensor shape, dtype, and producer/consumer links.
- Running the analyzer on provided sample traces outputs at least one candidate per pattern with a score formula documented in-code.
- Unit tests cover trace serialization/deserialization and the detection of core patterns; analyzer gracefully handles unknown ops.

## Definition of Done

- Graph capture and reporting merged with tests and documentation.
- Sample trace (small) committed under `tests/data/graph_traces/` for regression coverage.
- Follow-up issues created for automated fused-kernel generation and integration with the AI pipeline.

## Notes / Dependencies

- Keep capture opt-in to avoid affecting baseline performance; ensure environment variable flag is documented.
- Align tensor metadata with existing IR structures to reduce duplication.
