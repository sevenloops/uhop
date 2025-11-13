# Issue 12: MLIR Matmul Pipeline Spike

## Summary
Bootstrap the MLIR toolchain inside UHOP by standing up a matmul-only pipeline that lowers the existing IR to runnable GPU code. The goal is to prove out the end-to-end toolchain while keeping scope tight.

## Deliverables

- [ ] `uhop/compiler/mlir/matmul.py` that emits an MLIR module for a single matmul workload using existing IR metadata.
- [ ] A lowering driver (CLI or module entry point) that converts the MLIR module into OpenCL or LLVM IR and dumps the generated kernel alongside timing info.
- [ ] Unit test exercising the driver through `tests/test_mlir_matmul_pipeline.py` with a golden-text fixture for the MLIR module.
- [ ] README snippet in `docs/PRODUCTION_VISION.md` or `docs/ROADMAP.md` linking to the new pipeline and explaining how to run it.

## Acceptance Criteria

- Running `python -m uhop.compiler.mlir.matmul --shape 128 128 128` (or equivalent CLI) writes an MLIR file and the compiled kernel artifact to `build/`.
- The generated kernel executes through the existing OpenCL backend and matches the baseline matmul output within existing tolerances.
- Tests pass on CI with the new MLIR pathway covered. Golden files only change when intentionally updated.

## Definition of Done

- Minimal matmul MLIR pipeline merged with documentation, tests, and example artifacts.
- CI job updated if new build tools (e.g., `mlir-opt`) are required; installation steps documented in `CONTRIBUTING.md` if applicable.

## Notes / Dependencies

- Requires local LLVM/MLIR toolchain. Capture install steps during implementation.
- Follow-ups will expand to additional ops and optimization passes; this issue is only for the initial vertical slice.
