# Issue 14: Hardware Capability Manifest

## Summary

Create a shared hardware capabilities manifest that kernel generation and policy code can query. This gives prompts, schedulers, and heuristics concrete data about caches, wavefronts, Tensor Cores, and memory fabrics per device family.

## Deliverables

- [ ] Schema definition (`docs/hardware_manifest.md`) describing the required fields for each architecture (cache sizes, shared memory, wavefront width, tensor engines, peak bandwidth, etc.).
- [ ] Machine-readable manifests stored under `uhop/hardware/manifests/*.json` with entries for at least: NVIDIA Ada/Hopper, AMD MI200, Intel PVC, Apple M-series, and a generic fallback.
- [ ] Loader utility `uhop/hardware/registry.py` that exposes `get_capabilities(device_name)` and returns a typed dataclass.
- [ ] Unit tests covering schema validation and representative lookups, plus a stub ensuring manifests stay synced with the schema.

## Acceptance Criteria

- `python -m uhop.hardware.registry --list` prints the known device families and key stats (SM count, shared-memory per SM, tensor engine availability).
- Kernel generation prompts switch to consuming the manifest (e.g., `HardwareAwareGenerator` now reads from `get_capabilities`).
- Adding a new manifest triggers validation errors if required fields are missing or incorrectly typed.

## Definition of Done

- Manifest schema reviewed and linked from `docs/PRODUCTION_VISION.md` (Hardware Awareness pillar).
- Initial vendor manifests land with automated validation and cross references in documentation.
- Follow-up issues filed for per-device prompt tuning and runtime detection integration.

## Notes / Dependencies

- Reuse existing device-detection code to seed manifest defaults where possible.
- Keep manifests lightweight and JSON-serializable so they can ship with binaries or fetch remotely in future.
