# Vulkan Backend PoC (stub)

This is a placeholder backend to demonstrate how new backends are wired into UHOP. It does not execute real kernels.

## Enable

Set an environment variable to opt in:

```bash
export UHOP_ENABLE_VULKAN_POC=1
```

Then run any command that queries backends (e.g., `uhop backends list` or `python -m uhop.cli_kpi --show`) to see `vulkan` registered. The backend reports `available: true` but has no kernels and will not be selected by the optimizer.

CI notes: because the PoC is disabled by default and gated by an env var, it is safe for CI. No external Vulkan dependencies are required.

## Next steps (future work)

- Enumerate devices via a Python Vulkan binding
- Implement a minimal shader pipeline for matmul with SPIR-V
- Add correctness and micro-benchmarks
- Integrate schedule hints (workgroup sizes, tiling)
