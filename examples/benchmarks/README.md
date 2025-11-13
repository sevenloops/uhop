Benchmark artifacts

- Put benchmark result JSON files here (e.g., benchmark_results.json), not in the project root.
- Keep filenames descriptive (include device, date, and key params). Example: matmul_conv2d_opencl_radeon_rx6800_2025-01-15.json
- Prefer a stable schema:
  {
  "metadata": {
  "uhop_version": "0.1.0",
  "device": "OpenCL AMD Radeon RX 6800",
  "timestamp": "2025-01-15T12:34:56Z"
  },
  "results": {
  "matmul": {
  "256x256": {"opencl_tiled_ms": 0.42, "torch_cpu_ms": 0.65}
  },
  "relu": {
  "1M": {"opencl_ms": 0.08, "torch_cpu_ms": 0.11}
  }
  }
  }

Tips

- For reproducibility, record shapes, dtypes, iterations, and warmup.
- If results come from CLI, capture the command used in metadata.
- Consider committing a small, representative sample; large files should be kept out of version control or compressed.
