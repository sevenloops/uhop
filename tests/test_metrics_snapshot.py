import json

from uhop.metrics import write_kpi_snapshot


def test_write_kpi_snapshot(tmp_path):
    # Prepare fake minimal cache structure
    cache_dir = tmp_path / ".uhop_mvp_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "index.json").write_text('{"_meta": {"cache_version": 1}, "matmul|foo": {"backend": "opencl"}}')
    (cache_dir / "autotune.json").write_text('{"opencl|matmul|matmul_tiled|MockDevice|N64_M64_K64": {"last_ms": 0.5, "last_gflops": 10.0}, "opencl|conv2d|im2col_gemm|MockDevice|N1_C3_H32_W32_K16": {"last_ms": 1.2, "im2col_ms": 0.3, "gemm_ms": 0.7, "copy_ms": 0.2, "chunked": true, "chunk_count": 2, "var_ms": 0.05, "retune_suggested": false}}')

    out_file = tmp_path / "snapshot.json"
    write_kpi_snapshot(out_file, cache_dir=cache_dir)
    data = json.loads(out_file.read_text())
    assert "backend_selection_counts" in data
    assert data["backend_selection_counts"].get("opencl") == 1
    assert any(r.get("kernel") == "matmul_tiled" for r in data.get("opencl_matmul", []))
    # Conv2D row presence and stage timing propagation
    conv_rows = data.get("opencl_conv2d", [])
    assert any(r.get("kernel") == "im2col_gemm" and r.get("chunked") is True for r in conv_rows)
    # Ensure stage metrics captured
    assert any(r.get("im2col_ms") == 0.3 and r.get("gemm_ms") == 0.7 for r in conv_rows)
