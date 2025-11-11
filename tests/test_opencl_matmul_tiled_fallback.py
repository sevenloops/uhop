import numpy as np
import pytest

from uhop.backends import is_opencl_available

@pytest.mark.skipif(not is_opencl_available(), reason="OpenCL not available")
def test_opencl_tiled_fallback(monkeypatch):
    # Force tiled implementation
    monkeypatch.setenv("UHOP_OPENCL_MATMUL_IMPL", "tiled")
    monkeypatch.setenv("UHOP_OPENCL_ENABLE_TILED", "1")
    # Inject failure by marking shape unstable ahead of time
    from uhop.cache import UhopAutotune as _Auto
    _ = _Auto()
    # We don't know actual device name here without context; fallback to marking after first attempt
    # Instead monkeypatch the kernel source loader to raise, simulating build failure -> fallback
    import uhop.backends.opencl_backend as ocl

    def _bad_src():
        raise RuntimeError("simulated source load failure")

    monkeypatch.setattr(ocl, "_load_matmul_tiled_source", _bad_src)
    # Provide small matrices
    a = np.random.default_rng(0).random((8, 8), dtype=np.float32)
    b = np.random.default_rng(1).random((8, 8), dtype=np.float32)
    # Execute; should fallback to naive path and still be correct
    out = ocl.opencl_matmul(a, b)
    ref = a @ b
    err = float(np.max(np.abs(out - ref)))
    assert err < 1e-3, f"Fallback naive result incorrect err={err}"
