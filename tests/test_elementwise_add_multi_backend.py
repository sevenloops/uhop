import importlib.util
import shutil

import pytest


def has_cupy():
    return importlib.util.find_spec("cupy") is not None


def has_opencl():
    return importlib.util.find_spec("pyopencl") is not None


def has_metal_toolchain():
    return shutil.which("metal") is not None and shutil.which("metallib") is not None


@pytest.mark.parametrize("backend", ["cuda", "opencl", "hip", "metal"])
def test_elementwise_add_autotune_multi_backend(backend):
    from uhop.autotuner import autotune_elementwise

    # Skip when deps/toolchains are not present
    if backend == "cuda" and not has_cupy():
        pytest.skip("CuPy not available; skipping CUDA autotune test")
    if backend == "opencl" and not has_opencl():
        pytest.skip("PyOpenCL not available; skipping OpenCL autotune test")
    if backend == "metal" and not has_metal_toolchain():
        pytest.skip("Apple metal/metallib not available; skipping Metal compile test")

    try:
        result = autotune_elementwise("add", size=1 << 16, dtype="float32", device=backend)
    except RuntimeError as e:
        # If HIP requires cupy-rocm and isn't present, skip gracefully
        if backend == "hip":
            pytest.skip(f"HIP runtime not available: {e}")
        raise

    assert isinstance(result, dict)
    assert "latency_s" in result
    assert result["latency_s"] >= 0.0
    assert result.get("backend") == backend
