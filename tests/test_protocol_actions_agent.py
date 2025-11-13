import pytest

from uhop.agent import _compile_kernel, _validate


def test_compile_kernel_opencl_or_skip():
    try:
        import pyopencl  # noqa: F401
    except Exception:
        pytest.skip("pyopencl not available")
    # Minimal passthrough kernel
    src = r"""
    __kernel void id_copy(__global const float* A, __global float* B, const int N){
        int i = get_global_id(0);
        if(i < N) B[i] = A[i];
    }
    """
    out = _compile_kernel({"source": {"lang": "opencl", "text": src}, "schedule": {"tile": 8, "vec": 1}})
    art = out.get("artifact") or {}
    assert art.get("id", "").startswith("opencl:")
    assert "compiler_opts" in art


def test_validate_matmul_numpy():
    res = _validate(
        {"op": "matmul", "backend": "numpy", "shapes": {"A": [8, 8], "B": [8, 8]}, "runs": 2, "tolerance": 1e-4}
    )
    assert res.get("ok") is True
    v = res.get("validated") or {}
    assert v.get("passed") is True
    assert v.get("max_abs_err") <= 1e-4
