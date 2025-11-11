import pytest


def _has_pyopencl():
    try:
        import pyopencl as cl  # noqa: F401

        return True
    except Exception:
        return False


@pytest.mark.skipif(not _has_pyopencl(), reason="pyopencl not available")
def test_agent_compile_and_validate_ir_matmul():
    from uhop.agent import _compile_kernel, _validate
    from uhop.ir import Tensor, MatMul

    M, K, N = 4, 3, 5
    ir = MatMul(A=Tensor("A", (M, K)), B=Tensor("B", (K, N))).to_dict()
    comp = _compile_kernel({"ir": ir})
    art = comp.get("artifact", {})
    assert art.get("id", "").startswith("opencl:")
    assert art.get("kernel_name") in (None, "uhop_matmul") or art.get("kernel_name").startswith("uhop_")

    res = _validate({"ir": ir, "tolerance": 1e-4})
    assert res["ok"] is True
    assert res["validated"]["passed"] is True


@pytest.mark.skipif(not _has_pyopencl(), reason="pyopencl not available")
def test_agent_validate_ir_fused():
    from uhop.agent import _validate
    from uhop.ir import Tensor, FusedMatMulRelu

    M, K, N = 4, 3, 5
    ir = FusedMatMulRelu(A=Tensor("A", (M, K)), B=Tensor("B", (K, N))).to_dict()
    res = _validate({"ir": ir, "tolerance": 1e-4})
    assert res["ok"] is True
    assert res["validated"]["passed"] is True
