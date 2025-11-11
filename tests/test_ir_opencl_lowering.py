import pytest
import numpy as np


def _has_pyopencl():
    try:
        import pyopencl as cl  # noqa: F401

        return True
    except Exception:
        return False


@pytest.mark.skipif(not _has_pyopencl(), reason="pyopencl not available")
def test_lowered_matmul_correct_small():
    from uhop.ir import Tensor, MatMul
    from uhop.ir.opencl_lowering import lower_to_opencl
    import pyopencl as cl

    M, K, N = 4, 3, 5
    A = np.random.default_rng(0).random((M, K), dtype=np.float32)
    B = np.random.default_rng(1).random((K, N), dtype=np.float32)
    ref = A @ B

    op = MatMul(A=Tensor("A", (M, K)), B=Tensor("B", (K, N)))
    low = lower_to_opencl(op)
    assert low["language"] == "opencl"
    ctx = cl.create_some_context(interactive=False)
    q = cl.CommandQueue(ctx)
    prg = cl.Program(ctx, low["source"]).build()
    kern = getattr(prg, low.get("kernel_name") or "uhop_matmul")
    mf = cl.mem_flags
    bufA = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
    bufB = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
    out = np.empty((M, N), dtype=np.float32)
    bufC = cl.Buffer(ctx, mf.WRITE_ONLY, size=out.nbytes)
    kern.set_args(np.int32(M), np.int32(N), np.int32(K), bufA, bufB, bufC)
    cl.enqueue_nd_range_kernel(q, kern, (M, N), None)
    cl.enqueue_copy(q, out, bufC)
    q.finish()
    assert np.allclose(out, ref, atol=1e-4)


@pytest.mark.skipif(not _has_pyopencl(), reason="pyopencl not available")
def test_lowered_fused_matches_separate():
    from uhop.ir import Tensor, FusedMatMulRelu
    from uhop.ir.opencl_lowering import lower_to_opencl
    import pyopencl as cl

    M, K, N = 4, 3, 5
    A = np.random.default_rng(0).random((M, K), dtype=np.float32)
    B = np.random.default_rng(1).random((K, N), dtype=np.float32)
    ref = np.maximum(A @ B, 0)

    op = FusedMatMulRelu(A=Tensor("A", (M, K)), B=Tensor("B", (K, N)))
    low = lower_to_opencl(op)
    ctx = cl.create_some_context(interactive=False)
    q = cl.CommandQueue(ctx)
    prg = cl.Program(ctx, low["source"]).build()
    kern = getattr(prg, low.get("kernel_name") or "uhop_fused_matmul_relu")
    mf = cl.mem_flags
    bufA = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
    bufB = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
    out = np.empty((M, N), dtype=np.float32)
    bufY = cl.Buffer(ctx, mf.WRITE_ONLY, size=out.nbytes)
    kern.set_args(np.int32(M), np.int32(N), np.int32(K), bufA, bufB, bufY)
    cl.enqueue_nd_range_kernel(q, kern, (M, N), None)
    cl.enqueue_copy(q, out, bufY)
    q.finish()
    assert np.allclose(out, ref, atol=1e-4)
