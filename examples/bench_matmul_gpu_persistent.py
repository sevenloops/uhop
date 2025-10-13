# examples/bench_matmul_gpu_persistent.py
"""
Demonstrate AMD GPU speedup using OpenCL by persisting device buffers across many matmul iterations.
This shows the compute advantage when transfer overhead is amortized.
"""
import time
import statistics
import numpy as np
from uhop.backends.opencl_backend import _OPENCL_AVAILABLE

if not _OPENCL_AVAILABLE:
    raise SystemExit("OpenCL not available")

import pyopencl as cl

from uhop.backends.opencl_backend import _ensure_ctx_queue, _ensure_program


def run_opencl_matmul_persistent(A: np.ndarray, B: np.ndarray, iters: int = 30):
    ctx, q = _ensure_ctx_queue()
    _, k_matmul, _ = _ensure_program()
    m, k = A.shape
    k2, n = B.shape
    assert k == k2
    mf = cl.mem_flags
    a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
    b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
    c_buf = cl.Buffer(ctx, mf.WRITE_ONLY, A.dtype.itemsize * m * n)

    BLOCK = 32
    global_m = ((m + BLOCK - 1) // BLOCK) * BLOCK
    global_n = ((n + BLOCK - 1) // BLOCK) * BLOCK

    # Warmup
    k_matmul.set_args(np.int32(m), np.int32(n), np.int32(k), a_buf, b_buf, c_buf)
    cl.enqueue_nd_range_kernel(q, k_matmul, (global_m, global_n), (BLOCK, BLOCK))
    q.finish()

    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        cl.enqueue_nd_range_kernel(q, k_matmul, (global_m, global_n), (BLOCK, BLOCK))
        q.finish()
        ts.append(time.perf_counter() - t0)

    # Copy result back once
    C = np.empty((m, n), dtype=A.dtype)
    cl.enqueue_copy(q, C, c_buf)
    q.finish()

    return C, statistics.median(ts)


def run_numpy_matmul_repeated(A: np.ndarray, B: np.ndarray, iters: int = 30):
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _ = A @ B
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def main():
    np.random.seed(0)
    N = 2048  # adjust upwards if GPU is underutilized
    A = np.random.rand(N, N).astype(np.float32)
    B = np.random.rand(N, N).astype(np.float32)

    C, t_gpu = run_opencl_matmul_persistent(A, B, iters=20)
    t_cpu = run_numpy_matmul_repeated(A, B, iters=20)

    print(f"GPU OpenCL (persistent) median: {t_gpu:.4f} s")
    print(f"CPU NumPy               median: {t_cpu:.4f} s")
    print("Result check:", np.allclose(C, A @ B, atol=1e-3))

if __name__ == "__main__":
    main()
