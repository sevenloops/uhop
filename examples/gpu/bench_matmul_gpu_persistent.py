# examples/bench_matmul_gpu_persistent.py
"""
Demonstrate AMD GPU speedup using OpenCL by persisting device buffers across many matmul iterations.
This shows the compute advantage when transfer overhead is amortized.
"""
import statistics
import time

import numpy as np

from uhop.backends.opencl_backend import _OPENCL_AVAILABLE

if not _OPENCL_AVAILABLE:
    raise SystemExit("OpenCL not available")

import pyopencl as cl

from uhop.backends.opencl_backend import _build_ctx_queue, _get_program


def run_opencl_matmul_persistent(A: np.ndarray, B: np.ndarray, iters: int = 30):
    ctx, q = _build_ctx_queue()
    # Simple inline matmul kernel source used by backend path for consistency
    prg_src = r"""
    __kernel void matmul(const int M, const int N, const int K,
                         __global const float* A,
                         __global const float* B,
                         __global float* C) {
        int row = get_global_id(0);
        int col = get_global_id(1);
        if (row < M && col < N) {
            float s = 0.0f;
            for (int k = 0; k < K; ++k) {
                s += A[row*K + k] * B[k*N + col];
            }
            C[row*N + col] = s;
        }
    }
    """
    prg = _get_program(ctx, prg_src, "matmul")
    k_matmul = prg.matmul
    m, k = A.shape
    k2, n = B.shape
    assert k == k2
    mf = cl.mem_flags
    a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
    b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
    c_buf = cl.Buffer(ctx, mf.WRITE_ONLY, A.dtype.itemsize * m * n)

    # Warmup
    evt = k_matmul(q, (m, n), None, np.int32(m), np.int32(n), np.int32(k), a_buf, b_buf, c_buf)
    evt.wait()

    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        evt = k_matmul(q, (m, n), None, np.int32(m), np.int32(n), np.int32(k), a_buf, b_buf, c_buf)
        evt.wait()
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
    N = 1024  # adjust upwards if GPU is underutilized
    A = np.random.rand(N, N).astype(np.float32)
    B = np.random.rand(N, N).astype(np.float32)

    C, t_gpu = run_opencl_matmul_persistent(A, B, iters=20)
    t_cpu = run_numpy_matmul_repeated(A, B, iters=20)

    print(f"GPU OpenCL (persistent) median: {t_gpu:.4f} s")
    print(f"CPU NumPy               median: {t_cpu:.4f} s")
    print("Result check:", np.allclose(C, A @ B, atol=1e-3))


if __name__ == "__main__":
    main()
