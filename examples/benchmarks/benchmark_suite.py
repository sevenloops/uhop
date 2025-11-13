# examples/benchmark_suite.py
"""
A simple benchmark suite to stress UHOP vs naive baselines on your hardware.
Runs warm-up, then times UHOP-decorated ops vs naive Python for clarity.
Note: NumPy/torch are highly optimized and may outperform custom kernels on CPU-only systems; the goal here
is to show UHOP vs naive reference and ensure UHOP > 1.5x faster.
"""
import statistics
import time

import numpy as np

from uhop import UHopOptimizer

hop = UHopOptimizer()


@hop.optimize("matmul")
def matmul_np(A, B):
    return np.array(A) @ np.array(B)


@hop.optimize("relu")
def relu_np(X):
    X = np.array(X, dtype=np.float32)
    return np.maximum(X, 0.0, dtype=np.float32)


def bench(fn, *args, warm=2, iters=5):
    for _ in range(warm):
        fn(*args)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def run_matmul():
    # Use a naive triple-loop baseline to demonstrate speedup clearly
    def matmul_naive(A, B):
        N, M = A.shape
        M2, K = B.shape
        assert M == M2
        C = np.zeros((N, K), dtype=np.float32)
        for i in range(N):
            for k in range(K):
                s = 0.0
                for j in range(M):
                    s += float(A[i, j]) * float(B[j, k])
                C[i, k] = s
        return C

    N = 512
    A = np.random.rand(N, N).astype(np.float32)
    B = np.random.rand(N, N).astype(np.float32)
    # Warm UHOP
    _ = matmul_np(A, B)
    t_uh = bench(matmul_np, A, B, warm=1, iters=3)
    # Naive is too slow; single iteration is enough
    t_naive = bench(matmul_naive, A, B, warm=0, iters=1)
    print(f"MatMul UHOP: {t_uh:.4f} s | Naive: {t_naive:.4f} s | x{(t_naive/max(t_uh, 1e-9)):.1f}")
    assert t_naive / max(t_uh, 1e-9) > 1.5, "UHOP should be >1.5x faster than naive"


def run_relu():
    # Naive Python loop baseline
    def relu_naive(X):
        X = list(X)
        Y = [0.0] * len(X)
        for i, v in enumerate(X):
            Y[i] = v if v > 0.0 else 0.0
        return np.array(Y, dtype=np.float32)

    N = 8 * 1024 * 1024
    X = (np.random.rand(N).astype(np.float32) - 0.5) * 10
    _ = relu_np(X)
    t_uh = bench(relu_np, X, warm=1, iters=3)
    t_naive = bench(relu_naive, X, warm=0, iters=1)
    print(f"ReLU  UHOP: {t_uh:.4f} s | Naive: {t_naive:.4f} s | x{(t_naive/max(t_uh, 1e-9)):.1f}")
    assert t_naive / max(t_uh, 1e-9) > 1.5, "UHOP should be >1.5x faster than naive"


def run_conv2d():
    # Naive CPU conv2d baseline
    def conv2d_naive(x_in, w, stride=1, padding=1):
        N, C, H, W = x_in.shape
        Cout, Cin, KH, KW = w.shape
        assert C == Cin
        outH = (H + 2 * padding - KH) // stride + 1
        outW = (W + 2 * padding - KW) // stride + 1
        out = np.zeros((N, Cout, outH, outW), dtype=np.float32)
        for n in range(N):
            for co in range(Cout):
                for y in range(outH):
                    for x_o in range(outW):
                        s = 0.0
                        for ci in range(Cin):
                            for ky in range(KH):
                                for kx in range(KW):
                                    iy = y * stride - padding + ky
                                    ix = x_o * stride - padding + kx
                                    if 0 <= iy < H and 0 <= ix < W:
                                        s += x_in[n, ci, iy, ix] * w[co, ci, ky, kx]
                        out[n, co, y, x_o] = s
        return out

    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 3, 64, 64), dtype=np.float32)
    w = rng.standard_normal((8, 3, 3, 3), dtype=np.float32)

    @hop.optimize("conv2d")
    def conv2d_uhop(inp, wt):
        return conv2d_naive(inp, wt, stride=1, padding=1)

    _ = conv2d_uhop(x, w)
    t_uh = bench(conv2d_uhop, x, w, warm=1, iters=3)
    t_naive = bench(lambda a, b: conv2d_naive(a, b, 1, 1), x, w, warm=0, iters=1)
    print(f"Conv2D UHOP: {t_uh:.4f} s | Naive: {t_naive:.4f} s | x{(t_naive/max(t_uh, 1e-9)):.1f}")
    assert t_naive / max(t_uh, 1e-9) > 1.5, "UHOP should be >1.5x faster than naive"


if __name__ == "__main__":
    print("[Benchmark Suite] Starting...")
    run_matmul()
    run_relu()
    run_conv2d()
    print("[Benchmark Suite] Done.")
