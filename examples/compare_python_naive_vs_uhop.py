# examples/compare_python_naive_vs_uhop.py
"""
Demonstrate UHOP optimizing a naive Python baseline to GPU via OpenCL.
This reflects UHOP's goal: take developer code and accelerate it by choosing the best backend.
"""
import statistics
import time

import numpy as np

from uhop import UHopOptimizer

hop = UHopOptimizer()


# Naive python matmul baseline (triple loop): correctness-first, very slow
@hop.optimize("matmul")
def matmul_naive(A, B):
    A = np.array(A, dtype=np.float32)
    B = np.array(B, dtype=np.float32)
    N, M = A.shape
    M2, K = B.shape
    assert M == M2
    C = np.zeros((N, K), dtype=np.float32)
    for i in range(N):
        for j in range(K):
            s = 0.0
            for k in range(M):
                s += float(A[i, k]) * float(B[k, j])
            C[i, j] = s
    return C


def bench(fn, A, B, warmup=1, iters=3):
    # small iters to avoid long waits for naive python
    for _ in range(warmup):
        fn(A, B)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn(A, B)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def main():
    np.random.seed(0)
    N = 256  # choose a size that is heavy for naive python but tractable
    A = np.random.rand(N, N).astype(np.float32)
    B = np.random.rand(N, N).astype(np.float32)

    t_uhop = bench(matmul_naive, A, B)

    # Pure naive baseline time (call undecorated function logic)
    def baseline_naive(A, B):
        A = np.array(A, dtype=np.float32)
        B = np.array(B, dtype=np.float32)
        N, M = A.shape
        M2, K = B.shape
        assert M == M2
        C = np.zeros((N, K), dtype=np.float32)
        for i in range(N):
            for j in range(K):
                s = 0.0
                for k in range(M):
                    s += float(A[i, k]) * float(B[k, j])
                C[i, j] = s
        return C

    t_base = bench(baseline_naive, A, B)

    print(f"UHOP (optimized over naive): {t_uhop:.3f} s median")
    print(f"Naive Python baseline     : {t_base:.3f} s median")
    if t_uhop < t_base:
        print("UHOP wins ✅")
    else:
        print("Baseline faster (unexpected for naive baseline)")


if __name__ == "__main__":
    main()
