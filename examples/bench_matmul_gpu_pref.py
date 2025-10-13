# examples/bench_matmul_gpu_pref.py
"""
Benchmark matmul with UHOP (GPU-preferred backends) vs NumPy baseline.
On AMD Radeon, UHOP should select OpenCL and beat CPU NumPy for moderate sizes.
"""
import time
import statistics
import numpy as np
from uhop import UHopOptimizer

hop = UHopOptimizer()

@hop.optimize("matmul")
def matmul_np(A, B):
    return np.array(A) @ np.array(B)


def bench(fn, A, B, warmup=2, iters=10):
    for _ in range(warmup):
        fn(A, B)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn(A, B)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return statistics.median(times)


def main():
    np.random.seed(0)
    # Use a size that favors GPU; tune as needed
    N = 1024
    A = np.random.rand(N, N).astype(np.float32)
    B = np.random.rand(N, N).astype(np.float32)

    # UHOP (will pick OpenCL/Torch/Triton based on availability)
    t_uhop = bench(matmul_np, A, B)

    # Pure NumPy baseline
    def baseline(A, B):
        return A @ B
    t_base = bench(baseline, A, B)

    print(f"UHOP (optimized): {t_uhop:.4f} s median")
    print(f"Baseline NumPy  : {t_base:.4f} s median")
    if t_uhop < t_base:
        print("UHOP wins ✅")
    else:
        print("Baseline faster (try larger N or ensure GPU drivers are active)")

if __name__ == "__main__":
    main()
