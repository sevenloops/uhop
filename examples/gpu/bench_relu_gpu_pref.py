# examples/bench_relu_gpu_pref.py
"""
Benchmark ReLU with UHOP (OpenCL) vs NumPy baseline on a large array.
ReLU is bandwidth-bound and should show GPU advantage more clearly than matmul on CPU MKL.
"""
import time
import statistics
import numpy as np
from uhop import UHopOptimizer

hop = UHopOptimizer()

@hop.optimize("relu")
def relu_np(x):
    x = np.array(x, dtype=np.float32)
    return np.maximum(x, 0.0, dtype=np.float32)


def bench(fn, X, warmup=3, iters=20):
    for _ in range(warmup):
        fn(X)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn(X)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def main():
    np.random.seed(0)
    N = 64 * 1024 * 1024  # 64M elements (~256MB)
    X = (np.random.rand(N).astype(np.float32) - 0.5) * 10.0

    t_uhop = bench(relu_np, X)
    t_base = bench(lambda x: np.maximum(x, 0.0, dtype=np.float32), X)

    print(f"UHOP (OpenCL) ReLU: {t_uhop:.4f} s median")
    print(f"NumPy CPU ReLU   : {t_base:.4f} s median")
    if t_uhop < t_base:
        print("UHOP wins ✅")
    else:
        print("Baseline faster (ensure AMD driver + OpenCL runtime, try larger N)")

if __name__ == "__main__":
    main()
