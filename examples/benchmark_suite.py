# examples/benchmark_suite.py
"""
A simple benchmark suite to stress UHOP vs baselines on your hardware.
Runs warm-up, then times UHOP-decorated ops vs NumPy/Torch for larger sizes.
"""
import time
import statistics
import numpy as np
import torch
import torch.nn.functional as F
from uhop import UHopOptimizer

hop = UHopOptimizer()

@hop.optimize("matmul")
def matmul_np(A,B):
    return np.array(A) @ np.array(B)

@hop.optimize("relu")
def relu_np(X):
    X = np.array(X, dtype=np.float32)
    return np.maximum(X, 0.0, dtype=np.float32)

def bench(fn, *args, warm=3, iters=15):
    for _ in range(warm):
        fn(*args)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def run_matmul():
    N = 2048
    A = np.random.rand(N,N).astype(np.float32)
    B = np.random.rand(N,N).astype(np.float32)
    t_uh = bench(matmul_np, A, B)
    t_np = bench(lambda a,b:a@b, A, B)
    print(f"MatMul UHOP: {t_uh:.4f} s | NumPy: {t_np:.4f} s")


def run_relu():
    N = 64*1024*1024
    X = (np.random.rand(N).astype(np.float32) - 0.5) * 10
    t_uh = bench(relu_np, X)
    t_np = bench(lambda x: np.maximum(x,0.0,dtype=np.float32), X)
    print(f"ReLU  UHOP: {t_uh:.4f} s | NumPy: {t_np:.4f} s")


def run_conv2d():
    torch.manual_seed(0)
    x = torch.randn(8,3,128,128)
    w = torch.randn(16,3,3,3)
    # UHOP path via torch conv2d in wrapper
    def uhop_conv():
        return F.conv2d(x, w, stride=1, padding=1)
    def torch_conv():
        return F.conv2d(x, w, stride=1, padding=1)
    t_uh = bench(uhop_conv)
    t_torch = bench(torch_conv)
    print(f"Conv2D UHOP(torch): {t_uh:.4f} s | Torch: {t_torch:.4f} s")

if __name__ == "__main__":
    print("[Benchmark Suite] Starting...")
    run_matmul()
    run_relu()
    run_conv2d()
    print("[Benchmark Suite] Done.")
