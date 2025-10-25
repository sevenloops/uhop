"""
Benchmark matmul on Apple MPS with resident torch tensors.

Compares UHOP(@optimize("matmul")) vs pure torch.matmul, keeping data on MPS
to avoid PCIe/host copies that skew smaller benchmarks.
"""
import time
import statistics
import numpy as np

import torch
from uhop import optimize


def _sync():
    try:
        if hasattr(torch, "mps"):
            torch.mps.synchronize()
    except Exception:
        pass


@optimize("matmul")
def matmul_np(a, b):
    # UHOP will keep tensor outputs if inputs are tensors; when we
    # pass torch tensors below, this is effectively a no-op wrapper.
    return np.array(a) @ np.array(b)


def bench(fn, A, B, warmup=2, iters=10):
    for _ in range(warmup):
        _ = fn(A, B)
        _sync()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _ = fn(A, B)
        _sync()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def main():
    if not (
        getattr(torch.backends, "mps", None)
        and torch.backends.mps.is_available()
    ):
        print("MPS not available; this example targets Apple Silicon.")
        return
    torch.manual_seed(0)
    dev = torch.device("mps")
    # choose sizes that make matmul heavy enough to amortize launch overhead
    N = 2048
    A = torch.randn(N, N, device=dev, dtype=torch.float32)
    B = torch.randn(N, N, device=dev, dtype=torch.float32)

    # Warm UHOP path once
    _ = matmul_np(A, B)
    _sync()

    t_uhop = bench(matmul_np, A, B, warmup=1, iters=5)

    def torch_mm(a, b):
        return torch.matmul(a, b)

    t_torch = bench(torch_mm, A, B, warmup=1, iters=5)

    print(f"UHOP (matmul, MPS resident): {t_uhop:.4f} s median")
    print(f"Torch matmul (MPS)       : {t_torch:.4f} s median")


if __name__ == "__main__":
    main()
