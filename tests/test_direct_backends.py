#!/usr/bin/env python3
"""
Direct backend testing - bypassing UHOP decorator overhead.
"""

import time

import numpy as np
import torch

from uhop.backends.opencl_backend import opencl_matmul

# Import backends directly
from uhop.backends.torch_backend import torch_matmul
from uhop.backends.triton_backend import triton_matmul


def benchmark_direct_backends(size=1024, iters=10):
    """Benchmark backends directly without decorator overhead."""
    print(f"\n{'='*70}")
    print(f"Direct Backend Benchmark (size={size}x{size}, iters={iters})")
    print("Testing raw backend performance without UHOP decorator overhead")
    print(f"{'='*70}\n")

    # Create test data
    np.random.seed(42)
    a_np = np.random.randn(size, size).astype(np.float32)
    b_np = np.random.randn(size, size).astype(np.float32)

    results = {}

    # Test 1: Raw PyTorch CUDA
    print("1. Raw PyTorch CUDA (cuBLAS) - Direct torch.matmul")
    if torch.cuda.is_available():
        a_torch = torch.tensor(a_np, device="cuda")
        b_torch = torch.tensor(b_np, device="cuda")

        # Warmup
        for _ in range(3):
            _ = torch.matmul(a_torch, b_torch)
        torch.cuda.synchronize()

        times = []
        for _ in range(iters):
            start = time.perf_counter()
            _ = torch.matmul(a_torch, b_torch)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

        avg_time = np.mean(times)
        std_time = np.std(times)
        gflops = (2 * size**3) / (avg_time * 1e9)
        results["raw_cuda"] = {"time": avg_time, "std": std_time, "gflops": gflops}
        print(f"   Average: {avg_time*1000:.3f} ms ± {std_time*1000:.3f} ms")
        print(f"   Performance: {gflops:.2f} GFLOPS\n")

    # Test 2: UHOP torch_backend wrapper (includes numpy conversion)
    print("2. UHOP torch_backend - torch_matmul() with numpy input")
    if torch.cuda.is_available():
        # Warmup
        for _ in range(3):
            _ = torch_matmul(a_np, b_np)
        torch.cuda.synchronize()

        times = []
        for _ in range(iters):
            start = time.perf_counter()
            _ = torch_matmul(a_np, b_np)
            times.append(time.perf_counter() - start)

        avg_time = np.mean(times)
        std_time = np.std(times)
        gflops = (2 * size**3) / (avg_time * 1e9)
        results["torch_backend"] = {"time": avg_time, "std": std_time, "gflops": gflops}
        print(f"   Average: {avg_time*1000:.3f} ms ± {std_time*1000:.3f} ms")
        print(f"   Performance: {gflops:.2f} GFLOPS")

        # Calculate conversion overhead
        if "raw_cuda" in results:
            overhead = avg_time - results["raw_cuda"]["time"]
            print(f"   Conversion overhead: {overhead*1000:.3f} ms\n")

    # Test 3: UHOP torch_backend with torch tensor input (no conversion)
    print("3. UHOP torch_backend - torch_matmul() with torch tensor input")
    if torch.cuda.is_available():
        a_torch = torch.tensor(a_np, device="cuda")
        b_torch = torch.tensor(b_np, device="cuda")

        # Warmup
        for _ in range(3):
            _ = torch_matmul(a_torch, b_torch)
        torch.cuda.synchronize()

        times = []
        for _ in range(iters):
            start = time.perf_counter()
            _ = torch_matmul(a_torch, b_torch)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

        avg_time = np.mean(times)
        std_time = np.std(times)
        gflops = (2 * size**3) / (avg_time * 1e9)
        results["torch_backend_tensor"] = {"time": avg_time, "std": std_time, "gflops": gflops}
        print(f"   Average: {avg_time*1000:.3f} ms ± {std_time*1000:.3f} ms")
        print(f"   Performance: {gflops:.2f} GFLOPS\n")

    # Test 4: OpenCL backend
    print("4. UHOP OpenCL backend - opencl_matmul() (tiled kernel)")
    try:
        # Warmup
        for _ in range(3):
            _ = opencl_matmul(a_np, b_np)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        times = []
        for _ in range(iters):
            start = time.perf_counter()
            _ = opencl_matmul(a_np, b_np)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

        avg_time = np.mean(times)
        std_time = np.std(times)
        gflops = (2 * size**3) / (avg_time * 1e9)
        results["opencl"] = {"time": avg_time, "std": std_time, "gflops": gflops}
        print(f"   Average: {avg_time*1000:.3f} ms ± {std_time*1000:.3f} ms")
        print(f"   Performance: {gflops:.2f} GFLOPS\n")
    except Exception as e:
        print(f"   OpenCL failed: {e}\n")

    # Test 5: Triton backend
    print("5. UHOP Triton backend - triton_matmul()")
    try:
        # Warmup
        for _ in range(3):
            _ = triton_matmul(a_np, b_np)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        times = []
        for _ in range(iters):
            start = time.perf_counter()
            _ = triton_matmul(a_np, b_np)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

        avg_time = np.mean(times)
        std_time = np.std(times)
        gflops = (2 * size**3) / (avg_time * 1e9)
        results["triton"] = {"time": avg_time, "std": std_time, "gflops": gflops}
        print(f"   Average: {avg_time*1000:.3f} ms ± {std_time*1000:.3f} ms")
        print(f"   Performance: {gflops:.2f} GFLOPS\n")
    except Exception as e:
        print(f"   Triton failed: {e}\n")

    # Test 6: NumPy CPU baseline
    print("6. NumPy CPU baseline")
    times = []
    for _ in range(min(3, iters)):
        start = time.perf_counter()
        _ = np.matmul(a_np, b_np)
        times.append(time.perf_counter() - start)

    avg_time = np.mean(times)
    std_time = np.std(times)
    gflops = (2 * size**3) / (avg_time * 1e9)
    results["cpu"] = {"time": avg_time, "std": std_time, "gflops": gflops}
    print(f"   Average: {avg_time*1000:.3f} ms ± {std_time*1000:.3f} ms")
    print(f"   Performance: {gflops:.2f} GFLOPS\n")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY - Direct Backend Performance")
    print(f"{'='*70}")
    print(f"{'Backend':<25} {'Time (ms)':<12} {'GFLOPS':<12} {'vs CPU':<10}")
    print("-" * 70)

    baseline_time = results["cpu"]["time"]
    sorted_results = sorted(results.items(), key=lambda x: x[1]["time"])

    for backend, data in sorted_results:
        speedup = baseline_time / data["time"]
        print(f"{backend:<25} {data['time']*1000:<12.3f} {data['gflops']:<12.2f} {speedup:<10.1f}x")

    print(f"{'='*70}\n")

    # Analysis
    print("ANALYSIS:")
    if "raw_cuda" in results and "torch_backend" in results:
        overhead_pct = ((results["torch_backend"]["time"] / results["raw_cuda"]["time"]) - 1) * 100
        print(f"  - torch_backend adds {overhead_pct:.1f}% overhead vs raw CUDA (numpy conversion)")

    if "raw_cuda" in results and "opencl" in results:
        ratio = results["raw_cuda"]["gflops"] / results["opencl"]["gflops"]
        print(f"  - Raw CUDA is {ratio:.1f}x faster than OpenCL (cuBLAS vs custom kernel)")

    if "opencl" in results:
        print("  - OpenCL uses tiled kernel with local memory (TILE=16)")
        print("  - OpenCL could be optimized with larger tiles, vectorization, etc.")

    print()

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test direct backend performance")
    parser.add_argument("--size", type=int, default=1024, help="Matrix size")
    parser.add_argument("--iters", type=int, default=10, help="Number of iterations")

    args = parser.parse_args()

    benchmark_direct_backends(size=args.size, iters=args.iters)
