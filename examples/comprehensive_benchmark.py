#!/usr/bin/env python3
"""
Comprehensive UHOP Benchmark Suite

Tests all operations × all backends × current hardware and generates
a detailed performance matrix with:
- Operation performance across backends
- Hardware utilization metrics
- Speedup comparisons
- HTML report generation

Usage:
    python3 comprehensive_benchmark.py --all
    python3 comprehensive_benchmark.py --op matmul --backends torch,opencl
    python3 comprehensive_benchmark.py --report report.html
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Import UHOP backends
try:
    from uhop.backends.torch_backend import (
        is_torch_available,
        torch_conv2d,
        torch_matmul,
        torch_relu,
    )
except:
    is_torch_available = lambda: False

try:
    from uhop.backends.opencl_backend import (
        is_opencl_available,
        opencl_conv2d,
        opencl_matmul,
        opencl_relu,
    )
except:
    is_opencl_available = lambda: False

try:
    from uhop.backends.triton_backend import (
        is_triton_available,
        triton_matmul,
        triton_relu,
    )
except:
    is_triton_available = lambda: False

try:
    from uhop.backends.lite_backend import (
        is_lite_backend_available,
        lite_conv2d,
        lite_matmul,
        lite_relu,
    )
except:
    is_lite_backend_available = lambda: False

from uhop.hardware import detect_hardware


class BenchmarkResult:
    """Store results for a single benchmark run"""

    def __init__(
        self,
        op_name: str,
        backend: str,
        size: str,
        time_ms: float,
        gflops: float,
        error: float,
        success: bool,
        error_msg: str = "",
    ):
        self.op_name = op_name
        self.backend = backend
        self.size = size
        self.time_ms = time_ms
        self.gflops = gflops
        self.error = error
        self.success = success
        self.error_msg = error_msg

    def to_dict(self):
        return {
            "op": self.op_name,
            "backend": self.backend,
            "size": self.size,
            "time_ms": round(self.time_ms, 3),
            "gflops": round(self.gflops, 2),
            "error": f"{self.error:.2e}",
            "success": self.success,
            "error_msg": self.error_msg,
        }


class BenchmarkSuite:
    """Comprehensive benchmark suite for UHOP"""

    def __init__(self):
        self.results: List[BenchmarkResult] = []
        self.hardware = detect_hardware()

        # Detect available backends
        self.backends = {
            "numpy": True,  # Always available
            "torch": is_torch_available(),
            "opencl": is_opencl_available(),
            "triton": is_triton_available(),
            "lite": is_lite_backend_available(),
        }

        print("=" * 70)
        print("UHOP Comprehensive Benchmark Suite")
        print("=" * 70)
        print(f"\nHardware Detected:")
        print(f"  Vendor: {self.hardware.vendor}")
        print(f"  Kind: {self.hardware.kind}")
        print(f"  Name: {self.hardware.name}")
        print(f"\nAvailable Backends:")
        for backend, available in self.backends.items():
            status = "✓" if available else "✗"
            print(f"  {status} {backend}")
        print("=" * 70)
        print()

    def benchmark_function(self, fn, *args, warmup=2, iters=5, **kwargs):
        """Benchmark a function with warmup and multiple iterations"""
        try:
            # Warmup
            for _ in range(warmup):
                _ = fn(*args, **kwargs)

            # Benchmark
            times = []
            for _ in range(iters):
                start = time.perf_counter()
                result = fn(*args, **kwargs)
                end = time.perf_counter()
                times.append((end - start) * 1000)  # Convert to ms

            # Return median time and last result
            times.sort()
            median_time = times[len(times) // 2]
            return median_time, result, None

        except Exception as e:
            return float("inf"), None, str(e)

    def calculate_gflops(self, op_name: str, size: Tuple, time_ms: float) -> float:
        """Calculate GFLOPS for an operation"""
        if op_name == "matmul":
            M, K, N = size
            flops = 2 * M * K * N
        elif op_name == "conv2d":
            N, C_in, H, W, C_out, K = size
            H_out = H  # Assuming padding=1, stride=1
            W_out = W
            flops = N * C_out * H_out * W_out * C_in * K * K * 2
        elif op_name == "relu":
            elements = size[0]
            flops = elements  # One comparison per element
        else:
            return 0.0

        gflops = (flops / (time_ms / 1000)) / 1e9
        return gflops

    def verify_correctness(self, result, expected, rtol=1e-3, atol=1e-5):
        """Verify correctness of result vs expected"""
        try:
            result_np = np.array(result)
            expected_np = np.array(expected)

            if result_np.shape != expected_np.shape:
                return float("inf")

            error = np.max(np.abs(result_np - expected_np))
            return error
        except:
            return float("inf")

    def benchmark_matmul(self, size: Tuple[int, int, int], backends: Optional[List[str]] = None):
        """Benchmark matrix multiplication"""
        M, K, N = size
        print(f"\n📊 Benchmarking MatMul ({M}×{K} × {K}×{N})...")

        # Generate test data
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)

        # Baseline (NumPy)
        expected = np.matmul(A, B)

        backends_to_test = backends or [b for b, avail in self.backends.items() if avail]

        for backend in backends_to_test:
            if not self.backends.get(backend, False):
                continue

            try:
                if backend == "numpy":
                    time_ms, result, error_msg = self.benchmark_function(np.matmul, A, B)
                elif backend == "torch":
                    time_ms, result, error_msg = self.benchmark_function(torch_matmul, A, B)
                elif backend == "opencl":
                    time_ms, result, error_msg = self.benchmark_function(opencl_matmul, A, B)
                elif backend == "triton":
                    time_ms, result, error_msg = self.benchmark_function(triton_matmul, A, B)
                elif backend == "lite":
                    time_ms, result, error_msg = self.benchmark_function(lite_matmul, A, B, use_gpu=True)
                else:
                    continue

                if error_msg:
                    print(f"  ✗ {backend:8s} - Failed: {error_msg[:50]}")
                    self.results.append(
                        BenchmarkResult(
                            "matmul", backend, f"{M}x{K}x{N}", float("inf"), 0, float("inf"), False, error_msg
                        )
                    )
                    continue

                error = self.verify_correctness(result, expected)
                gflops = self.calculate_gflops("matmul", (M, K, N), time_ms)
                success = error < 1e-2

                status = "✓" if success else "✗"
                print(f"  {status} {backend:8s} - {time_ms:7.2f} ms  {gflops:7.1f} GFLOPS  (error: {error:.2e})")

                self.results.append(BenchmarkResult("matmul", backend, f"{M}x{K}x{N}", time_ms, gflops, error, success))

            except Exception as e:
                print(f"  ✗ {backend:8s} - Exception: {str(e)[:50]}")
                self.results.append(
                    BenchmarkResult("matmul", backend, f"{M}x{K}x{N}", float("inf"), 0, float("inf"), False, str(e))
                )

    def benchmark_relu(self, size: int, backends: Optional[List[str]] = None):
        """Benchmark ReLU activation"""
        print(f"\n📊 Benchmarking ReLU ({size} elements)...")

        # Generate test data
        X = np.random.randn(size).astype(np.float32)
        expected = np.maximum(0, X)

        backends_to_test = backends or [b for b, avail in self.backends.items() if avail]

        for backend in backends_to_test:
            if not self.backends.get(backend, False):
                continue

            try:
                if backend == "numpy":
                    time_ms, result, error_msg = self.benchmark_function(lambda x: np.maximum(0, x), X)
                elif backend == "torch":
                    time_ms, result, error_msg = self.benchmark_function(torch_relu, X)
                elif backend == "opencl":
                    time_ms, result, error_msg = self.benchmark_function(opencl_relu, X)
                elif backend == "triton":
                    time_ms, result, error_msg = self.benchmark_function(triton_relu, X)
                elif backend == "lite":
                    time_ms, result, error_msg = self.benchmark_function(lite_relu, X, use_gpu=True)
                else:
                    continue

                if error_msg:
                    print(f"  ✗ {backend:8s} - Failed: {error_msg[:50]}")
                    self.results.append(
                        BenchmarkResult("relu", backend, str(size), float("inf"), 0, float("inf"), False, error_msg)
                    )
                    continue

                error = self.verify_correctness(result, expected)
                gflops = self.calculate_gflops("relu", (size,), time_ms)
                success = error < 1e-5

                status = "✓" if success else "✗"
                print(f"  {status} {backend:8s} - {time_ms:7.2f} ms  {gflops:7.3f} GFLOPS  (error: {error:.2e})")

                self.results.append(BenchmarkResult("relu", backend, str(size), time_ms, gflops, error, success))

            except Exception as e:
                print(f"  ✗ {backend:8s} - Exception: {str(e)[:50]}")
                self.results.append(
                    BenchmarkResult("relu", backend, str(size), float("inf"), 0, float("inf"), False, str(e))
                )

    def benchmark_conv2d(self, size: Tuple, backends: Optional[List[str]] = None):
        """Benchmark 2D convolution"""
        N, C_in, H, W, C_out, K = size
        print(f"\n📊 Benchmarking Conv2D ({N}×{C_in}×{H}×{W} * {C_out}×{C_in}×{K}×{K})...")

        # Generate test data
        input_data = np.random.randn(N, C_in, H, W).astype(np.float32)
        weight = np.random.randn(C_out, C_in, K, K).astype(np.float32)

        # Baseline (skip for now, just test backends)
        expected = None

        backends_to_test = backends or ["torch", "opencl", "lite"]  # Conv2D not in all backends

        for backend in backends_to_test:
            if not self.backends.get(backend, False):
                continue

            try:
                if backend == "torch":
                    time_ms, result, error_msg = self.benchmark_function(
                        torch_conv2d, input_data, weight, stride=1, padding=1
                    )
                elif backend == "opencl":
                    time_ms, result, error_msg = self.benchmark_function(
                        opencl_conv2d, input_data, weight, stride=1, padding=1
                    )
                elif backend == "lite":
                    time_ms, result, error_msg = self.benchmark_function(
                        lite_conv2d, input_data, weight, stride=1, padding=1, use_gpu=True
                    )
                else:
                    continue

                if expected is None and result is not None and error_msg is None:
                    expected = result  # Use first successful result as baseline

                if error_msg:
                    print(f"  ✗ {backend:8s} - Failed: {error_msg[:50]}")
                    self.results.append(
                        BenchmarkResult(
                            "conv2d", backend, f"{N}x{C_in}x{H}x{W}", float("inf"), 0, float("inf"), False, error_msg
                        )
                    )
                    continue

                error = self.verify_correctness(result, expected) if expected is not None else 0.0
                gflops = self.calculate_gflops("conv2d", size, time_ms)
                success = error < 1e-2 or expected is None

                status = "✓" if success else "✗"
                print(f"  {status} {backend:8s} - {time_ms:7.2f} ms  {gflops:7.1f} GFLOPS  (error: {error:.2e})")

                self.results.append(
                    BenchmarkResult("conv2d", backend, f"{N}x{C_in}x{H}x{W}", time_ms, gflops, error, success)
                )

            except Exception as e:
                print(f"  ✗ {backend:8s} - Exception: {str(e)[:50]}")
                self.results.append(
                    BenchmarkResult(
                        "conv2d", backend, f"{N}x{C_in}x{H}x{W}", float("inf"), 0, float("inf"), False, str(e)
                    )
                )

    def run_all_benchmarks(self, backends: Optional[List[str]] = None):
        """Run all benchmarks with various sizes"""
        print("\n" + "=" * 70)
        print("Running Comprehensive Benchmark Suite")
        print("=" * 70)

        # MatMul benchmarks - various sizes
        for size in [(256, 256, 256), (512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048)]:
            self.benchmark_matmul(size, backends)

        # ReLU benchmarks - various sizes
        for size in [100_000, 1_000_000, 10_000_000]:
            self.benchmark_relu(size, backends)

        # Conv2D benchmarks - various sizes
        for size in [(1, 3, 32, 32, 16, 3), (1, 16, 64, 64, 32, 3), (4, 32, 128, 128, 64, 3)]:
            self.benchmark_conv2d(size, backends)

        print("\n" + "=" * 70)
        print("Benchmark Suite Complete")
        print("=" * 70)

    def generate_summary(self):
        """Generate summary statistics"""
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)

        # Group by operation and backend
        op_backend_results = defaultdict(list)
        for result in self.results:
            if result.success and result.time_ms < float("inf"):
                key = (result.op_name, result.backend)
                op_backend_results[key].append(result.gflops)

        # Print average GFLOPS by operation and backend
        for op in ["matmul", "relu", "conv2d"]:
            print(f"\n{op.upper()}:")
            backend_gflops = {}
            for backend in self.backends.keys():
                key = (op, backend)
                if key in op_backend_results:
                    avg_gflops = np.mean(op_backend_results[key])
                    backend_gflops[backend] = avg_gflops

            # Sort by performance
            sorted_backends = sorted(backend_gflops.items(), key=lambda x: x[1], reverse=True)

            for backend, gflops in sorted_backends:
                print(f"  {backend:10s}: {gflops:8.1f} GFLOPS (avg)")

    def save_json(self, filename: str):
        """Save results to JSON file"""
        data = {
            "hardware": {
                "vendor": self.hardware.vendor,
                "kind": self.hardware.kind,
                "name": self.hardware.name,
            },
            "backends": {k: bool(v) for k, v in self.backends.items()},
            "results": [r.to_dict() for r in self.results],
        }

        with open(filename, "w") as f:
            json.dump(data, f, indent=2)

        print(f"\n✓ Results saved to {filename}")

    def generate_html_report(self, filename: str):
        """Generate HTML report with tables and charts"""
        html = """
<!DOCTYPE html>
<html>
<head>
    <title>UHOP Benchmark Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; }
        h1 { color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }
        h2 { color: #555; margin-top: 30px; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #4CAF50; color: white; }
        tr:hover { background-color: #f5f5f5; }
        .success { color: green; }
        .failure { color: red; }
        .hardware { background: #e8f5e9; padding: 15px; border-radius: 5px; margin: 20px 0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 UHOP Comprehensive Benchmark Report</h1>

        <div class="hardware">
            <h3>Hardware Information</h3>
            <p><strong>Device:</strong> {device}</p>
            <p><strong>Memory:</strong> {memory} GB</p>
            <p><strong>Compute Units:</strong> {compute_units}</p>
        </div>

        <h2>Backend Availability</h2>
        <ul>
"""

        # Add hardware info
        html = html.format(device=self.hardware.name or "Unknown", memory="N/A", compute_units="N/A")

        # Add backend availability
        for backend, available in self.backends.items():
            status = "✓ Available" if available else "✗ Not Available"
            html += f"            <li><strong>{backend}:</strong> {status}</li>\n"

        html += """
        </ul>

        <h2>Benchmark Results</h2>
"""

        # Add results table for each operation
        for op in ["matmul", "relu", "conv2d"]:
            op_results = [r for r in self.results if r.op_name == op]
            if not op_results:
                continue

            html += f"""
        <h3>{op.upper()} Performance</h3>
        <table>
            <tr>
                <th>Backend</th>
                <th>Size</th>
                <th>Time (ms)</th>
                <th>GFLOPS</th>
                <th>Error</th>
                <th>Status</th>
            </tr>
"""

            for result in op_results:
                status_class = "success" if result.success else "failure"
                status_text = "✓ Pass" if result.success else "✗ Fail"

                html += f"""
            <tr>
                <td>{result.backend}</td>
                <td>{result.size}</td>
                <td>{result.time_ms:.2f}</td>
                <td>{result.gflops:.1f}</td>
                <td>{result.error:.2e}</td>
                <td class="{status_class}">{status_text}</td>
            </tr>
"""

            html += """
        </table>
"""

        html += """
    </div>
</body>
</html>
"""

        with open(filename, "w") as f:
            f.write(html)

        print(f"✓ HTML report saved to {filename}")


def main():
    parser = argparse.ArgumentParser(description="UHOP Comprehensive Benchmark Suite")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    parser.add_argument("--op", choices=["matmul", "relu", "conv2d"], help="Benchmark specific operation")
    parser.add_argument("--backends", help="Comma-separated list of backends (e.g., torch,opencl)")
    parser.add_argument("--report", default="benchmark_report.html", help="Output HTML report filename")
    parser.add_argument("--json", default="benchmark_results.json", help="Output JSON results filename")

    args = parser.parse_args()

    # Parse backends
    backends = None
    if args.backends:
        backends = [b.strip() for b in args.backends.split(",")]

    # Create benchmark suite
    suite = BenchmarkSuite()

    # Run benchmarks
    if args.all or not args.op:
        suite.run_all_benchmarks(backends)
    else:
        if args.op == "matmul":
            for size in [(512, 512, 512), (1024, 1024, 1024)]:
                suite.benchmark_matmul(size, backends)
        elif args.op == "relu":
            for size in [1_000_000, 10_000_000]:
                suite.benchmark_relu(size, backends)
        elif args.op == "conv2d":
            for size in [(1, 3, 64, 64, 16, 3)]:
                suite.benchmark_conv2d(size, backends)

    # Generate outputs
    suite.generate_summary()
    suite.save_json(args.json)
    suite.generate_html_report(args.report)

    print(f"\n✓ Benchmark complete! Open {args.report} in your browser to view results.")


if __name__ == "__main__":
    main()
