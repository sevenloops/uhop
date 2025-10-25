"""
Comprehensive Multi-Backend Operator Benchmarking Suite for UHOP.

Compares performance across:
- PyTorch baseline (naive)
- UHOP manual kernels
- UHOP vendor kernels (cuDNN, MIOpen, etc.)
- UHOP auto-generated kernels

Usage:
    python -m uhop.benchmarks.heavy_ops_bench --ops matmul,conv2d --backends cuda,cpu
"""
import argparse
import time
import statistics
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import logging

import numpy as np
import torch

from uhop.core.op_registry import get_registry
from uhop.backends.base import get_backend_manager, KernelSource

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    op_name: str
    backend: str
    kernel_source: str  # manual, vendor, autogen, fallback
    input_shape: Tuple[int, ...]
    mean_time_ms: float
    median_time_ms: float
    min_time_ms: float
    max_time_ms: float
    std_time_ms: float
    iterations: int
    speedup_vs_baseline: Optional[float] = None
    throughput_gflops: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


class HeavyOpsBenchmark:
    """
    Benchmark suite for heavy deep learning operators.
    """
    
    def __init__(self, warmup: int = 3, iterations: int = 10):
        self.warmup = warmup
        self.iterations = iterations
        self.backend_manager = get_backend_manager()
        self.registry = get_registry()
        self.results: List[BenchmarkResult] = []
        
        # Initialize all backends
        self.backend_manager.initialize_all()
    
    def _sync_device(self, backend_name: str):
        """Synchronize device for accurate timing."""
        backend = self.backend_manager.get_backend(backend_name)
        if backend:
            backend._synchronize()
    
    def _measure_time(self, fn, backend_name: str) -> Dict[str, float]:
        """Measure execution time with proper synchronization."""
        times = []
        
        # Warmup
        for _ in range(self.warmup):
            fn()
            self._sync_device(backend_name)
        
        # Benchmark
        for _ in range(self.iterations):
            start = time.perf_counter()
            fn()
            self._sync_device(backend_name)
            times.append((time.perf_counter() - start) * 1000)  # Convert to ms
        
        return {
            "mean": statistics.mean(times),
            "median": statistics.median(times),
            "min": min(times),
            "max": max(times),
            "std": statistics.stdev(times) if len(times) > 1 else 0.0,
        }
    
    def benchmark_matmul(
        self,
        m: int = 512,
        n: int = 512,
        k: int = 512,
        backends: Optional[List[str]] = None
    ) -> List[BenchmarkResult]:
        """
        Benchmark matrix multiplication across backends.
        
        Args:
            m, n, k: Matrix dimensions (M x K) @ (K x N) = (M x N)
            backends: List of backends to test (None = all available)
        """
        logger.info(f"Benchmarking matmul with shape ({m}, {k}) @ ({k}, {n})")
        
        # Prepare inputs
        A = torch.randn(m, k, dtype=torch.float32)
        B = torch.randn(k, n, dtype=torch.float32)
        
        # PyTorch CPU baseline
        baseline_time = self._measure_time(
            lambda: torch.matmul(A.cpu(), B.cpu()),
            "cpu"
        )
        baseline_mean = baseline_time["mean"]
        
        results = []
        
        # Test each backend
        if backends is None:
            backends = self.backend_manager.list_available_backends()
        
        for backend_name in backends:
            backend = self.backend_manager.get_backend(backend_name)
            if backend is None or not backend.capabilities.available:
                logger.debug(f"Skipping unavailable backend: {backend_name}")
                continue
            
            kernel_info = backend.get_kernel("matmul")
            if kernel_info is None:
                logger.debug(f"No matmul kernel for backend: {backend_name}")
                continue
            
            # Prepare backend-specific inputs
            if backend_name in ["cuda", "cudnn", "rocm"]:
                A_dev = A.cuda()
                B_dev = B.cuda()
            else:
                A_dev = A.cpu()
                B_dev = B.cpu()
            
            # Benchmark
            timing = self._measure_time(
                lambda: kernel_info.kernel_fn(A_dev, B_dev),
                backend_name
            )
            
            # Calculate FLOPS (2 * M * N * K operations)
            flops = 2 * m * n * k
            throughput = (flops / (timing["mean"] / 1000)) / 1e9  # GFLOPS
            
            result = BenchmarkResult(
                op_name="matmul",
                backend=backend_name,
                kernel_source=kernel_info.source.value,
                input_shape=(m, k, n),
                mean_time_ms=timing["mean"],
                median_time_ms=timing["median"],
                min_time_ms=timing["min"],
                max_time_ms=timing["max"],
                std_time_ms=timing["std"],
                iterations=self.iterations,
                speedup_vs_baseline=baseline_mean / timing["mean"],
                throughput_gflops=throughput
            )
            
            results.append(result)
            logger.info(
                f"  {backend_name}/{kernel_info.source.value}: "
                f"{timing['mean']:.2f}ms ({throughput:.1f} GFLOPS, "
                f"{result.speedup_vs_baseline:.2f}x baseline)"
            )
        
        self.results.extend(results)
        return results
    
    def benchmark_conv2d(
        self,
        batch: int = 32,
        in_channels: int = 3,
        out_channels: int = 64,
        height: int = 224,
        width: int = 224,
        kernel_size: int = 3,
        backends: Optional[List[str]] = None
    ) -> List[BenchmarkResult]:
        """
        Benchmark 2D convolution across backends.
        """
        logger.info(
            f"Benchmarking conv2d: "
            f"input=({batch},{in_channels},{height},{width}), "
            f"weight=({out_channels},{in_channels},{kernel_size},{kernel_size})"
        )
        
        # Prepare inputs
        input = torch.randn(batch, in_channels, height, width, dtype=torch.float32)
        weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size, dtype=torch.float32)
        
        # PyTorch CPU baseline
        baseline_time = self._measure_time(
            lambda: torch.nn.functional.conv2d(input.cpu(), weight.cpu(), padding=1),
            "cpu"
        )
        baseline_mean = baseline_time["mean"]
        
        results = []
        
        if backends is None:
            backends = self.backend_manager.list_available_backends()
        
        for backend_name in backends:
            backend = self.backend_manager.get_backend(backend_name)
            if backend is None or not backend.capabilities.available:
                continue
            
            kernel_info = backend.get_kernel("conv2d")
            if kernel_info is None:
                continue
            
            # Prepare inputs
            if backend_name in ["cuda", "cudnn", "rocm"]:
                input_dev = input.cuda()
                weight_dev = weight.cuda()
            else:
                input_dev = input.cpu()
                weight_dev = weight.cpu()
            
            # Benchmark
            timing = self._measure_time(
                lambda: kernel_info.kernel_fn(input_dev, weight_dev, padding=1),
                backend_name
            )
            
            result = BenchmarkResult(
                op_name="conv2d",
                backend=backend_name,
                kernel_source=kernel_info.source.value,
                input_shape=(batch, in_channels, height, width, out_channels, kernel_size),
                mean_time_ms=timing["mean"],
                median_time_ms=timing["median"],
                min_time_ms=timing["min"],
                max_time_ms=timing["max"],
                std_time_ms=timing["std"],
                iterations=self.iterations,
                speedup_vs_baseline=baseline_mean / timing["mean"]
            )
            
            results.append(result)
            logger.info(
                f"  {backend_name}/{kernel_info.source.value}: "
                f"{timing['mean']:.2f}ms ({result.speedup_vs_baseline:.2f}x baseline)"
            )
        
        self.results.extend(results)
        return results
    
    def benchmark_activation(
        self,
        op_name: str,
        size: int = 1024 * 1024,
        backends: Optional[List[str]] = None
    ) -> List[BenchmarkResult]:
        """
        Benchmark activation functions (relu, gelu, silu, etc.).
        """
        logger.info(f"Benchmarking {op_name} with size {size}")
        
        # Prepare input
        x = torch.randn(size, dtype=torch.float32)
        
        # PyTorch CPU baseline
        if op_name == "relu":
            baseline_fn = lambda: torch.relu(x.cpu())
        elif op_name == "gelu":
            baseline_fn = lambda: torch.nn.functional.gelu(x.cpu())
        elif op_name == "silu":
            baseline_fn = lambda: torch.nn.functional.silu(x.cpu())
        else:
            logger.warning(f"Unknown activation: {op_name}")
            return []
        
        baseline_time = self._measure_time(baseline_fn, "cpu")
        baseline_mean = baseline_time["mean"]
        
        results = []
        
        if backends is None:
            backends = self.backend_manager.list_available_backends()
        
        for backend_name in backends:
            backend = self.backend_manager.get_backend(backend_name)
            if backend is None or not backend.capabilities.available:
                continue
            
            kernel_info = backend.get_kernel(op_name)
            if kernel_info is None:
                continue
            
            # Prepare input
            if backend_name in ["cuda", "cudnn", "rocm"]:
                x_dev = x.cuda()
            else:
                x_dev = x.cpu()
            
            # Benchmark
            timing = self._measure_time(
                lambda: kernel_info.kernel_fn(x_dev),
                backend_name
            )
            
            result = BenchmarkResult(
                op_name=op_name,
                backend=backend_name,
                kernel_source=kernel_info.source.value,
                input_shape=(size,),
                mean_time_ms=timing["mean"],
                median_time_ms=timing["median"],
                min_time_ms=timing["min"],
                max_time_ms=timing["max"],
                std_time_ms=timing["std"],
                iterations=self.iterations,
                speedup_vs_baseline=baseline_mean / timing["mean"]
            )
            
            results.append(result)
            logger.info(
                f"  {backend_name}/{kernel_info.source.value}: "
                f"{timing['mean']:.2f}ms ({result.speedup_vs_baseline:.2f}x baseline)"
            )
        
        self.results.extend(results)
        return results
    
    def run_full_suite(self, backends: Optional[List[str]] = None):
        """Run the complete benchmark suite."""
        logger.info("=" * 80)
        logger.info("UHOP Multi-Backend Operator Benchmark Suite")
        logger.info("=" * 80)
        
        # List available backends
        available = self.backend_manager.list_available_backends()
        logger.info(f"Available backends: {', '.join(available)}")
        
        if backends:
            logger.info(f"Testing backends: {', '.join(backends)}")
        
        # Matrix multiplication benchmarks
        logger.info("\n" + "=" * 80)
        logger.info("Matrix Multiplication Benchmarks")
        logger.info("=" * 80)
        self.benchmark_matmul(256, 256, 256, backends)
        self.benchmark_matmul(512, 512, 512, backends)
        self.benchmark_matmul(1024, 1024, 1024, backends)
        
        # Conv2D benchmarks
        logger.info("\n" + "=" * 80)
        logger.info("Conv2D Benchmarks")
        logger.info("=" * 80)
        self.benchmark_conv2d(16, 3, 64, 224, 224, 3, backends)
        self.benchmark_conv2d(32, 64, 128, 56, 56, 3, backends)
        
        # Activation benchmarks
        logger.info("\n" + "=" * 80)
        logger.info("Activation Function Benchmarks")
        logger.info("=" * 80)
        self.benchmark_activation("relu", 1024*1024, backends)
        self.benchmark_activation("gelu", 1024*1024, backends)
        self.benchmark_activation("silu", 1024*1024, backends)
        
        logger.info("\n" + "=" * 80)
        logger.info("Benchmark Suite Complete")
        logger.info("=" * 80)
    
    def generate_report(self, output_path: Optional[Path] = None) -> Dict:
        """Generate a comprehensive benchmark report."""
        report = {
            "summary": {
                "total_benchmarks": len(self.results),
                "backends_tested": list(set(r.backend for r in self.results)),
                "operators_tested": list(set(r.op_name for r in self.results)),
            },
            "results": [r.to_dict() for r in self.results],
            "best_performers": {},
        }
        
        # Find best performer for each op
        ops = set(r.op_name for r in self.results)
        for op in ops:
            op_results = [r for r in self.results if r.op_name == op]
            if op_results:
                best = min(op_results, key=lambda r: r.mean_time_ms)
                report["best_performers"][op] = {
                    "backend": best.backend,
                    "source": best.kernel_source,
                    "time_ms": best.mean_time_ms,
                    "speedup": best.speedup_vs_baseline,
                }
        
        # Save to file if requested
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Report saved to: {output_path}")
        
        return report
    
    def print_summary(self):
        """Print a human-readable summary of results."""
        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)
        
        ops = sorted(set(r.op_name for r in self.results))
        for op in ops:
            print(f"\n{op.upper()}:")
            op_results = sorted(
                [r for r in self.results if r.op_name == op],
                key=lambda r: r.mean_time_ms
            )
            
            for r in op_results:
                print(
                    f"  {r.backend:10s} ({r.kernel_source:10s}): "
                    f"{r.mean_time_ms:8.2f}ms "
                    f"(±{r.std_time_ms:6.2f}ms, {r.speedup_vs_baseline:5.2f}x)"
                )


def main():
    parser = argparse.ArgumentParser(description="UHOP Multi-Backend Benchmark Suite")
    parser.add_argument(
        "--ops",
        type=str,
        help="Comma-separated list of operators to benchmark (default: all)"
    )
    parser.add_argument(
        "--backends",
        type=str,
        help="Comma-separated list of backends to test (default: all available)"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of warmup iterations"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of benchmark iterations"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON file for results"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format='%(levelname)s: %(message)s'
    )
    
    # Parse backend list
    backends = None
    if args.backends:
        backends = [b.strip() for b in args.backends.split(',')]
    
    # Run benchmarks
    bench = HeavyOpsBenchmark(warmup=args.warmup, iterations=args.iterations)
    
    if args.ops:
        # Benchmark specific ops
        ops = [o.strip() for o in args.ops.split(',')]
        for op in ops:
            if op == "matmul":
                bench.benchmark_matmul(backends=backends)
            elif op == "conv2d":
                bench.benchmark_conv2d(backends=backends)
            elif op in ["relu", "gelu", "silu"]:
                bench.benchmark_activation(op, backends=backends)
    else:
        # Run full suite
        bench.run_full_suite(backends=backends)
    
    # Print summary
    bench.print_summary()
    
    # Generate report
    output_path = Path(args.output) if args.output else None
    report = bench.generate_report(output_path)
    
    print(f"\nTotal benchmarks: {len(bench.results)}")
    print(f"Backends tested: {', '.join(report['summary']['backends_tested'])}")


if __name__ == "__main__":
    main()
