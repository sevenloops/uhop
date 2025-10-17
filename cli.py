import time
import click
import numpy as np
from rich.console import Console

from uhop.hardware import detect_hardware
from uhop.backends import (
    is_torch_available,
    is_triton_available,
    is_opencl_available,
)
from uhop import optimize

console = Console()


def _hardware_summary() -> str:
    hw = detect_hardware()
    lines = [
        f"Vendor: {hw.vendor}",
        f"Kind: {hw.kind}",
        f"Name: {hw.name}",
        f"Torch available: {is_torch_available()}",
        f"Triton available: {is_triton_available()}",
        f"OpenCL available: {is_opencl_available()}",
    ]
    return "\n".join(lines)


@click.group()
def main():
    """UHOP CLI — AI-Powered Universal Hardware Optimizer."""
    pass


@main.command()
def info():
    """Display detected hardware and backend availability."""
    console.print("[bold cyan]UHOP Hardware Report[/bold cyan]")
    console.print(_hardware_summary())


@main.command()
@click.option(
    "--size",
    "size",
    default=192,
    show_default=True,
    help="Matrix size N for NxN matmul demo.",
)
def demo(size: int):
    """Run device info + Naive Python vs UHOP matmul benchmark."""
    console.print("[bold cyan]UHOP Demo[/bold cyan]")
    console.print(_hardware_summary())

    # Naive triple-loop Python baseline (very slow) shows UHOP clearly
    def matmul_naive(A: np.ndarray, B: np.ndarray) -> np.ndarray:
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

    # UHOP-optimized wrapper decorating NumPy baseline
    @optimize("matmul")
    def matmul_np(A, B):
        return np.array(A) @ np.array(B)

    rng = np.random.default_rng(0)
    A = rng.random((size, size), dtype=np.float32)
    B = rng.random((size, size), dtype=np.float32)

    # Warm-up UHOP path (trigger backend selection/caching)
    _ = matmul_np(A, B)

    def _med(run, iters=3):
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            run()
            times.append(time.perf_counter() - t0)
        return float(np.median(times))

    t_uhop = _med(lambda: matmul_np(A, B))
    # single iter; it's very slow
    t_naive = _med(lambda: matmul_naive(A, B), iters=1)

    console.print(f"UHOP (optimized over naive): [bold]{t_uhop:.6f} s[/bold]")
    console.print(f"Naive Python baseline     : [bold]{t_naive:.6f} s[/bold]")
    if t_uhop < t_naive:
        console.print("[green]UHOP wins ✅[/green]")
    else:
        console.print(
            "[yellow]Baseline faster. Try larger size or check GPU "
            "drivers.[/yellow]"
        )


@main.command()
@click.argument("function_path")
def optimize_func(function_path):
    """Optimize a Python function via @uhop.optimize by importing it."""
    console.print(f"[green]Optimizing function:[/green] {function_path}")
    __import__(function_path)
    console.print("[bold green]Optimization complete![/bold green]")
