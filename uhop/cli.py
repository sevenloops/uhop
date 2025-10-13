import time
import click
import numpy as np
from rich.console import Console

from .hardware import detect_hardware
from .backends import (
    is_torch_available,
    is_triton_available,
    is_opencl_available,
)
from .backends.opencl_backend import set_opencl_device as _set_opencl_device
from . import optimize

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
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON.")
@click.option("--ocl-device", type=int, default=None, help="OpenCL GPU device index override (across all platforms).")
def info(as_json: bool, ocl_device: int | None):
    """Display detected hardware and backend availability."""
    import json as _json

    if ocl_device is not None:
        try:
            _set_opencl_device(ocl_device)
        except Exception as e:
            console.print(f"[yellow]Warning:[/yellow] could not set OpenCL device index {ocl_device}: {e}")

    console.print("[bold cyan]UHOP Hardware Report[/bold cyan]")
    hw_str = _hardware_summary()
    if as_json:
        from .hardware import detect_hardware
        hw = detect_hardware()
        console.print(_json.dumps({
            "vendor": hw.vendor,
            "kind": hw.kind,
            "name": hw.name,
            "details": hw.details,
            "torch_available": is_torch_available(),
            "triton_available": is_triton_available(),
            "opencl_available": is_opencl_available(),
        }, indent=2))
        return
    console.print(hw_str)

    # Torch details
    console.print("\n[bold]Torch[/bold]")
    try:
        import torch  # type: ignore
        console.print(f"- torch version: {getattr(torch, '__version__', 'unknown')}")
        has_cuda = hasattr(torch, 'cuda') and torch.cuda.is_available()
        console.print(f"- CUDA available: {has_cuda}")
        if has_cuda:
            try:
                device_count = torch.cuda.device_count()
                console.print(f"- CUDA device count: {device_count}")
                for i in range(device_count):
                    name = torch.cuda.get_device_name(i)
                    props = torch.cuda.get_device_properties(i)
                    total_mem_gb = getattr(props, 'total_memory', 0) / (1024**3)
                    console.print(f"  - [{i}] {name} ({total_mem_gb:.2f} GB)")
            except Exception as e:
                console.print(f"  (error querying CUDA devices: {e})")
    except Exception as e:
        console.print(f"- torch not importable: {e}")

    # OpenCL details
    console.print("\n[bold]OpenCL[/bold]")
    try:
        import pyopencl as cl  # type: ignore

        def _dtype(dev_type: int) -> str:
            # Prefer the most informative type label
            if dev_type & cl.device_type.GPU:
                return "GPU"
            if dev_type & cl.device_type.CPU:
                return "CPU"
            if dev_type & cl.device_type.ACCELERATOR:
                return "ACCELERATOR"
            if dev_type & cl.device_type.CUSTOM:
                return "CUSTOM"
            if dev_type & cl.device_type.DEFAULT:
                return "DEFAULT"
            return str(dev_type)

        def _fmt_bytes(n: int) -> str:
            try:
                gb = n / (1024**3)
                if gb >= 1:
                    return f"{gb:.2f} GB"
                mb = n / (1024**2)
                if mb >= 1:
                    return f"{mb:.1f} MB"
                kb = n / 1024
                return f"{kb:.0f} KB"
            except Exception:
                return str(n)

        plats = cl.get_platforms()
        console.print(f"- platforms: {len(plats)}")
        for pi, p in enumerate(plats):
            console.print(f"  - [{pi}] {p.name} (vendor={p.vendor}, version={p.version})")
            for di, d in enumerate(p.get_devices()):
                try:
                    console.print(
                        f"      * [{di}] {_dtype(d.type)} | name={d.name} | vendor={d.vendor} | version={d.version} | CUs={d.max_compute_units} | maxWG={d.max_work_group_size} | global={_fmt_bytes(d.global_mem_size)} | local={_fmt_bytes(d.local_mem_size)}"
                    )
                except Exception as e:
                    console.print(f"      * [{di}] (error reading device info: {e})")
    except Exception as e:
        console.print(f"- pyopencl not importable or no platforms: {e}")


@main.command()
@click.option("--size", "size", default=192, show_default=True, help="Matrix size N for NxN matmul demo.")
@click.option("--iters", default=3, show_default=True, help="Median timing iterations for UHOP path.")
@click.option("--ocl-device", type=int, default=None, help="OpenCL GPU device index override (across all platforms).")
def demo(size: int, iters: int, ocl_device: int | None):
    """Run a short demo: device info + Naive Python vs UHOP (GPU-preferred) matmul benchmark."""
    console.print("[bold cyan]UHOP Demo[/bold cyan]")
    if ocl_device is not None:
        try:
            _set_opencl_device(ocl_device)
        except Exception as e:
            console.print(f"[yellow]Warning:[/yellow] could not set OpenCL device index {ocl_device}: {e}")
    console.print(_hardware_summary())

    # Naive triple-loop Python baseline (very slow, but proves UHOP acceleration clearly)
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

    def _med(run, iters=iters):
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            run()
            times.append(time.perf_counter() - t0)
        return float(np.median(times))

    t_uhop = _med(lambda: matmul_np(A, B))
    t_naive = _med(lambda: matmul_naive(A, B), iters=1)  # single iter; it's very slow

    console.print(f"UHOP (optimized over naive): [bold]{t_uhop:.6f} s[/bold]")
    console.print(f"Naive Python baseline     : [bold]{t_naive:.6f} s[/bold]")
    if t_uhop < t_naive:
        console.print("[green]UHOP wins ✅[/green]")
    else:
        console.print("[yellow]Baseline was faster in this config. Try larger size or check GPU drivers.[/yellow]")


@main.command()
@click.argument("function_path")
def optimize_func(function_path):
    """Optimize a Python function via @uhop.optimize by importing it."""
    console.print(f"[green]Optimizing function:[/green] {function_path}")
    __import__(function_path)
    console.print("[bold green]Optimization complete![/bold green]")

if __name__ == "__main__":
    main()
