import time
import os
from pathlib import Path
import click
import numpy as np
from rich.console import Console

from .hardware import detect_hardware
from .backends import (
    is_torch_available,
    is_triton_available,
    is_opencl_available,
)

# Conditional imports for OpenCL functions
try:
    if is_opencl_available():
        from .backends.opencl_backend import set_opencl_device as _set_opencl_device
        from .backends.opencl_backend import opencl_conv2d_relu as _ocl_conv2d_relu
    else:
        _set_opencl_device = None
        _ocl_conv2d_relu = None
except ImportError:
    _set_opencl_device = None
    _ocl_conv2d_relu = None
from . import optimize
from .ai_codegen import AICodegen
from .cache import UhopCache as _UhopCache
from datetime import datetime as _dt
import json as _json

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


def _load_env():
    # Load environment variables from a .env file if present
    root = Path(__file__).resolve().parents[1]  # repo root (uhop/)
    candidates = [
        Path.cwd() / ".env",
        root / ".env",
    ]
    for p in candidates:
        try:
            if p.exists():
                for line in p.read_text().splitlines():
                    s = line.strip()
                    if not s or s.startswith("#"):
                        continue
                    if "=" in s:
                        k, v = s.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip('"').strip("'")
                        if k and v and k not in os.environ:
                            os.environ[k] = v
        except Exception:
            continue


_load_env()


@click.group()
@click.option(
    "--strict-validate",
    is_flag=True,
    help=(
        "Tighten tolerances and gate AI kernels with validation. "
        "Equivalent to UHOP_STRICT_VALIDATE=1"
    ),
)
def main(strict_validate: bool):
    """UHOP CLI — AI-Powered Universal Hardware Optimizer."""
    if strict_validate:
        os.environ["UHOP_STRICT_VALIDATE"] = "1"


@main.command()
@click.option(
    "--json", "as_json", is_flag=True, help="Emit machine-readable JSON."
)
@click.option(
    "--ocl-device",
    type=int,
    default=None,
    help="OpenCL GPU device index override (across all platforms).",
)
def info(as_json: bool, ocl_device: int | None):
    """Display detected hardware and backend availability."""
    import json as _json

    if ocl_device is not None:
        try:
            _set_opencl_device(ocl_device)
        except Exception as e:
            console.print(
                f"[yellow]Warning:[/yellow] could not set OpenCL device index "
                f"{ocl_device}: {e}"
            )

    if not as_json:
        console.print("[bold cyan]UHOP Hardware Report[/bold cyan]")
    hw_str = _hardware_summary()
    if as_json:
        from .hardware import detect_hardware
        hw = detect_hardware()
        mps_avail = None
        torch_pref = None
        try:
            import torch  # type: ignore
            mps_avail = bool(
                getattr(torch.backends, 'mps', None)
                and torch.backends.mps.is_available()
            )
            try:
                from .backends.torch_backend import (
                    _torch_device_preference as _pref,
                )
                dev = _pref()
                torch_pref = (
                    getattr(dev, 'type', None) if dev is not None else None
                )
            except Exception:
                torch_pref = None
        except Exception:
            mps_avail = None
        console.print(_json.dumps({
            "vendor": hw.vendor,
            "kind": hw.kind,
            "name": hw.name,
            "details": hw.details,
            "torch_available": is_torch_available(),
            "torch_mps_available": mps_avail,
            "torch_preferred_device": torch_pref,
            "triton_available": is_triton_available(),
            "opencl_available": is_opencl_available(),
        }, indent=2))
        return
    console.print(hw_str)

    # Torch details
    console.print("\n[bold]Torch[/bold]")
    try:
        import torch  # type: ignore
        console.print(
            f"- torch version: {getattr(torch, '__version__', 'unknown')}"
        )
        has_cuda = hasattr(torch, 'cuda') and torch.cuda.is_available()
        console.print(f"- CUDA available: {has_cuda}")
        # Apple Metal Performance Shaders (MPS)
        try:
            mps_ok = bool(
                getattr(torch.backends, 'mps', None)
                and torch.backends.mps.is_available()
            )
        except Exception:
            mps_ok = False
        console.print(f"- MPS available: {mps_ok}")
        # UHOP's torch device preference (cuda > mps > cpu)
        try:
            from .backends.torch_backend import (
                _torch_device_preference as _pref,
            )
            dev = _pref()
            pref = getattr(dev, 'type', None) if dev is not None else None
            console.print(f"- UHOP preferred device: {pref}")
        except Exception:
            pass
        if has_cuda:
            try:
                device_count = torch.cuda.device_count()
                console.print(f"- CUDA device count: {device_count}")
                for i in range(device_count):
                    name = torch.cuda.get_device_name(i)
                    props = torch.cuda.get_device_properties(i)
                    total_mem_gb = getattr(props, 'total_memory', 0) / (
                        1024**3
                    )
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
            console.print(
                f"  - [{pi}] {p.name} (vendor={p.vendor}, version={p.version})"
            )
            for di, d in enumerate(p.get_devices()):
                try:
                    info1 = (
                        f"      * [{di}] {_dtype(d.type)} | name={d.name}"
                    )
                    info2 = (
                        f"vendor={d.vendor} | version={d.version}"
                    )
                    info3 = (
                        f"CUs={d.max_compute_units} | "
                        f"maxWG={d.max_work_group_size}"
                    )
                    info4 = (
                        f"global={_fmt_bytes(d.global_mem_size)} | "
                        f"local={_fmt_bytes(d.local_mem_size)}"
                    )
                    console.print(info1)
                    console.print(f"        {info2}")
                    console.print(f"        {info3}")
                    console.print(f"        {info4}")
                except Exception as e:
                    console.print(
                        f"      * [{di}] (error reading device info: {e})"
                    )
    except Exception as e:
        console.print(f"- pyopencl not importable or no platforms: {e}")


@main.command()
@click.option(
    "--size",
    "size",
    default=192,
    show_default=True,
    help="Matrix size N for NxN matmul demo.",
)
@click.option(
    "--iters",
    default=3,
    show_default=True,
    help="Median timing iterations for UHOP path.",
)
@click.option(
    "--ocl-device",
    type=int,
    default=None,
    help="OpenCL GPU device index override (across all platforms).",
)
def demo(size: int, iters: int, ocl_device: int | None):
    """
    Run a short demo: device info + naive Python vs UHOP matmul benchmark.
    """
    console.print("[bold cyan]UHOP Demo[/bold cyan]")
    if ocl_device is not None:
        try:
            if _set_opencl_device is not None:
                _set_opencl_device(ocl_device)
            else:
                console.print(
                    "[yellow]Warning:[/yellow] OpenCL not available, "
                    f"cannot set device index {ocl_device}"
                )
        except Exception as e:
            console.print(
                "[yellow]Warning:[/yellow] could not set OpenCL device index "
                f"{ocl_device}: {e}"
            )
    console.print(_hardware_summary())

    # Naive triple-loop baseline (slow, but shows UHOP acceleration clearly)
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
    # single iter; it's very slow
    t_naive = _med(lambda: matmul_naive(A, B), iters=1)

    console.print(f"UHOP (optimized over naive): [bold]{t_uhop:.6f} s[/bold]")
    console.print(f"Naive Python baseline     : [bold]{t_naive:.6f} s[/bold]")
    if t_uhop < t_naive:
        console.print("[green]UHOP wins ✅[/green]")
    else:
        console.print(
            "[yellow]Baseline was faster in this config. Try larger size or "
            "check GPU drivers.[/yellow]"
        )


@main.command(name="demo-conv2d-relu")
@click.option("--n", default=1, show_default=True, help="Batch size N.")
@click.option("--c-in", default=3, show_default=True, help="Input channels.")
@click.option("--c-out", default=16, show_default=True, help="Output ch.")
@click.option("--h", default=64, show_default=True, help="Input height.")
@click.option("--w", default=64, show_default=True, help="Input width.")
@click.option("--k", default=3, show_default=True, help="Kernel size.")
@click.option("--stride", default=1, show_default=True, help="Stride.")
@click.option("--padding", default=1, show_default=True, help="Padding.")
@click.option(
    "--iters",
    default=2,
    show_default=True,
    help="Median timing iterations for UHOP path.",
)
@click.option(
    "--ocl-device",
    type=int,
    default=None,
    help="OpenCL GPU device index override (across all platforms).",
)
def demo_conv2d_relu(
    n: int,
    c_in: int,
    c_out: int,
    h: int,
    w: int,
    k: int,
    stride: int,
    padding: int,
    iters: int,
    ocl_device: int | None,
):
    """
    Benchmark fused Conv2D+ReLU (OpenCL) vs CPU baselines (naive, torch).
    """
    console.print("[bold cyan]UHOP Fused Conv2D+ReLU Demo[/bold cyan]")
    if ocl_device is not None:
        try:
            _set_opencl_device(ocl_device)
        except Exception as e:
            console.print(
                f"[yellow]Warning:[/yellow] could not set OpenCL device index "
                f"{ocl_device}: {e}"
            )
    console.print(_hardware_summary())

    import numpy as np
    rng = np.random.default_rng(0)
    x = rng.random((n, c_in, h, w), dtype=np.float32)
    wgt = rng.random((c_out, c_in, k, k), dtype=np.float32)

    # UHOP fused path (OpenCL)
    def _med(run, iters=iters):
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            run()
            times.append(time.perf_counter() - t0)
        return float(np.median(times))

    # Warm-up
    if _ocl_conv2d_relu is None:
        console.print("[red]Error:[/red] OpenCL Conv2D+ReLU not available")
        return
    
    _ = _ocl_conv2d_relu(x, wgt, stride=stride, padding=padding)
    t_uhop = _med(
        lambda: _ocl_conv2d_relu(x, wgt, stride=stride, padding=padding)
    )

    # Baseline 1: naive CPU loops (may be slow, do single iteration)
    def _conv2d_relu_naive(inp, wt):
        N, C, H, W = inp.shape
        Cout, Cin, KH, KW = wt.shape
        assert C == Cin
        outH = (H + 2*padding - KH) // stride + 1
        outW = (W + 2*padding - KW) // stride + 1
        out = np.zeros((N, Cout, outH, outW), dtype=np.float32)
        for n_i in range(N):
            for co in range(Cout):
                for y_o in range(outH):
                    for x_o in range(outW):
                        s = 0.0
                        for ci in range(Cin):
                            for ky in range(KH):
                                for kx in range(KW):
                                    iy = y_o*stride - padding + ky
                                    ix = x_o*stride - padding + kx
                                    if 0 <= iy < H and 0 <= ix < W:
                                        s += (
                                            inp[n_i, ci, iy, ix]
                                            * wt[co, ci, ky, kx]
                                        )
                        out[n_i, co, y_o, x_o] = s if s > 0.0 else 0.0
        return out

    t_naive = _med(lambda: _conv2d_relu_naive(x, wgt), iters=1)

    # Baseline 2: torch CPU
    t_torch = None
    try:
        import torch
        import torch.nn.functional as F
        xt = torch.from_numpy(x)
        wt = torch.from_numpy(wgt)

        def _run_torch():
            y = F.conv2d(xt, wt, stride=stride, padding=padding)
            y = F.relu(y)
            return y

        # warm
        _ = _run_torch()
        t_torch = _med(_run_torch)
    except Exception as e:
        console.print(f"[yellow]Torch baseline unavailable: {e}")

    console.print(f"UHOP fused Conv2D+ReLU: [bold]{t_uhop:.6f} s[/bold]")
    console.print(f"Naive CPU baseline    : [bold]{t_naive:.6f} s[/bold]")
    if t_torch is not None:
        console.print(f"Torch CPU baseline    : [bold]{t_torch:.6f} s[/bold]")
    if t_uhop < t_naive:
        console.print("[green]UHOP beats naive ✅[/green]")
    else:
        console.print(
            "[yellow]Naive was faster at this size. Try larger H/W or "
            "channels.[/yellow]"
        )


@main.command(name="ai-generate")
@click.argument("operation")
@click.option(
    "--target",
    type=click.Choice(
        ["cuda", "opencl", "python", "triton"],
        case_sensitive=False,
    ),
    default="opencl",
    show_default=True,
)
@click.option(
    "--validate",
    is_flag=True,
    help=(
        "Validate generated code (python: import; opencl: build)."
    ),
)
@click.option(
    "--smoke",
    is_flag=True,
    help="Run a small correctness+timing smoke test vs NumPy.",
)
@click.option(
    "--temperature",
    default=0.0,
    show_default=True,
    help="Sampling temperature for generation.",
)
@click.option(
    "--samples",
    default=1,
    show_default=True,
    type=int,
    help=(
        "If >1, gen multiple and bench (OpenCL matmul only)."
    ),
)
@click.option(
    "--model",
    default=None,
    show_default=False,
    help=(
        "Override OpenAI model (else uses env/default)."
    ),
)
def ai_generate(
    operation: str,
    target: str,
    validate: bool,
    smoke: bool,
    temperature: float,
    samples: int,
    model: str | None,
):
    """
    Generate a kernel with AI for OPERATION and TARGET.

    Saved to generated_kernels/.

    Example:
      python -m uhop.cli ai-generate matmul --target opencl --validate
    """
    console.print(
        f"[bold cyan]AI Codegen[/bold cyan] — op={operation}, target={target}"
    )
    gen = AICodegen(model=model) if model else AICodegen()
    if samples <= 1:
        try:
            path = gen.generate(
                operation, target=target, temperature=temperature
            )
            console.print(f"Saved generated code to: [bold]{path}[/bold]")
        except Exception as e:
            console.print(f"[red]Generation failed:[/red] {e}")
            return
        paths = [path]
    else:
        # Multi-candidate generation for OpenCL matmul only (bench fastest)
        paths = []
        for i in range(samples):
            try:
                p = gen.generate(
                    operation,
                    target=target,
                    temperature=temperature,
                    suffix=f"_cand{i+1}",
                )
                console.print(f"Saved candidate #{i+1}: [bold]{p}[/bold]")
                paths.append(p)
            except Exception as e:
                console.print(f"[yellow]Candidate {i+1} failed:[/yellow] {e}")
        if not paths:
            console.print("[red]No successful candidates generated.[/red]")
            return

    if not validate:
        return

    if target.lower() == "python":
        # Syntax validation already performed in generator; ensure importable
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "uhop.generated_kernels.ai_generated", str(path)
            )
            mod = importlib.util.module_from_spec(spec)
            assert spec and spec.loader
            spec.loader.exec_module(mod)  # type: ignore
            console.print("[green]Python code imported successfully.[/green]")
        except Exception as e:
            console.print(f"[red]Import failed:[/red] {e}")
    elif target.lower() == "opencl":
        # Try to compile OpenCL source to catch obvious errors
        try:
            import pyopencl as cl
            ctx, q = _ensure_opencl_context_for_validation()
            built = []
            for pth in paths:
                src = Path(pth).read_text()
                prg = cl.Program(ctx, src).build()
                built.append((pth, prg))
            console.print("[green]OpenCL program(s) built OK.[/green]")
        except Exception as e:
            console.print(f"[red]OpenCL build failed:[/red] {e}")
            return

    if smoke:
        import numpy as np
        if operation.lower() == "matmul":
            A = np.random.RandomState(0).rand(32, 16).astype(np.float32)
            B = np.random.RandomState(1).rand(16, 24).astype(np.float32)
            ref = A @ B
            if target.lower() == "python":
                try:
                    import importlib.util
                    spec = importlib.util.spec_from_file_location(
                        "uhop.generated_kernels.ai_generated",
                        str(path),
                    )
                    mod = importlib.util.module_from_spec(spec)
                    assert spec and spec.loader
                    spec.loader.exec_module(mod)  # type: ignore
                    fn = getattr(mod, f"generated_{operation}")
                    out = fn(A, B)
                    err = float(np.max(np.abs(out - ref)))
                    console.print(f"Smoke diff (Linf): {err:.3e}")
                except Exception as e:
                    console.print(f"[red]Smoke test failed:[/red] {e}")
            elif target.lower() == "opencl":
                try:
                    import pyopencl as cl
                    from time import perf_counter
                    ctx, q = _ensure_opencl_context_for_validation()
                    results = []
                    for pth in paths:
                        src = Path(pth).read_text()
                        prg = cl.Program(ctx, src).build()
                        M, K = A.shape
                        K2, N = B.shape
                        assert K == K2
                        C = np.empty((M, N), dtype=np.float32)
                        mf = cl.mem_flags
                        a_buf = cl.Buffer(
                            ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A
                        )
                        b_buf = cl.Buffer(
                            ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B
                        )
                        c_buf = cl.Buffer(ctx, mf.WRITE_ONLY, C.nbytes)
                        kn = cl.Kernel(prg, f"generated_{operation}")
                        kn.set_args(
                            np.int32(M), np.int32(N), np.int32(K),
                            a_buf, b_buf, c_buf,
                        )
                        gsz = (int(M), int(N))
                        # warm
                        cl.enqueue_nd_range_kernel(q, kn, gsz, None)
                        q.finish()
                        t0 = perf_counter()
                        cl.enqueue_nd_range_kernel(q, kn, gsz, None)
                        q.finish()
                        cl.enqueue_copy(q, C, c_buf)
                        q.finish()
                        err = float(np.max(np.abs(C - ref)))
                        dt = float(perf_counter() - t0)
                        results.append((pth, err, dt))
                        msg = (
                            f"Candidate {pth.name}: Linf={err:.3e}, "
                            f"time={dt:.6f}s"
                        )
                        console.print(msg)
                    # Pick best by time with acceptable error
                    ok = [r for r in results if r[1] < 1e-3]
                    if ok:
                        best = min(ok, key=lambda r: r[2])
                    else:
                        best = min(results, key=lambda r: r[1])
                    console.print(
                        f"[bold]Selected best:[/bold] {best[0].name} "
                        f"(Linf={best[1]:.3e}, time={best[2]:.6f}s)"
                    )
                    # Write manifest
                    ai_dir = Path(paths[0]).parent
                    manifest = {
                        "operation": operation,
                        "target": target,
                        "model": gen.model,
                        "prompt": gen.last_prompt,
                        "created_at": _dt.utcnow().isoformat() + "Z",
                        "candidates": [
                            {
                                "path": str(pth),
                                "linf": float(err),
                                "time": float(dt),
                            }
                            for pth, err, dt in results
                        ],
                        "selected": str(best[0]),
                    }
                    (ai_dir / f"ai_{operation}_manifest.json").write_text(
                        _json.dumps(manifest, indent=2)
                    )
                    # Auto-cache best kernel for optimizer (matmul only)
                    try:
                        cache = _UhopCache()
                        from .hardware import detect_hardware as _detect
                        cache.set(
                            "matmul",
                            {
                                "backend": "ai_opencl",
                                "path": str(best[0]),
                                "kernel_name": f"generated_{operation}",
                                "hardware": _detect().__dict__,
                            },
                        )
                        console.print(
                            "[green]Cached best kernel for optimizer "
                            "(ai_opencl/matmul).[/green]"
                        )
                    except Exception as e:
                        console.print(
                            "[yellow]Warn:[/yellow] cache set failed: " f"{e}"
                        )
                except Exception as e:
                    console.print(f"[red]OpenCL smoke test failed:[/red] {e}")
        elif operation.lower() == "relu" and target.lower() == "opencl":
            try:
                import pyopencl as cl
                from time import perf_counter
                ctx, q = _ensure_opencl_context_for_validation()
                X = np.random.RandomState(0).randn(1_024).astype(np.float32)
                ref = np.maximum(X, 0)
                results = []
                for pth in paths:
                    src = Path(pth).read_text()
                    prg = cl.Program(ctx, src).build()
                    N = np.int32(X.size)
                    Y = np.empty_like(X)
                    mf = cl.mem_flags
                    x_buf = cl.Buffer(
                        ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=X
                    )
                    y_buf = cl.Buffer(ctx, mf.WRITE_ONLY, Y.nbytes)
                    kn = cl.Kernel(prg, "generated_relu")
                    kn.set_args(N, x_buf, y_buf)
                    # warm
                    cl.enqueue_nd_range_kernel(q, kn, (int(N),), None)
                    q.finish()
                    t0 = perf_counter()
                    cl.enqueue_nd_range_kernel(q, kn, (int(N),), None)
                    q.finish()
                    cl.enqueue_copy(q, Y, y_buf)
                    q.finish()
                    err = float(np.max(np.abs(Y - ref)))
                    dt = float(perf_counter() - t0)
                    results.append((pth, err, dt))
                    console.print(
                        f"Candidate {pth.name}: Linf={err:.3e}, time={dt:.6f}s"
                    )
                ok = [r for r in results if r[1] < 1e-6]
                best = min(ok or results, key=lambda r: (r[2], r[1]))
                console.print(
                    f"[bold]Selected best:[/bold] {best[0].name} "
                    f"(Linf={best[1]:.3e}, time={best[2]:.6f}s)"
                )
                ai_dir = Path(paths[0]).parent
                manifest = {
                    "operation": operation,
                    "target": target,
                    "model": gen.model,
                    "prompt": gen.last_prompt,
                    "created_at": _dt.utcnow().isoformat() + "Z",
                    "candidates": [
                        {
                            "path": str(pth),
                            "linf": float(err),
                            "time": float(dt),
                        }
                        for pth, err, dt in results
                    ],
                    "selected": str(best[0]),
                }
                (ai_dir / f"ai_{operation}_manifest.json").write_text(
                    _json.dumps(manifest, indent=2)
                )
                # Auto-cache best kernel for optimizer
                try:
                    cache = _UhopCache()
                    from .hardware import detect_hardware as _detect
                    cache.set(
                        "relu",
                        {
                            "backend": "ai_opencl",
                            "path": str(best[0]),
                            "kernel_name": "generated_relu",
                            "hardware": _detect().__dict__,
                        },
                    )
                    console.print(
                        "[green]Cached best kernel for optimizer "
                        "(ai_opencl/relu).[/green]"
                    )
                except Exception as e:
                    console.print(
                        "[yellow]Warn:[/yellow] cache set failed: " f"{e}"
                    )
            except Exception as e:
                console.print(f"[red]OpenCL ReLU smoke test failed:[/red] {e}")
        elif operation.lower() == "conv2d" and target.lower() == "opencl":
            try:
                import pyopencl as cl
                from time import perf_counter
                ctx, q = _ensure_opencl_context_for_validation()
                # Small config
                N, C_in, H, W = 1, 3, 32, 32
                C_out, KH, KW = 4, 3, 3
                stride, pad = 1, 1
                rng = np.random.default_rng(0)
                X = rng.standard_normal((N, C_in, H, W), dtype=np.float32)
                Wt = rng.standard_normal(
                    (C_out, C_in, KH, KW), dtype=np.float32
                )
                outH = (H + 2*pad - KH)//stride + 1
                outW = (W + 2*pad - KW)//stride + 1
                # Reference via torch if available, else naive
                try:
                    import torch
                    import torch.nn.functional as F
                    ref = F.conv2d(
                        torch.from_numpy(X),
                        torch.from_numpy(Wt),
                        stride=stride,
                        padding=pad,
                    ).numpy()
                except Exception:
                    ref = np.zeros((N, C_out, outH, outW), dtype=np.float32)
                    for n in range(N):
                        for co in range(C_out):
                            for y in range(outH):
                                for x in range(outW):
                                    s = 0.0
                                    for ci in range(C_in):
                                        for ky in range(KH):
                                            for kx in range(KW):
                                                iy = y*stride - pad + ky
                                                ix = x*stride - pad + kx
                                                if 0 <= iy < H and 0 <= ix < W:
                                                    s += (
                                                        X[n, ci, iy, ix]
                                                        * Wt[co, ci, ky, kx]
                                                    )
                                    ref[n, co, y, x] = s
                results = []
                for pth in paths:
                    src = Path(pth).read_text()
                    prg = cl.Program(ctx, src).build()
                    mf = cl.mem_flags
                    x_buf = cl.Buffer(
                        ctx,
                        mf.READ_ONLY | mf.COPY_HOST_PTR,
                        hostbuf=X.astype(np.float32),
                    )
                    w_buf = cl.Buffer(
                        ctx,
                        mf.READ_ONLY | mf.COPY_HOST_PTR,
                        hostbuf=Wt.astype(np.float32),
                    )
                    Y = np.empty((N, C_out, outH, outW), dtype=np.float32)
                    y_buf = cl.Buffer(ctx, mf.WRITE_ONLY, Y.nbytes)
                    kn = cl.Kernel(prg, "generated_conv2d")
                    # args: N,C_in,H,W,C_out,KH,KW,stride,pad,input,weight,
                    #       output,outH,outW
                    kn.set_args(
                        np.int32(N), np.int32(C_in), np.int32(H), np.int32(W),
                        np.int32(C_out), np.int32(KH), np.int32(KW),
                        np.int32(stride), np.int32(pad),
                        x_buf, w_buf, y_buf,
                        np.int32(outH), np.int32(outW)
                    )
                    gsz = (int(outW), int(outH), int(N*C_out))
                    # warm
                    cl.enqueue_nd_range_kernel(q, kn, gsz, None)
                    q.finish()
                    t0 = perf_counter()
                    cl.enqueue_nd_range_kernel(q, kn, gsz, None)
                    q.finish()
                    cl.enqueue_copy(q, Y, y_buf)
                    q.finish()
                    err = float(np.max(np.abs(Y - ref)))
                    dt = float(perf_counter() - t0)
                    results.append((pth, err, dt))
                    console.print(
                        f"Candidate {pth.name}: Linf={err:.3e}, time={dt:.6f}s"
                    )
                ok = [r for r in results if r[1] < 1e-2]
                best = min(ok or results, key=lambda r: (r[2], r[1]))
                console.print(
                    f"[bold]Selected best:[/bold] {best[0].name} "
                    f"(Linf={best[1]:.3e}, time={best[2]:.6f}s)"
                )
                ai_dir = Path(paths[0]).parent
                manifest = {
                    "operation": operation,
                    "target": target,
                    "model": gen.model,
                    "prompt": gen.last_prompt,
                    "created_at": _dt.utcnow().isoformat() + "Z",
                    "candidates": [
                        {
                            "path": str(pth),
                            "linf": float(err),
                            "time": float(dt),
                        }
                        for pth, err, dt in results
                    ],
                    "selected": str(best[0]),
                }
                (ai_dir / f"ai_{operation}_manifest.json").write_text(
                    _json.dumps(manifest, indent=2)
                )
                # Auto-cache best kernel for optimizer
                try:
                    cache = _UhopCache()
                    from .hardware import detect_hardware as _detect
                    cache.set(
                        "conv2d",
                        {
                            "backend": "ai_opencl",
                            "path": str(best[0]),
                            "kernel_name": "generated_conv2d",
                            "hardware": _detect().__dict__,
                        },
                    )
                    console.print(
                        "[green]Cached best kernel for optimizer "
                        "(ai_opencl/conv2d).[/green]"
                    )
                except Exception as e:
                    console.print(
                        "[yellow]Warn:[/yellow] cache set failed: "
                        f"{e}"
                    )
            except Exception as e:
                console.print(
                    f"[red]OpenCL Conv2D smoke test failed:[/red] {e}"
                )


@main.command(name="ai-generate-fused")
@click.option("--stride", default=1, show_default=True, type=int)
@click.option("--padding", default=0, show_default=True, type=int)
@click.option("--temperature", default=0.0, show_default=True, type=float)
@click.option(
    "--model",
    default=None,
    show_default=False,
    help=(
        "Override OpenAI model (else uses env/default)."
    ),
)
def ai_generate_fused(
    stride: int,
    padding: int,
    temperature: float,
    model: str | None,
):
    """
    Generate a fused OpenCL Conv2D+ReLU kernel and benchmark it.

    Caches if it's competitive vs the current fused backend.
    """
    console.print("[bold cyan]AI Codegen — Fused Conv2D+ReLU[/bold cyan]")
    from .ai_codegen.generator import AICodegen
    gen = AICodegen(model=model) if model else AICodegen()
    # Build a prompt using the fused template constants from prompt_templates
    # (already imported in generator via generator prompt strings)
    prompt_extra = (
        "Produce a single OpenCL C kernel named conv2d_relu with signature:\n"
        "__kernel void conv2d_relu(\n"
        "    __global const float* input,   // [C_in,H_in,W_in]\n"
        "    __global const float* weight,  // [C_out,C_in,KH,KW]\n"
        "    __global const float* bias,    // [C_out]\n"
        "    __global float* output,        // [C_out,H_out,W_out]\n"
        "    const int C_in,const int H_in,const int W_in,\n"
        "    const int C_out,const int KH,const int KW,\n"
        "    const int stride,const int pad,const int N,\n"
        "    const int H_out,const int W_out);\n"
        "Assume N==1 for performance; use local size (8,8,1). "
        "Bounds-check and write valid OpenCL C."
    )
    try:
        path = gen.generate(
            "conv2d_relu",
            target="opencl",
            prompt_extra=prompt_extra,
            temperature=temperature,
        )
        console.print(f"Saved generated fused kernel to: [bold]{path}[/bold]")
    except Exception as e:
        console.print(f"[red]Generation failed:[/red] {e}")
        return

    # Compile and benchmark vs current backend fused path
    try:
        import pyopencl as cl
        import numpy as np
        from time import perf_counter
        from .backends.opencl_backend import (
            opencl_conv2d_relu as fused_baseline,
        )
        ctx, q = _ensure_opencl_context_for_validation()
        src = Path(path).read_text()
        prg = cl.Program(ctx, src).build()
        # Small test config
        N, C_in, H, W = 1, 3, 64, 64
        C_out, KH, KW = 8, 3, 3
        rng = np.random.default_rng(0)
        X = rng.random((N, C_in, H, W), dtype=np.float32)
        Wt = rng.random((C_out, C_in, KH, KW), dtype=np.float32)
        bias = np.zeros((C_out,), dtype=np.float32)
        outH = (H + 2*padding - KH)//stride + 1
        outW = (W + 2*padding - KW)//stride + 1
        # Reference via baseline fused path
        ref = fused_baseline(X, Wt, stride=stride, padding=padding)
        # Prepare OpenCL buffers
        mf = cl.mem_flags
        inp = X[0]  # [C_in,H,W]
        in_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=inp)
        w_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=Wt)
        b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=bias)
        Y = np.empty((C_out, outH, outW), dtype=np.float32)
        y_buf = cl.Buffer(ctx, mf.WRITE_ONLY, Y.nbytes)
        kn = cl.Kernel(prg, "conv2d_relu")
        kn.set_args(
            in_buf, w_buf, b_buf, y_buf,
            np.int32(C_in), np.int32(H), np.int32(W),
            np.int32(C_out), np.int32(KH), np.int32(KW),
            np.int32(stride), np.int32(padding), np.int32(1),
            np.int32(outH), np.int32(outW)
        )
        gsz = (int(outW), int(outH), int(C_out))
        lsz = (8, 8, 1)
        # Warm
        cl.enqueue_nd_range_kernel(q, kn, gsz, lsz)
        q.finish()
        t0 = perf_counter()
        cl.enqueue_nd_range_kernel(q, kn, gsz, lsz)
        q.finish()
        cl.enqueue_copy(q, Y, y_buf)
        q.finish()
        dt_ai = float(perf_counter() - t0)
        out_ai = Y.reshape(1, C_out, outH, outW)
        err = float(np.max(np.abs(out_ai - ref)))
        console.print(f"AI fused Linf={err:.3e}, time={dt_ai:.6f}s")
        # Baseline timing
        t0 = perf_counter()
        _ = fused_baseline(X, Wt, stride=stride, padding=padding)
        dt_base = float(perf_counter() - t0)
        console.print(f"Baseline fused time={dt_base:.6f}s")
        # Cache if reasonable
        ai_dir = Path(path).parent
        manifest = {
            "operation": "conv2d_relu_fused",
            "target": "opencl",
            "model": gen.model,
            "prompt": gen.last_prompt,
            "created_at": _dt.utcnow().isoformat() + "Z",
            "result": {
                "linf": err,
                "time": dt_ai,
                "baseline_time": dt_base,
                "selected": str(path),
            },
        }
        (ai_dir / "ai_conv2d_relu_manifest.json").write_text(
            _json.dumps(manifest, indent=2)
        )
        if err < 1e-2:
            try:
                cache = _UhopCache()
                from .hardware import detect_hardware as _detect
                cache.set(
                    "conv2d_relu",
                    {
                        "backend": "ai_opencl",
                        "path": str(path),
                        "kernel_name": "conv2d_relu",
                        "hardware": _detect().__dict__,
                    },
                )
                console.print(
                    "[green]Cached AI fused kernel for optimizer "
                    "(ai_opencl/conv2d_relu).[/green]"
                )
            except Exception as e:
                console.print(
                    f"[yellow]Warn:[/yellow] could not cache fused kernel: {e}"
                )
    except Exception as e:
        console.print(f"[red]Fused build/benchmark failed:[/red] {e}")


def _ensure_opencl_context_for_validation():
    try:
        import pyopencl as cl
        plats = cl.get_platforms()
        for p in plats:
            devs = [d for d in p.get_devices() if d.type & cl.device_type.GPU]
            if devs:
                ctx = cl.Context(devices=[devs[0]])
                q = cl.CommandQueue(ctx)
                return ctx, q
        # fallback: any device
        ctx = cl.create_some_context(interactive=False)
        q = cl.CommandQueue(ctx)
        return ctx, q
    except Exception as e:
        raise RuntimeError(f"No OpenCL context available: {e}")


@main.command()
@click.argument("function_path")
def optimize_func(function_path):
    """Optimize a Python function via @uhop.optimize by importing it."""
    console.print(f"[green]Optimizing function:[/green] {function_path}")
    __import__(function_path)
    console.print("[bold green]Optimization complete![/bold green]")


if __name__ == "__main__":
    main()


# Cache management commands
@main.group()
def cache():
    """Inspect and manage the UHOP backend decision cache."""
    pass


@cache.command("list")
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Emit machine-readable JSON.",
)
def cache_list(as_json: bool):
    c = _UhopCache()
    data = c.all()
    if as_json:
        console.print(_json.dumps(data, indent=2))
    else:
        if not data:
            console.print("(cache is empty)")
            return
        for k, v in data.items():
            backend = v.get("backend")
            cached_at = v.get("_cached_at")
            console.print(f"- {k}: backend={backend} cached_at={cached_at}")


@cache.command("show")
@click.argument("key")
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Emit machine-readable JSON.",
)
def cache_show(key: str, as_json: bool):
    c = _UhopCache()
    rec = c.get(key)
    if rec is None:
        console.print(f"[yellow]No record for key:[/yellow] {key}")
        return
    if as_json:
        console.print(_json.dumps(rec, indent=2))
    else:
        for kk, vv in rec.items():
            console.print(f"{kk}: {vv}")


@cache.command("clear")
def cache_clear():
    c = _UhopCache()
    c.clear()
    console.print("[green]Cache cleared.[/green]")


@cache.command("delete")
@click.argument("key")
def cache_delete(key: str):
    c = _UhopCache()
    c.delete(key)
    console.print(f"[green]Deleted cache entry:[/green] {key}")


@main.command(name="web-bridge")
@click.option("--port", type=int, default=5823, show_default=True)
def web_bridge(port: int):
    """Run a local HTTP bridge to execute whitelisted UHOP CLI commands.

    The server binds to 127.0.0.1 and exposes:
      - GET /health
      - POST /run {"cmd": "uhop info --json"}

    Intended for the docs/demo website to interact with your local machine.
    """
    try:
        from .web_bridge import run_web_bridge
    except Exception as e:
        console.print(f"[red]Failed to import web bridge:[/red] {e}")
        return
    console.print(
        f"[green]Starting local web bridge on http://127.0.0.1:{port}[/green]"
    )
    run_web_bridge(port)


@main.command(name="web-api")
@click.option("--host", type=str, default="0.0.0.0", show_default=True)
@click.option("--port", type=int, default=5824, show_default=True)
def web_api(host: str, port: int):
    """Run a minimal HTTP API for the online demo.

    Endpoints:
      - GET /health
      - GET /info
      - POST /demo/matmul {"size":256,"iters":3}
    """
    try:
        from .web_api import run as _run
    except Exception as e:
        console.print(f"[red]Failed to import web API:[/red] {e}")
        return
    console.print(
        f"[green]Starting web API on http://{host}:{port}[/green]"
    )
    _run(host, port)


@cache.command("invalidate")
@click.option(
    "--all",
    "invalidate_all",
    is_flag=True,
    help="Remove all cache entries.",
)
@click.option(
    "--device",
    "device_query",
    type=str,
    default=None,
    help=(
        "Substring filter to remove entries for a device (e.g. 'mps', 'cuda',"
        " vendor name). Case-insensitive."
    ),
)
@click.option(
    "--backend",
    "backend_name",
    type=str,
    default=None,
    help="Remove entries for a specific backend (exact match).",
)
def cache_invalidate(
    invalidate_all: bool, device_query: str | None, backend_name: str | None
):
    """Invalidate cache entries by scope.

    Examples:
      uhop cache invalidate --all
      uhop cache invalidate --device mps
      uhop cache invalidate --backend opencl
    """
    c = _UhopCache()
    total_removed = 0
    if invalidate_all:
        total_removed += c.invalidate_all()
    if device_query:
        total_removed += c.invalidate_device(device_query)
    if backend_name:
        total_removed += c.invalidate_backend(backend_name)
    console.print(
        f"[green]Invalidation complete.[/green] removed={total_removed}"
    )
