"""
OpenCL Operator Micro-Benchmarks (UHOP)

Benchmarks a curated set of OpenCL kernels in uhop/kernels/opencl using
pyopencl with a profiling-enabled command queue. Measures kernel time only
(excluding host<->device copies after warmup), with configurable warmup and
iterations.

Usage examples:
  python -m uhop.benchmarks.opencl_ops_bench --ops all --verbose
  python -m uhop.benchmarks.opencl_ops_bench --ops elementwise_add,matmul,conv2d --iterations 20
  python -m uhop.benchmarks.opencl_ops_bench --output .benchmarks/opencl_ops.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import pyopencl as cl
except Exception:
    cl = None  # type: ignore


@dataclass
class BenchRecord:
    op: str
    shape: Tuple[int, ...]
    mean_ms: float
    median_ms: float
    min_ms: float
    max_ms: float
    std_ms: float
    iters: int
    baseline_mean_ms: Optional[float] = None
    speedup_vs_baseline: Optional[float] = None
    gflops: Optional[float] = None
    gib_per_s: Optional[float] = None
    validated: Optional[bool] = None
    max_abs_err: Optional[float] = None
    max_rel_err: Optional[float] = None
    uhop_mean_ms: Optional[float] = None
    uhop_speedup_vs_naive: Optional[float] = None

    def to_dict(self) -> Dict:
        return asdict(self)


def _mk_queue() -> Tuple[Any, Any]:
    # Create context and profiling-enabled queue
    assert cl is not None, "pyopencl not available"
    ctx = cl.create_some_context(interactive=False)
    props = cl.command_queue_properties.PROFILING_ENABLE
    q = cl.CommandQueue(ctx, properties=props)
    return ctx, q


def _safe_div(n: float, d: float) -> float:
    return n / d if d != 0 else 0.0


def _measure_events_to_ms(evts: List[Any]) -> Dict[str, float]:
    times = []
    for e in evts:
        e.wait()
        try:
            dt = (e.profile.end - e.profile.start) * 1e-6  # ns -> ms
        except Exception:
            dt = 0.0
        times.append(dt)
    return {
        "mean": statistics.mean(times),
        "median": statistics.median(times),
        "min": min(times),
        "max": max(times),
        "std": statistics.stdev(times) if len(times) > 1 else 0.0,
    }


def _measure_host_callable_ms(fn, warmup: int, iters: int) -> Dict[str, float]:
    import time

    ts: List[float] = []
    for _ in range(max(0, warmup)):
        fn()
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000.0)
    return {
        "mean": statistics.mean(ts) if ts else 0.0,
        "median": statistics.median(ts) if ts else 0.0,
        "min": min(ts) if ts else 0.0,
        "max": max(ts) if ts else 0.0,
        "std": statistics.stdev(ts) if len(ts) > 1 else 0.0,
    }


def _build_program(ctx: Any, kernels_path: Path, fname: str) -> Any:
    src = (kernels_path / fname).read_text()
    return cl.Program(ctx, src).build()


def bench_elementwise(
    ctx: Any,
    q: Any,
    kernels_path: Path,
    op: str,
    N: int,
    warmup: int,
    iters: int,
    validate: bool = False,
) -> BenchRecord:
    prg = _build_program(ctx, kernels_path, "elementwise.cl")
    mf = cl.mem_flags
    a = np.random.randn(N).astype(np.float32)
    b = np.random.randn(N).astype(np.float32)
    out = np.empty_like(a)
    a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
    out_buf = cl.Buffer(ctx, mf.WRITE_ONLY, out.nbytes)

    # Determine kernel and args
    eps = np.float32(1e-6)
    if op == "elementwise_add":
        add_kn = prg.elementwise_add if hasattr(prg, "elementwise_add") else None

        def launch():
            return add_kn(q, (N,), None, a_buf, b_buf, out_buf, np.int32(N))

    elif op == "elementwise_sub":
        sub_kn = prg.elementwise_sub

        def launch():
            return sub_kn(q, (N,), None, a_buf, b_buf, out_buf, np.int32(N))

    elif op == "elementwise_mul":
        mul_kn = prg.elementwise_mul

        def launch():
            return mul_kn(q, (N,), None, a_buf, b_buf, out_buf, np.int32(N))

    elif op == "elementwise_div":
        div_kn = prg.elementwise_div

        def launch():
            return div_kn(q, (N,), None, a_buf, b_buf, out_buf, np.int32(N), eps)

    elif op == "elementwise_pow":
        pow_kn = prg.elementwise_pow

        def launch():
            return pow_kn(q, (N,), None, a_buf, b_buf, out_buf, np.int32(N))

    elif op == "elementwise_max":
        max_kn = prg.elementwise_max

        def launch():
            return max_kn(q, (N,), None, a_buf, b_buf, out_buf, np.int32(N))

    elif op == "elementwise_min":
        min_kn = prg.elementwise_min

        def launch():
            return min_kn(q, (N,), None, a_buf, b_buf, out_buf, np.int32(N))

    elif op == "elementwise_leakyrelu":
        alpha = np.float32(0.01)
        lr_kn = prg.elementwise_leakyrelu

        def launch():
            return lr_kn(q, (N,), None, a_buf, out_buf, np.int32(N), alpha)

    elif op == "elementwise_exp":
        exp_kn = prg.elementwise_exp

        def launch():
            return exp_kn(q, (N,), None, a_buf, out_buf, np.int32(N))

    elif op == "elementwise_log":
        log_kn = prg.elementwise_log

        def launch():
            return log_kn(q, (N,), None, a_buf, out_buf, np.int32(N), eps)

    elif op == "elementwise_sqrt":
        sqrt_kn = prg.elementwise_sqrt

        def launch():
            return sqrt_kn(q, (N,), None, a_buf, out_buf, np.int32(N))

    else:
        raise ValueError(f"Unsupported elementwise op: {op}")

    # Warmup
    for _ in range(warmup):
        evt = launch()
        evt.wait()

    evts = []
    for _ in range(iters):
        evts.append(launch())
    q.finish()

    stats = _measure_events_to_ms(evts)

    # Baseline with NumPy
    if op == "elementwise_add":
        base_fn = lambda: (a + b)
    elif op == "elementwise_sub":
        base_fn = lambda: (a - b)
    elif op == "elementwise_mul":
        base_fn = lambda: (a * b)
    elif op == "elementwise_div":
        base_fn = lambda: (a / (b + 1e-6))
    elif op == "elementwise_pow":
        base_fn = lambda: np.power(a, b)
    elif op == "elementwise_max":
        base_fn = lambda: np.maximum(a, b)
    elif op == "elementwise_min":
        base_fn = lambda: np.minimum(a, b)
    elif op == "elementwise_leakyrelu":
        base_fn = lambda: np.where(a > 0.0, a, 0.01 * a)
    elif op == "elementwise_exp":
        base_fn = lambda: np.exp(a)
    elif op == "elementwise_log":
        base_fn = lambda: np.log(np.maximum(np.abs(a), 1e-6))
    elif op == "elementwise_sqrt":
        base_fn = lambda: np.sqrt(np.abs(a))
    else:
        base_fn = None

    baseline = _measure_host_callable_ms(base_fn, warmup=1, iters=max(1, iters)) if base_fn else {"mean": None}
    spd = (baseline["mean"] / stats["mean"]) if (baseline.get("mean") and stats["mean"] > 0) else None

    # Validation
    vmax = None
    vrmax = None
    vpass = None
    if validate and base_fn is not None:
        out_host = np.empty_like(a)
        cl.enqueue_copy(q, out_host, out_buf).wait()
        ref = base_fn()
        ref = np.asarray(ref, dtype=np.float32)
        diff = np.abs(out_host - ref)
        vmax = float(np.max(diff))
        denom = np.maximum(np.abs(ref), 1e-6)
        vrmax = float(np.max(diff / denom))
        vpass = bool(np.allclose(out_host, ref, atol=1e-3, rtol=1e-3))

    # Performance metrics (approximate)
    # Bytes moved: unary ~ 2*N*4 (read+write), binary ~ 3*N*4
    unary_ops = {
        "elementwise_exp",
        "elementwise_log",
        "elementwise_sqrt",
        "elementwise_leakyrelu",
    }
    bytes_moved = (2 if op in unary_ops else 3) * N * 4
    secs = stats["mean"] * 1e-3
    gibps = _safe_div(bytes_moved, secs) / (1024**3) if secs > 0 else None
    # FLOPs (rough): add/sub/mul/min/max ~ 1 op, leakyrelu ~ 2 ops, div/pow/transcendentals not reported
    if op in {
        "elementwise_add",
        "elementwise_sub",
        "elementwise_mul",
        "elementwise_min",
        "elementwise_max",
    }:
        flops = float(N)
    elif op == "elementwise_leakyrelu":
        flops = float(2 * N)
    else:
        flops = None  # avoid misleading GFLOPS for div/pow/exp/log/sqrt
    gflops = _safe_div(flops, secs) / 1e9 if (flops is not None and secs > 0) else None

    return BenchRecord(
        op=op,
        shape=(N,),
        mean_ms=stats["mean"],
        median_ms=stats["median"],
        min_ms=stats["min"],
        max_ms=stats["max"],
        std_ms=stats["std"],
        iters=iters,
        baseline_mean_ms=baseline.get("mean"),
        speedup_vs_baseline=spd,
        gflops=gflops,
        gib_per_s=gibps,
        validated=vpass,
        max_abs_err=vmax,
        max_rel_err=vrmax,
    )


def bench_activations(
    ctx: Any,
    q: Any,
    kernels_path: Path,
    op: str,
    N: int,
    warmup: int,
    iters: int,
    validate: bool = False,
    compare_optimize: bool = False,
) -> BenchRecord:
    prg = _build_program(ctx, kernels_path, "activations.cl")
    mf = cl.mem_flags
    x = np.random.randn(N).astype(np.float32)
    x_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=x)

    if op == "relu":
        relu_kn = prg.relu_kernel

        def launch():
            return relu_kn(q, (N,), None, x_buf, np.int32(N))

    elif op == "gelu":
        gelu_kn = prg.gelu_kernel

        def launch():
            return gelu_kn(q, (N,), None, x_buf, np.int32(N))

    elif op == "silu":
        silu_kn = prg.silu_kernel

        def launch():
            return silu_kn(q, (N,), None, x_buf, np.int32(N))

    else:
        raise ValueError(f"Unsupported activation: {op}")

    for _ in range(warmup):
        evt = launch()
        evt.wait()

    evts = []
    for _ in range(iters):
        evts.append(launch())
    q.finish()

    stats = _measure_events_to_ms(evts)

    # NumPy baseline activations
    if op == "relu":
        base_fn = lambda: np.maximum(x, 0.0, dtype=np.float32)
    elif op == "gelu":
        base_fn = lambda: 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * (x**3))))
    elif op == "silu":
        base_fn = lambda: x / (1.0 + np.exp(-x))
    else:
        base_fn = None
    baseline = _measure_host_callable_ms(base_fn, warmup=1, iters=max(1, iters)) if base_fn else {"mean": None}
    spd = (baseline["mean"] / stats["mean"]) if (baseline.get("mean") and stats["mean"] > 0) else None

    # Validation
    vpass = vmax = vrmax = None
    if validate and base_fn is not None:
        out_host = np.empty_like(x)
        cl.enqueue_copy(q, out_host, x_buf).wait()
        ref = base_fn()
        ref = np.asarray(ref, dtype=np.float32)
        diff = np.abs(out_host - ref)
        vmax = float(np.max(diff))
        denom = np.maximum(np.abs(ref), 1e-6)
        vrmax = float(np.max(diff / denom))
        vpass = bool(np.allclose(out_host, ref, atol=1e-3, rtol=1e-3))

    # Optional compare via @uhop.optimize for relu only
    uhop_mean = None
    uhop_spd = None
    if compare_optimize and op == "relu":
        try:
            from uhop import optimize as _opt

            @_opt("relu")
            def _naive_relu(a):
                return np.maximum(a, 0.0)

            # Warm + measure
            _ = _naive_relu(x.copy())
            mm = []
            import time

            for _ in range(max(1, iters)):
                t0 = time.perf_counter()
                _ = _naive_relu(x.copy())
                mm.append((time.perf_counter() - t0) * 1000.0)
            uhop_mean = float(statistics.mean(mm))
            uhop_spd = (
                (baseline["mean"] / uhop_mean) if (baseline.get("mean") and uhop_mean and uhop_mean > 0) else None
            )
        except Exception:
            uhop_mean = None
            uhop_spd = None

    return BenchRecord(
        op=op,
        shape=(N,),
        mean_ms=stats["mean"],
        median_ms=stats["median"],
        min_ms=stats["min"],
        max_ms=stats["max"],
        std_ms=stats["std"],
        iters=iters,
        baseline_mean_ms=baseline.get("mean"),
        speedup_vs_baseline=spd,
        validated=vpass,
        max_abs_err=vmax,
        max_rel_err=vrmax,
        uhop_mean_ms=uhop_mean,
        uhop_speedup_vs_naive=uhop_spd,
    )


def bench_reduce_sum(
    ctx: Any,
    q: Any,
    kernels_path: Path,
    N: int,
    warmup: int,
    iters: int,
    validate: bool = False,
) -> BenchRecord:
    prg = _build_program(ctx, kernels_path, "reduce.cl")
    mf = cl.mem_flags
    x = np.random.randn(N).astype(np.float32)
    # Choose work-group size and number of groups
    lsz = 256
    num_groups = max(1, min(64, (N + lsz - 1) // lsz))
    gsz = num_groups * lsz

    partials = np.empty(num_groups, dtype=np.float32)
    out = np.zeros(1, dtype=np.float32)

    x_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x)
    parts_buf = cl.Buffer(ctx, mf.READ_WRITE, partials.nbytes)
    out_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=out)

    r1_kn = prg.reduce_sum_partials
    r2_kn = prg.reduce_sum_finalize

    def launch_two_stage():
        e1 = r1_kn(q, (gsz,), (lsz,), x_buf, parts_buf, np.int32(N))
        e2 = r2_kn(q, (1,), None, parts_buf, out_buf, np.int32(num_groups))
        return [e1, e2]

    # Warmup
    for _ in range(warmup):
        for e in launch_two_stage():
            e.wait()

    evts = []
    for _ in range(iters):
        evts.extend(launch_two_stage())
    q.finish()

    stats = _measure_events_to_ms(evts)
    # Each iteration launches two kernels; approximate per-iteration total time as sum of both kernel durations
    mean_ms = stats["mean"]
    median_ms = stats["median"]

    # NumPy baseline
    baseline = _measure_host_callable_ms(lambda: np.sum(x), warmup=1, iters=max(1, iters))
    spd = (baseline["mean"] / mean_ms) if (baseline.get("mean") and mean_ms > 0) else None
    # Validation: download final output
    vpass = vmax = vrmax = None
    if validate:
        out_host = np.zeros(1, dtype=np.float32)
        cl.enqueue_copy(q, out_host, out_buf).wait()
        ref = np.sum(x).astype(np.float32)
        vmax = float(np.max(np.abs(out_host - ref)))
        vrmax = float(np.max(np.abs(out_host - ref) / max(abs(float(ref)), 1e-6)))
        vpass = bool(np.allclose(out_host, ref, atol=1e-3, rtol=1e-3))

    return BenchRecord(
        op="reduce_sum",
        shape=(N,),
        mean_ms=mean_ms,
        median_ms=median_ms,
        min_ms=stats["min"],
        max_ms=stats["max"],
        std_ms=stats["std"],
        iters=iters * 2,
        baseline_mean_ms=baseline.get("mean"),
        speedup_vs_baseline=spd,
        validated=vpass,
        max_abs_err=vmax,
        max_rel_err=vrmax,
    )


def bench_matmul(
    ctx: Any,
    q: Any,
    kernels_path: Path,
    N: int,
    M: int,
    K: int,
    warmup: int,
    iters: int,
    validate: bool = False,
    compare_optimize: bool = False,
    kernel_variant: str = "tiled",
) -> BenchRecord:
    fname = "matmul_tiled.cl" if kernel_variant == "tiled" else "matmul.cl"
    prg = _build_program(ctx, kernels_path, fname)
    mf = cl.mem_flags
    # A: (N,M), B: (M,K), C: (N,K)
    A = np.random.randn(N, M).astype(np.float32)
    B = np.random.randn(M, K).astype(np.float32)
    C = np.empty((N, K), dtype=np.float32)

    A_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
    B_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
    C_buf = cl.Buffer(ctx, mf.WRITE_ONLY, C.nbytes)

    if kernel_variant == "tiled":
        kn = cl.Kernel(prg, "matmul_tiled")
        lsz = (16, 16)
        gsz = ((K + 15) // 16 * 16, (N + 15) // 16 * 16)

        def launch():
            kn.set_args(A_buf, B_buf, C_buf, np.int32(N), np.int32(M), np.int32(K))
            return cl.enqueue_nd_range_kernel(q, kn, gsz, lsz)

    else:
        kn = prg.matmul_kernel
        gsz = (K, N)
        lsz = None

        def launch():
            return kn(q, gsz, lsz, A_buf, B_buf, C_buf, np.int32(N), np.int32(M), np.int32(K))

    for _ in range(warmup):
        launch().wait()

    evts = []
    for _ in range(iters):
        evts.append(launch())
    q.finish()

    stats = _measure_events_to_ms(evts)

    # NumPy baseline
    baseline = _measure_host_callable_ms(lambda: A @ B, warmup=1, iters=max(1, iters))
    spd = (baseline["mean"] / stats["mean"]) if (baseline.get("mean") and stats["mean"] > 0) else None

    # Validation
    vpass = vmax = vrmax = None
    if validate:
        C_host = np.empty_like(C)
        cl.enqueue_copy(q, C_host, C_buf).wait()
        ref = (A @ B).astype(np.float32)
        diff = np.abs(C_host - ref)
        vmax = float(np.max(diff))
        denom = np.maximum(np.abs(ref), 1e-6)
        vrmax = float(np.max(diff / denom))
        vpass = bool(np.allclose(C_host, ref, atol=1e-3, rtol=1e-3))

    # Perf metrics
    secs = stats["mean"] * 1e-3
    flops = float(2 * N * M * K)
    bytes_moved = float((N * M + M * K + N * K) * 4)
    gflops = _safe_div(flops, secs) / 1e9 if secs > 0 else None
    gibps = _safe_div(bytes_moved, secs) / (1024**3) if secs > 0 else None

    # Optional compare via @uhop.optimize("matmul")
    uhop_mean = None
    uhop_spd = None
    if compare_optimize:
        try:
            from uhop import optimize as _opt

            @_opt("matmul")
            def _naive(a, b):
                return a @ b

            # Warm + measure
            _ = _naive(A, B)
            mm = []
            import time

            for _ in range(max(1, iters)):
                t0 = time.perf_counter()
                _ = _naive(A, B)
                mm.append((time.perf_counter() - t0) * 1000.0)
            uhop_mean = float(statistics.mean(mm))
            uhop_spd = (
                (baseline["mean"] / uhop_mean) if (baseline.get("mean") and uhop_mean and uhop_mean > 0) else None
            )
        except Exception:
            uhop_mean = None
            uhop_spd = None

    return BenchRecord(
        op="matmul",
        shape=(N, M, K),
        mean_ms=stats["mean"],
        median_ms=stats["median"],
        min_ms=stats["min"],
        max_ms=stats["max"],
        std_ms=stats["std"],
        iters=iters,
        baseline_mean_ms=baseline.get("mean"),
        speedup_vs_baseline=spd,
        gflops=gflops,
        gib_per_s=gibps,
        validated=vpass,
        max_abs_err=vmax,
        max_rel_err=vrmax,
        uhop_mean_ms=uhop_mean,
        uhop_speedup_vs_naive=uhop_spd,
    )


def bench_conv2d(
    ctx: Any,
    q: Any,
    kernels_path: Path,
    B: int,
    C: int,
    H: int,
    W: int,
    K: int,
    R: int,
    S: int,
    stride: int,
    pad: int,
    warmup: int,
    iters: int,
    validate: bool = False,
    compare_optimize: bool = False,
) -> BenchRecord:
    prg = _build_program(ctx, kernels_path, "conv2d.cl")
    mf = cl.mem_flags
    outH = (H + 2 * pad - R) // stride + 1
    outW = (W + 2 * pad - S) // stride + 1

    inp = np.random.randn(B, C, H, W).astype(np.float32)
    w = np.random.randn(K, C, R, S).astype(np.float32)
    out = np.empty((B, K, outH, outW), dtype=np.float32)

    inp_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=inp)
    w_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=w)
    out_buf = cl.Buffer(ctx, mf.WRITE_ONLY, out.nbytes)

    gsz = (outW, outH, B * K)

    def launch():
        return prg.conv2d(
            q,
            gsz,
            None,
            inp_buf,
            w_buf,
            out_buf,
            np.int32(B),
            np.int32(C),
            np.int32(H),
            np.int32(W),
            np.int32(K),
            np.int32(R),
            np.int32(S),
            np.int32(outH),
            np.int32(outW),
            np.int32(stride),
            np.int32(pad),
        )

    for _ in range(warmup):
        launch().wait()

    evts = []
    for _ in range(iters):
        evts.append(launch())
    q.finish()

    stats = _measure_events_to_ms(evts)

    # CPU baseline using torch.nn.functional.conv2d
    import torch

    inp_t = torch.from_numpy(inp)
    w_t = torch.from_numpy(w)

    def baseline():
        torch.nn.functional.conv2d(inp_t, w_t, stride=stride, padding=pad)

    base = _measure_host_callable_ms(baseline, warmup=1, iters=max(1, iters))
    spd = (base["mean"] / stats["mean"]) if (base.get("mean") and stats["mean"] > 0) else None

    # Validation
    vpass = vmax = vrmax = None
    if validate:
        out_host = np.empty_like(out)
        cl.enqueue_copy(q, out_host, out_buf).wait()
        ref = np.asarray(
            torch.nn.functional.conv2d(inp_t, w_t, stride=stride, padding=pad).numpy(),
            dtype=np.float32,
        )
        diff = np.abs(out_host - ref)
        vmax = float(np.max(diff))
        denom = np.maximum(np.abs(ref), 1e-6)
        vrmax = float(np.max(diff / denom))
        vpass = bool(np.allclose(out_host, ref, atol=1e-3, rtol=1e-3))

    # Perf metrics
    secs = stats["mean"] * 1e-3
    outH = (H + 2 * pad - R) // stride + 1
    outW = (W + 2 * pad - S) // stride + 1
    flops = float(B) * K * outH * outW * C * R * S * 2.0
    bytes_moved = float((B * C * H * W + K * C * R * S + B * K * outH * outW) * 4)
    gflops = _safe_div(flops, secs) / 1e9 if secs > 0 else None
    gibps = _safe_div(bytes_moved, secs) / (1024**3) if secs > 0 else None

    # Optional compare via @uhop.optimize("conv2d")
    uhop_mean = None
    uhop_spd = None
    if compare_optimize:
        try:
            from uhop import optimize as _opt

            @_opt("conv2d")
            def _naive(x, w, stride=1, padding=0):
                import torch

                return torch.nn.functional.conv2d(x, w, stride=stride, padding=padding)

            # inputs remain torch tensors
            # Warm + measure
            _ = _naive(inp_t, w_t, stride=stride, padding=pad)
            mm = []
            import time

            for _ in range(max(1, iters)):
                t0 = time.perf_counter()
                _ = _naive(inp_t, w_t, stride=stride, padding=pad)
                mm.append((time.perf_counter() - t0) * 1000.0)
            uhop_mean = float(statistics.mean(mm))
            uhop_spd = (base["mean"] / uhop_mean) if (base.get("mean") and uhop_mean and uhop_mean > 0) else None
        except Exception:
            uhop_mean = None
            uhop_spd = None

    return BenchRecord(
        op="conv2d",
        shape=(B, C, H, W, K, R, S, stride, pad),
        mean_ms=stats["mean"],
        median_ms=stats["median"],
        min_ms=stats["min"],
        max_ms=stats["max"],
        std_ms=stats["std"],
        iters=iters,
        baseline_mean_ms=base.get("mean"),
        speedup_vs_baseline=spd,
        gflops=gflops,
        gib_per_s=gibps,
        validated=vpass,
        max_abs_err=vmax,
        max_rel_err=vrmax,
        uhop_mean_ms=uhop_mean,
        uhop_speedup_vs_naive=uhop_spd,
    )


def bench_conv2d_relu(
    B: int,
    C: int,
    H: int,
    W: int,
    K: int,
    R: int,
    S: int,
    stride: int,
    pad: int,
    warmup: int,
    iters: int,
    validate: bool = False,
) -> BenchRecord:
    """Benchmark fused OpenCL conv2d+relu vs separate conv2d then relu.

    Returns a BenchRecord where baseline_mean_ms represents the separate conv2d->relu time,
    and mean_ms is the fused time. Speedup is baseline/fused.
    """
    import time

    import torch

    from uhop.backends.opencl_backend import (
        opencl_conv2d,
        opencl_conv2d_relu,
        opencl_relu,
    )

    outH = (H + 2 * pad - R) // stride + 1
    outW = (W + 2 * pad - S) // stride + 1
    x = np.random.randn(B, C, H, W).astype(np.float32)
    w = np.random.randn(K, C, R, S).astype(np.float32)

    # Warmup fused and separate
    for _ in range(max(0, warmup)):
        _ = opencl_conv2d_relu(x, w, stride=stride, padding=pad)
        y = opencl_conv2d(x, w, stride=stride, padding=pad)
        _ = opencl_relu(y.reshape(-1)).reshape(B, K, outH, outW)

    # Measure fused
    fused_ts = []
    for _ in range(max(1, iters)):
        t0 = time.perf_counter()
        yf = opencl_conv2d_relu(x, w, stride=stride, padding=pad)
        fused_ts.append((time.perf_counter() - t0) * 1000.0)

    # Measure separate
    sep_ts = []
    for _ in range(max(1, iters)):
        t0 = time.perf_counter()
        y = opencl_conv2d(x, w, stride=stride, padding=pad)
        y = opencl_relu(y.reshape(-1)).reshape(B, K, outH, outW)
        sep_ts.append((time.perf_counter() - t0) * 1000.0)

    stats_f = {
        "mean": float(statistics.mean(fused_ts)),
        "median": float(statistics.median(fused_ts)),
        "min": float(min(fused_ts)),
        "max": float(max(fused_ts)),
        "std": float(statistics.stdev(fused_ts)) if len(fused_ts) > 1 else 0.0,
    }
    stats_s = {
        "mean": float(statistics.mean(sep_ts)),
        "median": float(statistics.median(sep_ts)),
        "min": float(min(sep_ts)),
        "max": float(max(sep_ts)),
        "std": float(statistics.stdev(sep_ts)) if len(sep_ts) > 1 else 0.0,
    }

    spd = (stats_s["mean"] / stats_f["mean"]) if (stats_s["mean"] > 0 and stats_f["mean"] > 0) else None

    # Validation vs base ops (torch conv2d then relu)
    vpass = vmax = vrmax = None
    if validate:
        inp_t = torch.from_numpy(x)
        w_t = torch.from_numpy(w)
        ref = (
            torch.nn.functional.relu(torch.nn.functional.conv2d(inp_t, w_t, stride=stride, padding=pad))
            .numpy()
            .astype(np.float32)
        )
        diff = np.abs(yf - ref)
        vmax = float(np.max(diff))
        denom = np.maximum(np.abs(ref), 1e-6)
        vrmax = float(np.max(diff / denom))
        vpass = bool(np.allclose(yf, ref, atol=1e-3, rtol=1e-3))

    # Perf metric proxy: conv flops dominate; relu cost negligible here
    secs = stats_f["mean"] * 1e-3
    flops = float(B) * K * outH * outW * C * R * S * 2.0
    gflops = _safe_div(flops, secs) / 1e9 if secs > 0 else None

    return BenchRecord(
        op="conv2d_relu",
        shape=(B, C, H, W, K, R, S, stride, pad),
        mean_ms=stats_f["mean"],
        median_ms=stats_f["median"],
        min_ms=stats_f["min"],
        max_ms=stats_f["max"],
        std_ms=stats_f["std"],
        iters=iters,
        baseline_mean_ms=stats_s["mean"],
        speedup_vs_baseline=spd,
        gflops=gflops,
        validated=vpass,
        max_abs_err=vmax,
        max_rel_err=vrmax,
    )


def bench_conv2d_input_grad(
    B: int,
    C: int,
    H: int,
    W: int,
    K: int,
    R: int,
    S: int,
    stride: int,
    pad: int,
    warmup: int,
    iters: int,
    validate: bool = False,
) -> BenchRecord:
    import time

    import torch

    from uhop.backends.opencl_backend import opencl_conv2d_backward_input

    outH = (H + 2 * pad - R) // stride + 1
    outW = (W + 2 * pad - S) // stride + 1

    x = np.random.randn(B, C, H, W).astype(np.float32)
    w = np.random.randn(K, C, R, S).astype(np.float32)
    grad_out = np.random.randn(B, K, outH, outW).astype(np.float32)

    # Warmup
    for _ in range(max(0, warmup)):
        _ = opencl_conv2d_backward_input((B, C, H, W), w, grad_out, stride=stride, padding=pad)

    # Measure host time (backend handles device profiling internally where applicable)
    ts = []
    for _ in range(max(1, iters)):
        t0 = time.perf_counter()
        gi = opencl_conv2d_backward_input((B, C, H, W), w, grad_out, stride=stride, padding=pad)
        ts.append((time.perf_counter() - t0) * 1000.0)

    stats = {
        "mean": float(statistics.mean(ts)),
        "median": float(statistics.median(ts)),
        "min": float(min(ts)),
        "max": float(max(ts)),
        "std": float(statistics.stdev(ts)) if len(ts) > 1 else 0.0,
    }

    # Baseline with torch autograd (CPU)
    xt = torch.tensor(x, requires_grad=True)
    wt = torch.tensor(w, requires_grad=True)
    got = torch.tensor(grad_out)

    def baseline():
        xt.grad = None
        wt.grad = None
        y = torch.nn.functional.conv2d(xt, wt, stride=stride, padding=pad)
        y.backward(got)
        return xt.grad

    base = _measure_host_callable_ms(baseline, warmup=1, iters=max(1, iters))
    spd = (base["mean"] / stats["mean"]) if (base.get("mean") and stats["mean"] > 0) else None

    vpass = vmax = vrmax = None
    if validate:
        ref = baseline().detach().numpy().astype(np.float32)
        diff = np.abs(gi - ref)
        vmax = float(np.max(diff))
        denom = np.maximum(np.abs(ref), 1e-6)
        vrmax = float(np.max(diff / denom))
        vpass = bool(np.allclose(gi, ref, atol=1e-3, rtol=1e-3))

    flops = float(B) * K * outH * outW * C * R * S * 2.0  # same as fwd
    secs = stats["mean"] * 1e-3
    gflops = _safe_div(flops, secs) / 1e9 if secs > 0 else None

    return BenchRecord(
        op="conv2d_input_grad",
        shape=(B, C, H, W, K, R, S, stride, pad),
        mean_ms=stats["mean"],
        median_ms=stats["median"],
        min_ms=stats["min"],
        max_ms=stats["max"],
        std_ms=stats["std"],
        iters=iters,
        baseline_mean_ms=base.get("mean"),
        speedup_vs_baseline=spd,
        gflops=gflops,
        validated=vpass,
        max_abs_err=vmax,
        max_rel_err=vrmax,
    )


def bench_conv2d_weight_grad(
    B: int,
    C: int,
    H: int,
    W: int,
    K: int,
    R: int,
    S: int,
    stride: int,
    pad: int,
    warmup: int,
    iters: int,
    validate: bool = False,
) -> BenchRecord:
    import time

    import torch

    from uhop.backends.opencl_backend import opencl_conv2d_backward_weight

    outH = (H + 2 * pad - R) // stride + 1
    outW = (W + 2 * pad - S) // stride + 1

    x = np.random.randn(B, C, H, W).astype(np.float32)
    w_shape = (K, C, R, S)
    grad_out = np.random.randn(B, K, outH, outW).astype(np.float32)

    # Warmup
    for _ in range(max(0, warmup)):
        _ = opencl_conv2d_backward_weight(x, grad_out, w_shape, stride=stride, padding=pad)

    # Measure host time
    ts = []
    for _ in range(max(1, iters)):
        t0 = time.perf_counter()
        gw = opencl_conv2d_backward_weight(x, grad_out, w_shape, stride=stride, padding=pad)
        ts.append((time.perf_counter() - t0) * 1000.0)

    stats = {
        "mean": float(statistics.mean(ts)),
        "median": float(statistics.median(ts)),
        "min": float(min(ts)),
        "max": float(max(ts)),
        "std": float(statistics.stdev(ts)) if len(ts) > 1 else 0.0,
    }

    # Baseline with torch autograd (CPU)
    xt = torch.tensor(x, requires_grad=True)
    gt = torch.tensor(grad_out)
    wt = torch.nn.Parameter(torch.zeros(w_shape, dtype=torch.float32))

    def baseline():
        if wt.grad is not None:
            wt.grad.zero_()
        y = torch.nn.functional.conv2d(xt, wt, stride=stride, padding=pad)
        y.backward(gt)
        return wt.grad

    base = _measure_host_callable_ms(baseline, warmup=1, iters=max(1, iters))
    spd = (base["mean"] / stats["mean"]) if (base.get("mean") and stats["mean"] > 0) else None

    vpass = vmax = vrmax = None
    if validate:
        ref = baseline().detach().numpy().astype(np.float32)
        diff = np.abs(gw - ref)
        vmax = float(np.max(diff))
        denom = np.maximum(np.abs(ref), 1e-6)
        vrmax = float(np.max(diff / denom))
        vpass = bool(np.allclose(gw, ref, atol=1e-3, rtol=1e-3))

    flops = float(B) * K * outH * outW * C * R * S * 2.0  # proxy
    secs = stats["mean"] * 1e-3
    gflops = _safe_div(flops, secs) / 1e9 if secs > 0 else None

    return BenchRecord(
        op="conv2d_weight_grad",
        shape=(B, C, H, W, K, R, S, stride, pad),
        mean_ms=stats["mean"],
        median_ms=stats["median"],
        min_ms=stats["min"],
        max_ms=stats["max"],
        std_ms=stats["std"],
        iters=iters,
        baseline_mean_ms=base.get("mean"),
        speedup_vs_baseline=spd,
        gflops=gflops,
        validated=vpass,
        max_abs_err=vmax,
        max_rel_err=vrmax,
    )


def bench_softmax(
    ctx: Any,
    q: Any,
    kernels_path: Path,
    batch: int,
    classes: int,
    warmup: int,
    iters: int,
    log: bool,
    validate: bool = False,
) -> BenchRecord:
    prg = _build_program(ctx, kernels_path, "softmax.cl")
    mf = cl.mem_flags
    x = np.random.randn(batch, classes).astype(np.float32)
    y = np.empty_like(x)
    x_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x)
    y_buf = cl.Buffer(ctx, mf.WRITE_ONLY, y.nbytes)

    gsz = (batch,)

    def launch():
        if log:
            return prg.logsoftmax_kernel(q, gsz, None, x_buf, y_buf, np.int32(batch), np.int32(classes))
        else:
            return prg.softmax_kernel(q, gsz, None, x_buf, y_buf, np.int32(batch), np.int32(classes))

    for _ in range(warmup):
        launch().wait()

    evts = []
    for _ in range(iters):
        evts.append(launch())
    q.finish()

    stats = _measure_events_to_ms(evts)

    # NumPy baseline
    if log:

        def baseline():
            m = np.max(x, axis=1, keepdims=True)
            logsum = np.log(np.sum(np.exp(x - m), axis=1, keepdims=True)) + m
            _ = x - logsum

    else:

        def baseline():
            m = np.max(x, axis=1, keepdims=True)
            ex = np.exp(x - m)
            _ = ex / np.sum(ex, axis=1, keepdims=True)

    base = _measure_host_callable_ms(baseline, warmup=1, iters=max(1, iters))
    spd = (base["mean"] / stats["mean"]) if (base.get("mean") and stats["mean"] > 0) else None
    # Validation
    vpass = vmax = vrmax = None
    if validate:
        y_host = np.empty_like(x)
        cl.enqueue_copy(q, y_host, y_buf).wait()
        # compute baseline
        if log:
            m = np.max(x, axis=1, keepdims=True)
            logsum = np.log(np.sum(np.exp(x - m), axis=1, keepdims=True)) + m
            ref = x - logsum
        else:
            m = np.max(x, axis=1, keepdims=True)
            ex = np.exp(x - m)
            ref = ex / np.sum(ex, axis=1, keepdims=True)
        ref = np.asarray(ref, dtype=np.float32)
        diff = np.abs(y_host - ref)
        vmax = float(np.max(diff))
        denom = np.maximum(np.abs(ref), 1e-6)
        vrmax = float(np.max(diff / denom))
        vpass = bool(np.allclose(y_host, ref, atol=1e-3, rtol=1e-3))

    return BenchRecord(
        op="logsoftmax" if log else "softmax",
        shape=(batch, classes),
        mean_ms=stats["mean"],
        median_ms=stats["median"],
        min_ms=stats["min"],
        max_ms=stats["max"],
        std_ms=stats["std"],
        iters=iters,
        baseline_mean_ms=base.get("mean"),
        speedup_vs_baseline=spd,
        validated=vpass,
        max_abs_err=vmax,
        max_rel_err=vrmax,
    )


def bench_pool2d(
    ctx: Any,
    q: Any,
    kernels_path: Path,
    op: str,
    B: int,
    C: int,
    H: int,
    W: int,
    k: int,
    stride: int,
    pad: int,
    warmup: int,
    iters: int,
    validate: bool = False,
) -> BenchRecord:
    prg = _build_program(ctx, kernels_path, "pooling.cl")
    mf = cl.mem_flags
    outH = (H + 2 * pad - k) // stride + 1
    outW = (W + 2 * pad - k) // stride + 1

    inp = np.random.randn(B, C, H, W).astype(np.float32)
    out = np.empty((B, C, outH, outW), dtype=np.float32)
    inp_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=inp)
    out_buf = cl.Buffer(ctx, mf.WRITE_ONLY, out.nbytes)

    total = B * C * outH * outW

    if op == "maxpool2d":

        def launch():
            return prg.maxpool2d_kernel(
                q,
                (total,),
                None,
                inp_buf,
                out_buf,
                np.int32(B),
                np.int32(C),
                np.int32(H),
                np.int32(W),
                np.int32(k),
                np.int32(k),
                np.int32(stride),
                np.int32(stride),
                np.int32(pad),
                np.int32(pad),
            )

    elif op == "avgpool2d":

        def launch():
            return prg.avgpool2d_kernel(
                q,
                (total,),
                None,
                inp_buf,
                out_buf,
                np.int32(B),
                np.int32(C),
                np.int32(H),
                np.int32(W),
                np.int32(k),
                np.int32(k),
                np.int32(stride),
                np.int32(stride),
                np.int32(pad),
                np.int32(pad),
                np.int32(0),
            )

    else:
        raise ValueError(f"Unsupported pool op: {op}")

    for _ in range(warmup):
        launch().wait()

    evts = []
    for _ in range(iters):
        evts.append(launch())
    q.finish()

    stats = _measure_events_to_ms(evts)

    # Baseline with torch
    import torch

    inp_t = torch.from_numpy(inp)

    def baseline():
        if op == "maxpool2d":
            torch.nn.functional.max_pool2d(inp_t, kernel_size=k, stride=stride, padding=pad)
        else:
            torch.nn.functional.avg_pool2d(
                inp_t,
                kernel_size=k,
                stride=stride,
                padding=pad,
                ceil_mode=False,
                count_include_pad=False,
            )

    base = _measure_host_callable_ms(baseline, warmup=1, iters=max(1, iters))
    spd = (base["mean"] / stats["mean"]) if (base.get("mean") and stats["mean"] > 0) else None

    # Validation
    vpass = vmax = vrmax = None
    if validate:
        out_host = np.empty_like(out)
        cl.enqueue_copy(q, out_host, out_buf).wait()
        if op == "maxpool2d":
            ref = torch.nn.functional.max_pool2d(inp_t, kernel_size=k, stride=stride, padding=pad)
        else:
            ref = torch.nn.functional.avg_pool2d(
                inp_t,
                kernel_size=k,
                stride=stride,
                padding=pad,
                ceil_mode=False,
                count_include_pad=False,
            )
        ref = ref.numpy().astype(np.float32)
        diff = np.abs(out_host - ref)
        vmax = float(np.max(diff))
        denom = np.maximum(np.abs(ref), 1e-6)
        vrmax = float(np.max(diff / denom))
        vpass = bool(np.allclose(out_host, ref, atol=1e-3, rtol=1e-3))

    return BenchRecord(
        op=op,
        shape=(B, C, H, W, k, stride, pad),
        mean_ms=stats["mean"],
        median_ms=stats["median"],
        min_ms=stats["min"],
        max_ms=stats["max"],
        std_ms=stats["std"],
        iters=iters,
        baseline_mean_ms=base.get("mean"),
        speedup_vs_baseline=spd,
        validated=vpass,
        max_abs_err=vmax,
        max_rel_err=vrmax,
    )


def bench_depthwise_conv2d(
    ctx: Any,
    q: Any,
    kernels_path: Path,
    B: int,
    C: int,
    H: int,
    W: int,
    KH: int,
    KW: int,
    stride: int,
    pad: int,
    warmup: int,
    iters: int,
    validate: bool = False,
) -> BenchRecord:
    prg = _build_program(ctx, kernels_path, "depthwise_conv2d.cl")
    mf = cl.mem_flags
    outH = (H + 2 * pad - KH) // stride + 1
    outW = (W + 2 * pad - KW) // stride + 1

    inp = np.random.randn(B, C, H, W).astype(np.float32)
    w = np.random.randn(C, KH, KW).astype(np.float32)
    bias = np.random.randn(C).astype(np.float32)
    out = np.empty((B, C, outH, outW), dtype=np.float32)

    inp_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=inp)
    w_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=w)
    b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=bias)
    out_buf = cl.Buffer(ctx, mf.WRITE_ONLY, out.nbytes)

    gsz = (outW, outH, B * C)

    def launch():
        return prg.depthwise_conv2d(
            q,
            gsz,
            None,
            inp_buf,
            w_buf,
            b_buf,
            out_buf,
            np.int32(B),
            np.int32(C),
            np.int32(H),
            np.int32(W),
            np.int32(KH),
            np.int32(KW),
            np.int32(stride),
            np.int32(stride),
            np.int32(pad),
            np.int32(pad),
        )

    for _ in range(warmup):
        launch().wait()

    evts = []
    for _ in range(iters):
        evts.append(launch())
    q.finish()

    stats = _measure_events_to_ms(evts)

    # Baseline with torch depthwise conv (groups=C)
    import torch

    inp_t = torch.from_numpy(inp)
    w_t = torch.from_numpy(w.reshape(C, 1, KH, KW))
    b_t = torch.from_numpy(bias)

    def baseline():
        torch.nn.functional.conv2d(inp_t, w_t, b_t, stride=stride, padding=pad, groups=C)

    base = _measure_host_callable_ms(baseline, warmup=1, iters=max(1, iters))
    spd = (base["mean"] / stats["mean"]) if (base.get("mean") and stats["mean"] > 0) else None

    # Perf metrics
    secs = stats["mean"] * 1e-3
    flops = float(B) * C * outH * outW * (KH * KW) * 2.0
    bytes_moved = float((B * C * H * W + C * KH * KW + B * C * outH * outW) * 4)
    gflops = _safe_div(flops, secs) / 1e9 if secs > 0 else None
    gibps = _safe_div(bytes_moved, secs) / (1024**3) if secs > 0 else None

    # Validation
    vpass = vmax = vrmax = None
    if validate:
        out_host = np.empty_like(out)
        cl.enqueue_copy(q, out_host, out_buf).wait()
        ref = (
            torch.nn.functional.conv2d(inp_t, w_t, b_t, stride=stride, padding=pad, groups=C).numpy().astype(np.float32)
        )
        diff = np.abs(out_host - ref)
        vmax = float(np.max(diff))
        denom = np.maximum(np.abs(ref), 1e-6)
        vrmax = float(np.max(diff / denom))
        vpass = bool(np.allclose(out_host, ref, atol=1e-3, rtol=1e-3))

    return BenchRecord(
        op="depthwise_conv2d",
        shape=(B, C, H, W, KH, KW, stride, pad),
        mean_ms=stats["mean"],
        median_ms=stats["median"],
        min_ms=stats["min"],
        max_ms=stats["max"],
        std_ms=stats["std"],
        iters=iters,
        baseline_mean_ms=base.get("mean"),
        speedup_vs_baseline=spd,
        gflops=gflops,
        gib_per_s=gibps,
        validated=vpass,
        max_abs_err=vmax,
        max_rel_err=vrmax,
    )


def bench_groupnorm(
    ctx: Any,
    q: Any,
    kernels_path: Path,
    B: int,
    C: int,
    H: int,
    W: int,
    groups: int,
    eps: float,
    warmup: int,
    iters: int,
    validate: bool = False,
) -> BenchRecord:
    prg = _build_program(ctx, kernels_path, "groupnorm.cl")
    mf = cl.mem_flags
    X = np.random.randn(B, C, H, W).astype(np.float32)
    gamma = np.random.randn(C).astype(np.float32)
    beta = np.random.randn(C).astype(np.float32)
    Y = np.empty_like(X)

    X_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=X)
    g_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=gamma)
    b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=beta)
    Y_buf = cl.Buffer(ctx, mf.WRITE_ONLY, Y.nbytes)

    # Adjust groups to divide C
    def nearest_divisor(c: int, target: int) -> int:
        if c % target == 0:
            return target
        for d in sorted([d for d in range(target, 0, -1) if c % d == 0], reverse=True):
            return d
        return 1

    g = nearest_divisor(C, groups)

    gsz = (H * W, C, B)

    def launch():
        return prg.group_norm(
            q,
            gsz,
            None,
            X_buf,
            g_buf,
            b_buf,
            Y_buf,
            np.int32(B),
            np.int32(C),
            np.int32(H),
            np.int32(W),
            np.int32(g),
            np.float32(eps),
        )

    for _ in range(warmup):
        launch().wait()

    evts = []
    for _ in range(iters):
        evts.append(launch())
    q.finish()

    stats = _measure_events_to_ms(evts)

    # Baseline with torch GroupNorm
    import torch

    m = torch.nn.GroupNorm(num_groups=g, num_channels=C, eps=eps, affine=True)
    with torch.no_grad():
        m.weight = torch.nn.Parameter(torch.from_numpy(gamma))
        m.bias = torch.nn.Parameter(torch.from_numpy(beta))
    X_t = torch.from_numpy(X)

    def baseline():
        _ = m(X_t)

    base = _measure_host_callable_ms(baseline, warmup=1, iters=max(1, iters))
    spd = (base["mean"] / stats["mean"]) if (base.get("mean") and stats["mean"] > 0) else None

    # Memory bandwidth proxy (read+write)
    secs = stats["mean"] * 1e-3
    bytes_moved = float((2 * B * C * H * W + 2 * C) * 4)  # X->Y plus gamma/beta read
    gibps = _safe_div(bytes_moved, secs) / (1024**3) if secs > 0 else None

    # Validation
    vpass = vmax = vrmax = None
    if validate:
        Y_host = np.empty_like(Y)
        cl.enqueue_copy(q, Y_host, Y_buf).wait()
        # Compute reference matching kernel semantics: per-pixel group norm (normalize across channels within group for each spatial position)
        gs = g
        gsize = C // gs
        Xr = X.reshape(B, gs, gsize, H, W)
        m = Xr.mean(axis=2)
        var = (Xr**2).mean(axis=2) - m**2
        inv = 1.0 / np.sqrt(np.abs(var) + eps)
        # Broadcast to channels
        m = m[:, :, None, :, :]
        inv = inv[:, :, None, :, :]
        gamma_r = gamma.reshape(gs, gsize)[None, :, :, None, None]
        beta_r = beta.reshape(gs, gsize)[None, :, :, None, None]
        Yr = gamma_r * ((Xr - m) * inv) + beta_r
        ref = Yr.reshape(B, C, H, W).astype(np.float32)
        diff = np.abs(Y_host - ref)
        vmax = float(np.max(diff))
        denom = np.maximum(np.abs(ref), 1e-6)
        vrmax = float(np.max(diff / denom))
        vpass = bool(np.allclose(Y_host, ref, atol=1e-3, rtol=1e-3))

    return BenchRecord(
        op="group_norm",
        shape=(B, C, H, W, g),
        mean_ms=stats["mean"],
        median_ms=stats["median"],
        min_ms=stats["min"],
        max_ms=stats["max"],
        std_ms=stats["std"],
        iters=iters,
        baseline_mean_ms=base.get("mean"),
        speedup_vs_baseline=spd,
        gflops=None,
        gib_per_s=gibps,
        validated=vpass,
        max_abs_err=vmax,
        max_rel_err=vrmax,
    )


DEFAULT_OPS = [
    # Elementwise family
    "elementwise_add",
    "elementwise_mul",
    "elementwise_exp",
    "elementwise_log",
    "elementwise_sqrt",
    # Activations
    "relu",
    "gelu",
    "silu",
    # Reductions
    "reduce_sum",
    # Matrix and conv
    "matmul",
    "conv2d",
    # Fused conv+relu (opt-in via --ops)
    # "conv2d_relu",
    # Backward (opt-in via --ops)
    # "conv2d_input_grad",
    # "conv2d_weight_grad",
    # Pooling
    "maxpool2d",
    "avgpool2d",
    # Depthwise conv and norms
    "depthwise_conv2d",
    "group_norm",
    # Softmax
    "softmax",
]


def main():
    parser = argparse.ArgumentParser(description="UHOP OpenCL Operator Benchmarks")
    parser.add_argument(
        "--ops",
        type=str,
        default="all",
        help="Comma-separated ops or 'all' for a curated set",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument(
        "--size",
        type=int,
        default=1 << 20,
        help="Vector length for elementwise/reduce ops",
    )
    parser.add_argument("--matmul-shape", type=str, default="512,512,512", help="N,M,K for matmul")
    parser.add_argument(
        "--matmul-kernel",
        type=str,
        default="tiled",
        choices=["tiled", "naive"],
        help="Matmul kernel variant to use",
    )
    parser.add_argument(
        "--conv-shape",
        type=str,
        default="8,32,112,112,64,3,3,1,1",
        help="B,C,H,W,K,R,S,stride,pad for conv2d",
    )
    parser.add_argument(
        "--conv-shapes",
        type=str,
        default=None,
        help="Semicolon-separated list of conv2d shapes for batch compare, e.g., '8,32,112,112,64,3,3,1,1;8,64,56,56,64,3,3,1,1'",
    )
    parser.add_argument(
        "--softmax-shape",
        type=str,
        default="1024,1000",
        help="batch,classes for softmax",
    )
    parser.add_argument(
        "--pool-shape",
        type=str,
        default="8,32,112,112,3,2,1",
        help="B,C,H,W,k,stride,pad for pool2d ops",
    )
    parser.add_argument(
        "--depthwise-shape",
        type=str,
        default="8,32,112,112,3,3,1,1",
        help="B,C,H,W,KH,KW,stride,pad for depthwise_conv2d",
    )
    parser.add_argument(
        "--groupnorm-shape",
        type=str,
        default="8,32,56,56",
        help="B,C,H,W for group_norm",
    )
    parser.add_argument(
        "--groupnorm-groups",
        type=int,
        default=8,
        help="Number of groups for group_norm",
    )
    parser.add_argument("--groupnorm-eps", type=float, default=1e-5, help="Epsilon for group_norm")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate device outputs vs host baselines",
    )
    parser.add_argument(
        "--compare-optimize",
        action="store_true",
        help="Compare @uhop.optimize ops vs naive (supported ops only)",
    )
    parser.add_argument("--output", type=str, help="Optional JSON output path")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    if cl is None:
        raise SystemExit("pyopencl not available. Install it to run benchmarks.")

    ctx, q = _mk_queue()
    kpath = Path(__file__).resolve().parents[1] / "kernels" / "opencl"

    if args.ops.strip().lower() == "all":
        ops = DEFAULT_OPS
    else:
        ops = [s.strip() for s in args.ops.split(",") if s.strip()]

    # Parse shapes
    N = int(args.size)
    Nmm, Mmm, Kmm = [int(x) for x in args.matmul_shape.split(",")]
    Bc, Cc, Hc, Wc, Kc, Rc, Sc, stridec, padc = [int(x) for x in args.conv_shape.split(",")]
    Sb, Cb = [int(x) for x in args.softmax_shape.split(",")]
    Bp, Cp, Hp, Wp, kp, sp, pp = [int(x) for x in args.pool_shape.split(",")]
    Bd, Cd, Hd, Wd, KHd, KWd, sd, pd = [int(x) for x in args.depthwise_shape.split(",")]
    Bg, Cg, Hg, Wg = [int(x) for x in args.groupnorm_shape.split(",")]
    Gg, Eg = int(args.groupnorm_groups), float(args.groupnorm_eps)

    results: List[BenchRecord] = []

    compare_rows = []  # for conv2d_relu_compare
    for op in ops:
        if args.verbose:
            print(f"Benchmarking {op}...")
        if op.startswith("elementwise_"):
            results.append(
                bench_elementwise(
                    ctx,
                    q,
                    kpath,
                    op,
                    N,
                    args.warmup,
                    args.iterations,
                    validate=args.validate,
                )
            )
        elif op in ("relu", "gelu", "silu"):
            results.append(
                bench_activations(
                    ctx,
                    q,
                    kpath,
                    op,
                    N,
                    args.warmup,
                    args.iterations,
                    validate=args.validate,
                )
            )
        elif op == "reduce_sum":
            results.append(
                bench_reduce_sum(
                    ctx,
                    q,
                    kpath,
                    N,
                    args.warmup,
                    args.iterations,
                    validate=args.validate,
                )
            )
        elif op == "matmul":
            results.append(
                bench_matmul(
                    ctx,
                    q,
                    kpath,
                    Nmm,
                    Mmm,
                    Kmm,
                    args.warmup,
                    args.iterations,
                    validate=args.validate,
                    compare_optimize=args.compare_optimize,
                    kernel_variant=args.matmul_kernel,
                )
            )
        elif op == "conv2d":
            results.append(
                bench_conv2d(
                    ctx,
                    q,
                    kpath,
                    Bc,
                    Cc,
                    Hc,
                    Wc,
                    Kc,
                    Rc,
                    Sc,
                    stridec,
                    padc,
                    args.warmup,
                    args.iterations,
                    validate=args.validate,
                    compare_optimize=args.compare_optimize,
                )
            )
        elif op == "conv2d_relu":
            results.append(
                bench_conv2d_relu(
                    Bc,
                    Cc,
                    Hc,
                    Wc,
                    Kc,
                    Rc,
                    Sc,
                    stridec,
                    padc,
                    args.warmup,
                    args.iterations,
                    validate=args.validate,
                )
            )
        elif op == "conv2d_relu_compare":
            shapes = []
            if args.conv_shapes:
                for part in args.conv_shapes.split(";"):
                    part = part.strip()
                    if not part:
                        continue
                    try:
                        b, c, h, w, k, r, s, st, pd = [int(x) for x in part.split(",")]
                        shapes.append((b, c, h, w, k, r, s, st, pd))
                    except Exception:
                        pass
            if not shapes:
                shapes = [(Bc, Cc, Hc, Wc, Kc, Rc, Sc, stridec, padc)]
            for b, c, h, w, k, r, s, st, pd in shapes:
                rec = bench_conv2d_relu(
                    b,
                    c,
                    h,
                    w,
                    k,
                    r,
                    s,
                    st,
                    pd,
                    args.warmup,
                    args.iterations,
                    validate=args.validate,
                )
                results.append(rec)
                compare_rows.append(
                    {
                        "shape": (b, c, h, w, k, r, s, st, pd),
                        "fused_ms": rec.mean_ms,
                        "separate_ms": rec.baseline_mean_ms,
                        "speedup": rec.speedup_vs_baseline,
                        "validated": rec.validated,
                    }
                )
        elif op == "conv2d_input_grad":
            results.append(
                bench_conv2d_input_grad(
                    Bc,
                    Cc,
                    Hc,
                    Wc,
                    Kc,
                    Rc,
                    Sc,
                    stridec,
                    padc,
                    args.warmup,
                    args.iterations,
                    validate=args.validate,
                )
            )
        elif op == "conv2d_weight_grad":
            results.append(
                bench_conv2d_weight_grad(
                    Bc,
                    Cc,
                    Hc,
                    Wc,
                    Kc,
                    Rc,
                    Sc,
                    stridec,
                    padc,
                    args.warmup,
                    args.iterations,
                    validate=args.validate,
                )
            )
        elif op == "softmax":
            results.append(
                bench_softmax(
                    ctx,
                    q,
                    kpath,
                    Sb,
                    Cb,
                    args.warmup,
                    args.iterations,
                    False,
                    validate=args.validate,
                )
            )
        elif op == "logsoftmax":
            results.append(
                bench_softmax(
                    ctx,
                    q,
                    kpath,
                    Sb,
                    Cb,
                    args.warmup,
                    args.iterations,
                    True,
                    validate=args.validate,
                )
            )
        elif op in ("maxpool2d", "avgpool2d"):
            results.append(
                bench_pool2d(
                    ctx,
                    q,
                    kpath,
                    op,
                    Bp,
                    Cp,
                    Hp,
                    Wp,
                    kp,
                    sp,
                    pp,
                    args.warmup,
                    args.iterations,
                    validate=args.validate,
                )
            )
        elif op == "depthwise_conv2d":
            results.append(
                bench_depthwise_conv2d(
                    ctx,
                    q,
                    kpath,
                    Bd,
                    Cd,
                    Hd,
                    Wd,
                    KHd,
                    KWd,
                    sd,
                    pd,
                    args.warmup,
                    args.iterations,
                    validate=args.validate,
                )
            )
        elif op == "group_norm":
            results.append(
                bench_groupnorm(
                    ctx,
                    q,
                    kpath,
                    Bg,
                    Cg,
                    Hg,
                    Wg,
                    Gg,
                    Eg,
                    args.warmup,
                    args.iterations,
                    validate=args.validate,
                )
            )
        else:
            print(f"[skip] Unsupported op: {op}")

    # Print summary
    print("\nOpenCL Benchmarks Summary:")
    for r in results:
        spd = f" x{(r.speedup_vs_baseline):.2f}" if r.speedup_vs_baseline else ""
        base = f", baseline={r.baseline_mean_ms:.3f} ms" if r.baseline_mean_ms is not None else ""
        perf = ""
        if r.gflops is not None:
            perf += f", {r.gflops:.1f} GFLOPS"
        if r.gib_per_s is not None:
            perf += f", {r.gib_per_s:.2f} GiB/s"
        val = ""
        if r.validated is not None:
            status = "PASS" if r.validated else "FAIL"
            val = f", validate={status} (max|rel={r.max_abs_err:.2e}|{r.max_rel_err:.2e})"
        uh = ""
        if r.uhop_mean_ms is not None:
            uh = f", uhop={r.uhop_mean_ms:.3f} ms"
            if r.uhop_speedup_vs_naive is not None:
                uh += f" (x{r.uhop_speedup_vs_naive:.2f})"
        print(
            f"- {r.op:14s} shape={r.shape}  mean={r.mean_ms:8.3f} ms  (±{r.std_ms:6.3f} ms){base}{spd}{perf}{val}{uh}"
        )

    # Optional compact table for conv2d_relu_compare
    if compare_rows:
        print("\nconv2d+relu fused vs separate (batch):")
        header = f"{'shape':>40s} | {'fused (ms)':>10s} | {'separate (ms)':>13s} | {'speedup':>7s} | {'val':>4s}"
        print(header)
        print("-" * len(header))
        for row in compare_rows:
            shp = row["shape"]
            fused_ms = row["fused_ms"] or 0.0
            sep_ms = row["separate_ms"] or 0.0
            spd = row["speedup"] or 0.0
            v = row["validated"]
            val = "PASS" if v else ("FAIL" if v is not None else "-")
            print(f"{str(shp):>40s} | {fused_ms:10.3f} | {sep_ms:13.3f} | x{spd:5.2f} | {val:>4s}")

    # Output JSON
    if args.output:
        outp = Path(args.output)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w") as f:
            json.dump([r.to_dict() for r in results], f, indent=2)
        if args.verbose:
            print(f"Saved report to {outp}")


if __name__ == "__main__":
    main()
