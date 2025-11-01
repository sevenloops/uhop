"""
Autotuner for elementwise ops (minimal PoC).

Features:
- Compile candidate kernels (from kernel template files)
- Benchmark each candidate using selected backend
- Store best (kernel_id + launch config) to cache (JSON)

This module is intentionally compact so it can be expanded to other ops.
"""

import json
import math
from pathlib import Path
from typing import Any, Dict

from jinja2 import Template

try:
    import cupy as cp  # type: ignore
except Exception:
    cp = None

import importlib

# Import backend wrappers explicitly to avoid package __getattr__ interception
cupy_wrapper = importlib.import_module("uhop.backends.cupy_wrapper")
opencl_wrapper = importlib.import_module("uhop.backends.opencl_wrapper")
hip_wrapper = importlib.import_module("uhop.backends.hip_wrapper")
metal_wrapper = importlib.import_module("uhop.backends.metal_wrapper")

CACHE_DIR = Path("uhop/cache")
KERNELS_DIR = Path("uhop/kernels")


def _ensure_cache_dir():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_path(device: str, op_name: str) -> Path:
    return CACHE_DIR / device / f"{op_name}.json"


def _load_cache(device: str, op_name: str):
    p = _cache_path(device, op_name)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return None
    return None


def _save_cache(device: str, op_name: str, data: Dict[str, Any]):
    p = _cache_path(device, op_name)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2))


def _render_kernel_template(path: Path, context: Dict[str, Any]) -> str:
    text = path.read_text()
    tpl = Template(text)
    return tpl.render(**context)


def _default_block_candidates():
    # typical block sizes to try (one-dimensional kernels)
    return [128, 256, 512, 1024]


def _cuda_dtype(dtype: str) -> str:
    # Map numpy-style dtype strings to CUDA/HIP C types
    d = dtype.lower()
    if d in ("float32", "float"):
        return "float"
    if d in ("float64", "double"):
        return "double"
    if d in ("int32", "int"):
        return "int"
    if d in ("int64", "long long"):
        return "long long"
    # fallback
    return "float"


def autotune_elementwise(
    op_name: str, size: int, dtype: str = "float32", device: str = "cuda"
):
    """
    Autotune elementwise op 'op_name' for a single flat size.
    Returns cached best config or runs tuning and caches the result.
    """
    cached = _load_cache(device, op_name)
    if cached:
        return cached

    # Select backend templates and wrappers
    mapping = {
        "cuda": {
            "dir": KERNELS_DIR / "cuda",
            "map": {
                "add": "elementwise_add.cu.jinja",
                "mul": "elementwise_add.cu.jinja",
            },
        },
        "opencl": {
            "dir": KERNELS_DIR / "opencl",
            "map": {
                "add": "elementwise_add.cl.jinja",
                "mul": "elementwise_add.cl.jinja",
            },
        },
        "hip": {
            "dir": KERNELS_DIR / "hip",
            "map": {
                "add": "elementwise_add.hip.jinja",
                "mul": "elementwise_add.hip.jinja",
            },
        },
        "metal": {
            "dir": KERNELS_DIR / "metal",
            "map": {
                "add": "elementwise_add.metal.jinja",
                "mul": "elementwise_add.metal.jinja",
            },
        },
    }

    backend = mapping.get(device)
    if backend is None:
        raise ValueError(f"Unsupported device/backend: {device}")

    templates_dir = backend["dir"]
    mapping_for_op = backend["map"]
    filename = mapping_for_op.get(op_name)
    if filename is None:
        raise ValueError(
            f"No kernel template mapped for op {op_name} on backend {device}"
        )

    template_path = templates_dir / filename
    if not template_path.exists():
        raise FileNotFoundError(f"Kernel template {template_path} not found.")

    best = {"latency_s": float("inf")}
    candidates = _default_block_candidates()

    # Prepare sample arrays
    import numpy as np

    a = np.random.rand(size).astype(dtype)
    b = np.random.rand(size).astype(dtype)

    # pre-create device arrays for CUDA/HIP
    if device == "cuda":
        if cp is None:
            raise RuntimeError("CUDA backend requested but CuPy is not available.")
        da = cp.asarray(a)
        db = cp.asarray(b)
        dout = cp.empty_like(da)
    elif device == "hip":
        if hip_wrapper.cp is None:
            raise RuntimeError("HIP backend requested but cupy-rocm is not available.")
        cp_rocm = hip_wrapper.cp
        da = cp_rocm.asarray(a)
        db = cp_rocm.asarray(b)
        dout = cp_rocm.empty_like(da)

    for block in candidates:
        threads = block
        grid = math.ceil(size / threads)
        # Map dtype token to backend language type
        if device in ("cuda", "hip"):
            dtype_token = _cuda_dtype(dtype)
        elif device == "opencl":
            dtype_token = "float" if "float" in dtype else "int"
        else:
            dtype_token = "float"
        context = {
            "OP_EXPR": "a[i] + b[i]" if op_name == "add" else "a[i] * b[i]",
            "KERNEL_NAME": "elem_op",
            "DTYPE": dtype_token,
        }

        try:
            source = _render_kernel_template(template_path, context)
        except Exception:
            continue

        try:
            if device == "cuda":
                kernel = cupy_wrapper.CupyKernel(source, "elem_op")
                args = (da, db, dout, size)
                latency = cupy_wrapper.time_kernel_run(
                    kernel, (grid, 1, 1), (threads, 1, 1), args, warmups=2, runs=6
                )
            elif device == "opencl":
                # build OpenCL kernel and time via profiling
                kernel = opencl_wrapper.OpenCLKernel(source, "elem_op")
                cl = opencl_wrapper.cl
                mf = cl.mem_flags
                da_buf = cl.Buffer(
                    kernel.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a
                )
                db_buf = cl.Buffer(
                    kernel.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b
                )
                dout_buf = cl.Buffer(kernel.ctx, mf.WRITE_ONLY, a.nbytes)
                global_size = (int(grid * threads),)
                local_size = (int(threads),)
                args = (da_buf, db_buf, dout_buf, np.uint64(size))
                latency = opencl_wrapper.time_kernel_run(
                    kernel, global_size, local_size, args, warmups=2, runs=6
                )
            elif device == "hip":
                kernel = hip_wrapper.HipKernel(source, "elem_op")
                args = (da, db, dout, size)
                latency = hip_wrapper.time_kernel_run(
                    kernel, (grid, 1, 1), (threads, 1, 1), args, warmups=2, runs=6
                )
            elif device == "metal":
                # just attempt compilation to ensure template is valid; runtime not measured
                kernel = metal_wrapper.MetalKernel(source, "elem_op")
                latency = float(size) * 1e-9  # placeholder
            else:
                continue
        except Exception:
            continue

        if latency < best["latency_s"]:
            best = {
                "latency_s": latency,
                "block": threads,
                "grid": grid,
                "dtype": dtype,
                "kernel_source_context": context,
                "backend": device,
            }

    if best["latency_s"] == float("inf"):
        raise RuntimeError("No candidate kernel successfully compiled and timed.")

    _ensure_cache_dir()
    _save_cache(device, op_name, best)
    return best


def get_cached_or_tune(
    op_name: str, size: int, dtype: str = "float32", device: str = "cuda"
):
    return autotune_elementwise(op_name, size=size, dtype=dtype, device=device)
