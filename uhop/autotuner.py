"""
Autotuner for elementwise ops (minimal PoC).

Features:
- Compile candidate kernels (from kernel template files)
- Benchmark each candidate using cupy backend
- Store best (kernel_id + launch config) to cache (JSON)

This module is intentionally compact so it can be expanded to other ops.
"""
import json
import os
from pathlib import Path
import math
from typing import Dict, Any
from jinja2 import Template
try:
    import cupy as cp  # type: ignore
except Exception:
    cp = None

from . import ops_registry
from .backends import cupy_wrapper

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
    # Map numpy-style dtype strings to CUDA C types
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


def compile_kernel_from_template(template_path: Path, kernel_name: str, context: Dict[str, Any]):
    source = _render_kernel_template(template_path, context)
    return cupy_wrapper.CupyKernel(source, kernel_name)


def autotune_elementwise(op_name: str, size: int, dtype: str = "float32", device: str = "cuda"):
    """
    Autotune elementwise op `op_name` for a single flat size.
    Returns cached best config or runs tuning and caches the result.
    """
    if device != "cuda":
        raise NotImplementedError("This autotuner currently supports the 'cuda' device only.")

    cached = _load_cache(device, op_name)
    if cached:
        return cached

    templates_dir = KERNELS_DIR / "cuda"
    # For now we only have elementwise_add template; map op_name->filename
    mapping = {
        "add": "elementwise_add.cu.jinja",
        "mul": "elementwise_add.cu.jinja",  # same kernel with different op
    }
    filename = mapping.get(op_name)
    if filename is None:
        raise ValueError(f"No kernel template mapped for op {op_name}")

    template_path = templates_dir / filename
    if not template_path.exists():
        raise FileNotFoundError(f"Kernel template {template_path} not found.")

    best = {"latency_s": float("inf")}
    candidates = _default_block_candidates()

    # Prepare sample arrays
    import numpy as np
    a = np.random.rand(size).astype(dtype)
    b = np.random.rand(size).astype(dtype)
    # put arrays on device once per candidate to avoid transfer timing
    da = cp.asarray(a)
    db = cp.asarray(b)
    dout = cp.empty_like(da)

    c_dtype = _cuda_dtype(dtype)

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
            kernel = compile_kernel_from_template(template_path, "elem_op", context)
        except Exception:
            # compilation failed; skip candidate
            continue

        # args: pointers + size (use cupy arrays and Python int)
        args = (da, db, dout, size)
        try:
            latency = cupy_wrapper.time_kernel_run(kernel, (grid, 1, 1), (threads, 1, 1), args, warmups=2, runs=6)
        except Exception:
            continue

        if latency < best["latency_s"]:
            best = {
                "latency_s": latency,
                "block": threads,
                "grid": grid,
                "dtype": dtype,
                "kernel_source_context": context,
            }

    if best["latency_s"] == float("inf"):
        raise RuntimeError("No candidate kernel successfully compiled and timed.")

    _ensure_cache_dir()
    _save_cache(device, op_name, best)
    return best


def get_cached_or_tune(op_name: str, size: int, dtype: str = "float32", device: str = "cuda"):
    return autotune_elementwise(op_name, size=size, dtype=dtype, device=device)
