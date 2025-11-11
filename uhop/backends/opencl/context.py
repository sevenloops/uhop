"""OpenCL context & queue management (refactored from monolith).

Responsibilities:
  * Safe creation of a process-wide (ctx, queue) pair.
  * Device selection via config (UHOP_OPENCL_DEVICE_INDEX).
  * Lightweight helpers for querying driver/device metadata.

NOTE: Original logic lifted from `opencl_backend._build_ctx_queue` to
      support modular import (matmul/conv2d modules depend on this).
"""
from __future__ import annotations

from typing import Optional, Tuple, Dict, Any
import threading

try:
    import pyopencl as cl  # type: ignore
except Exception:  # pragma: no cover - platform w/o OpenCL
    cl = None  # type: ignore

from ... import config as _cfg

_CTX_LOCK = threading.Lock()
_CTX_CACHE: Optional[Tuple["cl.Context", "cl.CommandQueue"]] = None  # type: ignore


class OpenCLUnavailable(RuntimeError):
    pass


def is_opencl_available() -> bool:
    if cl is None:
        return False
    try:
        for p in cl.get_platforms():
            try:
                if p.get_devices():  # any device
                    return True
            except Exception:
                continue
        return False
    except Exception:
        return False


def get_ctx_queue() -> Tuple["cl.Context", "cl.CommandQueue"]:  # type: ignore
    """Return singleton (context, queue) creating if needed."""
    if cl is None:
        raise OpenCLUnavailable("pyopencl not available")
    global _CTX_CACHE
    if _CTX_CACHE is not None:
        return _CTX_CACHE
    with _CTX_LOCK:
        if _CTX_CACHE is not None:
            return _CTX_CACHE
        devices = []
        for plat in cl.get_platforms():
            try:
                devices.extend(plat.get_devices())
            except Exception:
                continue
        if not devices:
            raise OpenCLUnavailable("No OpenCL devices found")
        try:
            idx = int(_cfg.get("UHOP_OPENCL_DEVICE_INDEX") or 0)
        except Exception:
            idx = 0
        idx = max(0, min(idx, len(devices) - 1))
        dev = devices[idx]
        ctx = cl.Context(devices=[dev])
        q = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
        _CTX_CACHE = (ctx, q)
        return _CTX_CACHE


def get_device_metadata() -> Dict[str, Any]:  # pragma: no cover (simple accessor)
    if not is_opencl_available():
        return {}
    try:
        ctx, _ = get_ctx_queue()
        dev = ctx.devices[0]
        return {
            "name": getattr(dev, "name", None),
            "vendor": getattr(dev, "vendor", None),
            "version": getattr(dev, "version", None),
            "max_work_group_size": getattr(dev, "max_work_group_size", None),
            "local_mem_size": getattr(dev, "local_mem_size", None),
            "global_mem_size": getattr(dev, "global_mem_size", None),
            "max_compute_units": getattr(dev, "max_compute_units", None),
        }
    except Exception:
        return {}


__all__ = [
    "is_opencl_available",
    "get_ctx_queue",
    "get_device_metadata",
    "OpenCLUnavailable",
]
