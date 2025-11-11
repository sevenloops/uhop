"""CLBlast integration wrapped for OpenCL matmul/conv2d usage."""
from __future__ import annotations


from .context import get_ctx_queue


def load_clblast_safe():
    try:
        from ..clblast_integration import load_clblast  # type: ignore

        return load_clblast()
    except Exception:
        return None


def current_device_name() -> str:
    try:
        ctx, _ = get_ctx_queue()
        return ctx.devices[0].name if ctx.devices else "unknown"
    except Exception:
        return "unknown"


__all__ = ["load_clblast_safe", "current_device_name"]
