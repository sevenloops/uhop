"""Backend facade with lazy imports.

Avoid importing optional heavy dependencies (torch, triton) at module import
time. Import only when attributes are actually used.
"""
from __future__ import annotations

from typing import Any

__all__ = [
    "is_torch_available",
    "torch_matmul",
    "torch_conv2d",
    "torch_relu",
    "is_triton_available",
    "triton_matmul",
    "triton_conv2d",
    "triton_relu",
    "is_opencl_available",
    "opencl_matmul",
    "opencl_conv2d",
    "opencl_conv2d_relu",
    "opencl_relu",
    "is_mps_available",
    "mps_matmul",
    "mps_conv2d",
    "mps_relu",
]


def __getattr__(name: str) -> Any:
    if name.startswith("torch_") or name == "is_torch_available":
        from .torch_backend import (
            is_torch_available,
            torch_matmul,
            torch_conv2d,
            torch_relu,
        )

        return {
            "is_torch_available": is_torch_available,
            "torch_matmul": torch_matmul,
            "torch_conv2d": torch_conv2d,
            "torch_relu": torch_relu,
        }[name]
    if name.startswith("triton_") or name == "is_triton_available":
        from .triton_backend import (
            is_triton_available,
            triton_matmul,
            triton_conv2d,
            triton_relu,
        )

        return {
            "is_triton_available": is_triton_available,
            "triton_matmul": triton_matmul,
            "triton_conv2d": triton_conv2d,
            "triton_relu": triton_relu,
        }[name]
    if name.startswith("opencl_") or name == "is_opencl_available":
        from .opencl_backend import (
            is_opencl_available,
            opencl_matmul,
            opencl_conv2d,
            opencl_conv2d_relu,
            opencl_relu,
        )

        return {
            "is_opencl_available": is_opencl_available,
            "opencl_matmul": opencl_matmul,
            "opencl_conv2d": opencl_conv2d,
            "opencl_conv2d_relu": opencl_conv2d_relu,
            "opencl_relu": opencl_relu,
        }[name]
    if name.startswith("mps_") or name == "is_mps_available":
        from .mps_backend import (
            is_mps_available,
            mps_matmul,
            mps_conv2d,
            mps_relu,
        )

        return {
            "is_mps_available": is_mps_available,
            "mps_matmul": mps_matmul,
            "mps_conv2d": mps_conv2d,
            "mps_relu": mps_relu,
        }[name]
    raise AttributeError(name)
