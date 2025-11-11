"""Backend registry consolidation.

Provides a single helper `ensure_default_backends_registered()` which creates
thin adapter objects around existing functional backend wrappers (torch,
triton, opencl) and registers them with the global `BackendManager`.

This allows higher‑level components (policy layer, future IR scheduler) to
enumerate available backends and query capability summaries instead of relying
on a hard‑coded preference order.

Design notes:
 - We intentionally keep initialization very lightweight; heavy driver/library
   imports stay inside the underlying wrapper modules already guarded by
   availability checks.
 - Adapter classes expose `get_supported_ops()` and minimal capability info.
 - Manual kernels are registered to mark basic coverage; the optimizer still
   uses specialized fast paths directly for now (incremental migration).
 - Duplicate registration is idempotent (safe to call multiple times).
"""
from __future__ import annotations

from typing import List

from .base import Backend, get_backend_manager
from . import (
    is_torch_available,
    is_triton_available,
    is_opencl_available,
    torch_matmul,
    torch_conv2d,
    torch_relu,
    triton_matmul,
    triton_relu,
    opencl_matmul,
    opencl_conv2d,
    opencl_relu,
)


class TorchBackend(Backend):
    def __init__(self):
        super().__init__("torch")

    def initialize(self) -> bool:  # type: ignore[override]
        avail = bool(is_torch_available())
        self.capabilities.available = avail
        if avail:
            # Device count best‑effort
            try:
                import torch  # type: ignore
                if torch.cuda.is_available():
                    self.capabilities.device_count = torch.cuda.device_count()
                    self.capabilities.device_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
                else:
                    self.capabilities.device_count = 1
                    self.capabilities.device_names = ["cpu"]
            except Exception:
                self.capabilities.device_count = 1
                self.capabilities.device_names = ["cpu"]
        return avail

    def check_vendor_libs(self) -> dict:  # type: ignore[override]
        libs = {}
        try:
            import torch  # type: ignore
            libs["torch"] = True
            if getattr(torch, "cuda", None) and torch.cuda.is_available():
                libs["cublas"] = True
        except Exception:
            pass
        return libs

    def get_supported_ops(self) -> List[str]:  # type: ignore[override]
        return ["matmul", "conv2d", "relu"]


class TritonBackend(Backend):
    def __init__(self):
        super().__init__("triton")

    def initialize(self) -> bool:  # type: ignore[override]
        avail = bool(is_triton_available())
        self.capabilities.available = avail
        return avail

    def check_vendor_libs(self) -> dict:  # type: ignore[override]
        libs = {}
        if self.capabilities.available:
            libs["triton"] = True
        return libs

    def get_supported_ops(self) -> List[str]:  # type: ignore[override]
        # Current MVP triton coverage
        return ["matmul", "relu"]


class OpenCLBackend(Backend):
    def __init__(self):
        super().__init__("opencl")

    def initialize(self) -> bool:  # type: ignore[override]
        avail = bool(is_opencl_available())
        self.capabilities.available = avail
        if avail:
            try:
                import pyopencl as cl  # type: ignore
                plats = cl.get_platforms()
                devs = []
                for p in plats:
                    try:
                        devs.extend(p.get_devices())
                    except Exception:
                        continue
                self.capabilities.device_count = len(devs)
                self.capabilities.device_names = [getattr(d, "name", "device") for d in devs]
            except Exception:
                pass
        return avail

    def check_vendor_libs(self) -> dict:  # type: ignore[override]
        libs = {}
        if self.capabilities.available:
            libs["opencl"] = True
        return libs

    def get_supported_ops(self) -> List[str]:  # type: ignore[override]
        return ["matmul", "conv2d", "relu"]


def ensure_default_backends_registered() -> None:
    """Register default adapter backends if not already present.

    Safe to call multiple times. Only registers backends that are available
    (initialize returns True).
    """
    mgr = get_backend_manager()
    # Avoid double registration by checking internal dict
    existing = set(getattr(mgr, "_backends", {}).keys())
    adapters = [TorchBackend(), TritonBackend(), OpenCLBackend()]
    # Optionally register Vulkan PoC backend when enabled
    try:
        import os
        from .vulkan_backend import VulkanBackend  # type: ignore

        if str(os.environ.get("UHOP_ENABLE_VULKAN_POC", "0")).lower() in (
            "1",
            "true",
            "yes",
            "on",
        ):
            adapters.append(VulkanBackend())
    except Exception:
        pass
    for b in adapters:
        if b.name in existing:
            continue
        try:
            b.initialize()
        except Exception:
            # mark unavailable but still register for introspection
            b.capabilities.available = False
        mgr.register_backend(b)
        # Register manual kernels for supported ops
        try:
            if b.name == "torch" and b.capabilities.available:
                b.register_manual_kernel("matmul", torch_matmul)
                b.register_manual_kernel("conv2d", torch_conv2d)
                b.register_manual_kernel("relu", torch_relu)
            elif b.name == "triton" and b.capabilities.available:
                b.register_manual_kernel("matmul", triton_matmul)
                b.register_manual_kernel("relu", triton_relu)
            elif b.name == "opencl" and b.capabilities.available:
                b.register_manual_kernel("matmul", opencl_matmul)
                b.register_manual_kernel("conv2d", opencl_conv2d)
                b.register_manual_kernel("relu", opencl_relu)
        except Exception:
            continue

__all__ = ["ensure_default_backends_registered"]
