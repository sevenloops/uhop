# uhop/hardware.py
import platform
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class HardwareProfile:
    vendor: str
    kind: str
    name: Optional[str] = None
    details: Dict[str, Any] = None

def detect_hardware() -> HardwareProfile:
    """
    Simple multi-backend hardware detection (best-effort):
    Order:
      1) Torch CUDA GPU (NVIDIA)
      2) OpenCL GPU (AMD/Intel/NVIDIA)
      3) Fallback: CPU
    """
    # 1) Torch CUDA (NVIDIA)
    try:
        import torch  # type: ignore
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            return HardwareProfile(
                vendor="nvidia",
                kind="cuda",
                name=name,
                details={"torch": getattr(torch, "__version__", "unknown")},
            )
    except Exception:
        pass

    # 2) OpenCL GPU (AMD/Intel/NVIDIA)
    try:
        import pyopencl as cl  # type: ignore
        plats = cl.get_platforms()
        # Prefer a GPU device
        for p in plats:
            gpus = [d for d in p.get_devices() if d.type & cl.device_type.GPU]
            if gpus:
                dev = gpus[0]
                vendor_raw = (getattr(dev, "vendor", None) or getattr(p, "vendor", "")).lower()
                if "amd" in vendor_raw:
                    vendor = "amd"
                elif "nvidia" in vendor_raw:
                    vendor = "nvidia"
                elif "intel" in vendor_raw:
                    vendor = "intel"
                else:
                    vendor = getattr(p, "vendor", "unknown")
                name = getattr(dev, "name", None) or getattr(p, "name", None) or "OpenCL GPU"
                return HardwareProfile(
                    vendor=vendor,
                    kind="opencl",
                    name=name,
                    details={
                        "platform": getattr(p, "name", None),
                        "platform_vendor": getattr(p, "vendor", None),
                        "device_vendor": getattr(dev, "vendor", None),
                        "device_version": getattr(dev, "version", None),
                    },
                )
        # If no GPU but OpenCL exists, report CPU via OpenCL
        for p in plats:
            cpus = [d for d in p.get_devices() if d.type & cl.device_type.CPU]
            if cpus:
                dev = cpus[0]
                vendor_raw = (getattr(dev, "vendor", None) or getattr(p, "vendor", "")).lower()
                if "amd" in vendor_raw:
                    vendor = "amd"
                elif "nvidia" in vendor_raw:
                    vendor = "nvidia"
                elif "intel" in vendor_raw:
                    vendor = "intel"
                else:
                    vendor = getattr(p, "vendor", "unknown")
                name = getattr(dev, "name", None) or getattr(p, "name", None) or "OpenCL CPU"
                return HardwareProfile(
                    vendor=vendor,
                    kind="opencl-cpu",
                    name=name,
                    details={
                        "platform": getattr(p, "name", None),
                        "platform_vendor": getattr(p, "vendor", None),
                        "device_vendor": getattr(dev, "vendor", None),
                        "device_version": getattr(dev, "version", None),
                    },
                )
    except Exception:
        pass

    # 3) CPU fallback
    return HardwareProfile(vendor="generic", kind="cpu", name=platform.processor(), details={})
