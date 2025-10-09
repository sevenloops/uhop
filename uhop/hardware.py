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
    Simple hardware detection:
     - prefer CUDA (via torch.cuda)
     - fallback to CPU
    """
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            return HardwareProfile(vendor="nvidia", kind="cuda", name=name, details={"torch": torch.__version__})
    except Exception:
        pass

    # Could extend: check ROCm, Metal, OpenCL via system tools (clinfo) etc.
    return HardwareProfile(vendor="generic", kind="cpu", name=platform.processor(), details={})
