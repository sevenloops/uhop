"""
Vulkan Backend Stub (PoC)
-------------------------

This is a minimal, non-functional Vulkan backend placeholder used to
demonstrate multi-backend wiring and capability detection. It is disabled by
default and can be enabled for experimentation by setting the environment
variable UHOP_ENABLE_VULKAN_POC=1.

No real kernels are executed; matmul is a placeholder that raises at call time.
"""

from __future__ import annotations

import os
from typing import List

from .base import Backend


class VulkanBackend(Backend):
    def __init__(self):
        super().__init__("vulkan")

    def initialize(self) -> bool:  # type: ignore[override]
        # Guarded by env var to avoid CI instability and extra deps
        enabled = str(os.environ.get("UHOP_ENABLE_VULKAN_POC", "0")).lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self.capabilities.available = enabled
        if enabled:
            # Best-effort info; real device enumeration would require a Vulkan Python binding
            self.capabilities.device_count = 0
            self.capabilities.device_names = []
            self.capabilities.error_msg = "PoC stub"
        return enabled

    def check_vendor_libs(self) -> dict:  # type: ignore[override]
        return {"vulkan": self.capabilities.available}

    def get_supported_ops(self) -> List[str]:  # type: ignore[override]
        # Keep small for PoC
        return ["matmul"]
