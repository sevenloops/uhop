"""Autotune helpers facade for OpenCL ops.

Wraps UhopAutotune with convenience accessors and an optional disable gate.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ... import config as _cfg
from ...cache import UhopAutotune


class OCLAutotune:
    def __init__(self) -> None:
        self._at = UhopAutotune()

    @property
    def disabled(self) -> bool:
        return bool(_cfg.get("UHOP_DISABLE_AUTOTUNE"))

    def get_lsz(self, backend: str, op: str, kernel: str, device: str, shape_key: str) -> Optional[Tuple[int, ...]]:
        if self.disabled:
            return None
        return self._at.get_lsz(backend, op, kernel, device, shape_key)

    def set_lsz(self, backend: str, op: str, kernel: str, device: str, shape_key: str, lsz: List[int]) -> None:
        if self.disabled:
            return
        self._at.set_lsz(backend, op, kernel, device, shape_key, lsz)

    def get_params(self, backend: str, op: str, kernel: str, device: str, shape_key: str) -> Optional[Dict]:
        if self.disabled:
            return None
        return self._at.get_params(backend, op, kernel, device, shape_key)

    def set_params(self, backend: str, op: str, kernel: str, device: str, shape_key: str, params: Dict) -> None:
        if self.disabled:
            return
        self._at.set_params(backend, op, kernel, device, shape_key, params)

    def record_profile(
        self, backend: str, op: str, kernel: str, device: str, shape_key: str, gflops: float, ms: float
    ) -> None:
        if self.disabled:
            return
        self._at.record_profile(backend, op, kernel, device, shape_key, gflops, ms)

    def needs_retune(self, backend: str, op: str, kernel: str, device: str, shape_key: str) -> bool:
        if self.disabled:
            return False
        try:
            return self._at.needs_retune(backend, op, kernel, device, shape_key)
        except Exception:
            return False


__all__ = ["OCLAutotune"]
