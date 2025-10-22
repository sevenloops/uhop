"""
Top-level UHOP package exports (lightweight).

To keep imports fast and avoid importing heavy optional backends (e.g., torch)
on package import, we lazily import public symbols on first access.
"""
from __future__ import annotations

from typing import Any

__all__ = [
  "UHopOptimizer",
  "optimize",
  "detect_hardware",
  "UhopCache",
  "ops_registry",
  "autotuner",
]


def __getattr__(name: str) -> Any:  # lazy attribute loader
  if name in ("UHopOptimizer", "optimize"):
    from .optimizer import UHopOptimizer, optimize

    return {"UHopOptimizer": UHopOptimizer, "optimize": optimize}[name]
  if name == "detect_hardware":
    from .hardware import detect_hardware

    return detect_hardware
  if name == "UhopCache":
    from .cache import UhopCache

    return UhopCache
  if name == "ops_registry":
    from . import ops_registry as _ops_registry  # type: ignore

    return _ops_registry
  if name == "autotuner":
    from . import autotuner as _autotuner  # type: ignore

    return _autotuner
  raise AttributeError(f"module 'uhop' has no attribute {name!r}")
