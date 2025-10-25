"""
Top-level UHOP package exports (lightweight).

To keep imports fast and avoid importing heavy optional backends (e.g., torch)
on package import, we lazily import public symbols on first access.
"""
from __future__ import annotations

import importlib
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

    # Cache on the package module to avoid repeated imports
    globals()["UHopOptimizer"] = UHopOptimizer
    globals()["optimize"] = optimize
    return globals()[name]
  if name == "detect_hardware":
    from .hardware import detect_hardware

    globals()["detect_hardware"] = detect_hardware
    return detect_hardware
  if name == "UhopCache":
    from .cache import UhopCache

    globals()["UhopCache"] = UhopCache
    return UhopCache
  if name == "ops_registry":
    _ops_registry = importlib.import_module(__name__ + ".ops_registry")
    globals()["ops_registry"] = _ops_registry
    return _ops_registry
  if name == "autotuner":
    _autotuner = importlib.import_module(__name__ + ".autotuner")
    globals()["autotuner"] = _autotuner
    return _autotuner
  raise AttributeError(f"module 'uhop' has no attribute {name!r}")
