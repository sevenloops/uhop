"""
Top-level UHOP package exports.

Exports a friendly API:
- UHopOptimizer: runtime optimizer that chooses the best backend.
- optimize: decorator to optimize user functions per operation
  (e.g., "matmul").
- detect_hardware: quick hardware snapshot for CLI and runtime decisions.
- UhopCache: simple on-disk cache for backend choices and artifacts.
"""
from .optimizer import UHopOptimizer, optimize
from .hardware import detect_hardware
from .cache import UhopCache

__all__ = ["UHopOptimizer", "optimize", "detect_hardware", "UhopCache"]
