"""Torch extension helpers for UHOP.

Provides thin wrappers that integrate UHOP optimizer selection while preserving
Torch tensor outputs when desired.
"""

from __future__ import annotations

import torch

from .optimizer import UHopOptimizer

_OPT = UHopOptimizer(keep_format=True)


@_OPT.optimize("matmul")
def optimized_matmul(a, b):
    return a @ b


class UHopMatmulModule(torch.nn.Module):
    def forward(self, a, b):  # type: ignore
        return optimized_matmul(a, b)


__all__ = ["optimized_matmul", "UHopMatmulModule"]
