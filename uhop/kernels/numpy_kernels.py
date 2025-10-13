# uhop/kernels/numpy_kernels.py
"""
Simple NumPy reference kernels used for correctness tests and CPU fallback.

Currently exposes:
- numpy_matmul(a, b): matrix multiplication using NumPy (@)
"""
from __future__ import annotations

import numpy as np


def numpy_matmul(a, b):
    """Return A @ B as a NumPy array.

    Accepts array-like inputs and ensures float32 where possible for consistency
    with the rest of the project (but does not enforce dtype strictly).
    """
    A = np.array(a)
    B = np.array(b)
    return A @ B


__all__ = ["numpy_matmul"]
