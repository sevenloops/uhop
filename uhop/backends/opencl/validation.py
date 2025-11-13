"""Validation utilities for OpenCL ops.

Includes pre-validation kernels and output checks to guard against
mis-compiled kernels on some drivers.
"""

from __future__ import annotations

import numpy as _np

try:
    import pyopencl as cl  # type: ignore
except Exception:  # pragma: no cover
    cl = None  # type: ignore


def _compute_local_size(tile: int, device: "cl.Device") -> tuple[int, int]:
    """Return a safe local work-group size for a given tile, honoring device limits.

    Tries (tile, tile) but clamps second dimension so product <= max_work_group_size.
    Ensures both dimensions are at least 1.
    """
    try:
        max_wg = int(getattr(device, "max_work_group_size", 256))
    except Exception:
        max_wg = 256
    if tile * tile <= max_wg:
        return (tile, tile)
    other = max(1, min(tile, max_wg // max(1, tile)))
    # Fallback if still too large (e.g., tile > max_wg)
    while tile * other > max_wg and other > 1:
        other //= 2
    return (max(1, tile), max(1, other))


def prevalidate_matmul(ctx: "cl.Context", q: "cl.CommandQueue", kernel, tile: int, flip_gws: bool) -> float:
    """Run a small 64x64 matmul to catch indexing/build issues.

    Returns Linf error vs numpy.
    """
    testN = testM = testK = 64
    A = _np.random.rand(testN, testM).astype(_np.float32)
    B = _np.random.rand(testM, testK).astype(_np.float32)
    C = _np.zeros((testN, testK), dtype=_np.float32)
    mf = cl.mem_flags
    a_tb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
    b_tb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
    c_tb = cl.Buffer(ctx, mf.WRITE_ONLY, size=C.nbytes)
    if flip_gws:
        gsz_t = ((testN + tile - 1) // tile * tile, (testK + tile - 1) // tile * tile)
    else:
        gsz_t = ((testK + tile - 1) // tile * tile, (testN + tile - 1) // tile * tile)
    kernel.set_args(a_tb, b_tb, c_tb, _np.int32(testN), _np.int32(testM), _np.int32(testK))
    lsz = _compute_local_size(tile, q.device)
    evtv = cl.enqueue_nd_range_kernel(q, kernel, gsz_t, lsz)
    cl.enqueue_copy(q, C, c_tb, wait_for=[evtv])
    q.finish()
    err = float(_np.max(_np.abs(C - (A @ B))))
    return err


__all__ = ["prevalidate_matmul", "_compute_local_size"]
