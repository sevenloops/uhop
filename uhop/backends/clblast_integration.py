# uhop/backends/clblast_integration.py
"""
CLBlast integration helpers using ctypes.

Provides a thin wrapper around CLBlastSgemm for single-precision GEMM that
accepts PyOpenCL queue and buffers. If the CLBlast shared library is not found,
`load_clblast()` returns None.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import os
import sys
from typing import Optional

# CLBlast enums (subset)
CLBLAST_LAYOUT_ROW_MAJOR = 0  # clblast::Layout::kRowMajor
CLBLAST_TRANSPOSE_NO = 0  # clblast::Transpose::kNo
CLBLAST_TRANSPOSE_YES = 1  # clblast::Transpose::kYes


def _find_clblast_library() -> Optional[str]:
    # Honor explicit path
    env = os.environ.get("CLBLAST_LIBRARY")
    if env and os.path.exists(env):
        return env
    # Try platform-specific names
    names = []
    if sys.platform.startswith("win"):
        names = ["clblast.dll", "CLBlast.dll"]
    elif sys.platform == "darwin":
        names = ["libclblast.dylib"]
    else:
        names = ["libclblast.so"]
    for n in names:
        path = ctypes.util.find_library(n) or n
        try:
            # Just attempt to load then unload to verify existence
            ctypes.CDLL(path)
            return path
        except Exception:
            continue
    # Final fallback: plain name might resolve via loader path
    try:
        ctypes.CDLL("clblast")
        return "clblast"
    except Exception:
        return None


def load_clblast() -> Optional[ctypes.CDLL]:
    path = _find_clblast_library()
    if not path:
        return None
    try:
        if sys.platform.startswith("win"):
            # On Windows some builds may use stdcall; WinDLL handles that
            lib = ctypes.WinDLL(path)  # type: ignore[attr-defined]
        else:
            lib = ctypes.CDLL(path)
        return lib
    except Exception:
        return None


def sgemm(
    lib: ctypes.CDLL,
    queue_int_ptr: int,
    a_buf_int_ptr: int,
    b_buf_int_ptr: int,
    c_buf_int_ptr: int,
    M: int,
    N: int,
    K: int,
    alpha: float = 1.0,
    beta: float = 0.0,
    a_transpose: bool = False,
    b_transpose: bool = False,
) -> int:
    """
    Run CLBlastSgemm in row-major layout: C[M,N] = alpha * A[M,K] @ B[K,N] + beta*C.

    Args:
        lib: Loaded CLBlast CDLL
        queue_int_ptr: PyOpenCL command queue integer pointer (queue.int_ptr)
        a_buf_int_ptr/b_buf_int_ptr/c_buf_int_ptr: PyOpenCL cl.Buffer int_ptr
        M, N, K: GEMM dimensions
        alpha, beta: scalars
        a_transpose, b_transpose: whether to transpose A or B
    Returns: CLBlastStatusCode (0 = success)
    """
    # Resolve symbol
    try:
        func = lib.CLBlastSgemm
    except AttributeError:
        # Some builds prefix with clblast_ or similar (unlikely). Fail gracefully.
        raise RuntimeError("CLBlastSgemm not found in CLBlast library")

    # Set argtypes, restype (exact C ABI)
    # CLBlastStatusCode CLBlastSgemm(Layout, TransA, TransB, size_t m, size_t n, size_t k,
    #   float alpha, cl_mem A, size_t a_offset, size_t a_ld,
    #   cl_mem B, size_t b_offset, size_t b_ld,
    #   float beta, cl_mem C, size_t c_offset, size_t c_ld,
    #   cl_command_queue queue, cl_event* event)
    c_size_t = ctypes.c_size_t
    c_float = ctypes.c_float
    c_void_p = ctypes.c_void_p
    c_int = ctypes.c_int

    func.restype = c_int
    func.argtypes = [
        c_int,
        c_int,
        c_int,  # layout, transA, transB
        c_size_t,
        c_size_t,
        c_size_t,  # m, n, k
        c_float,  # alpha
        c_void_p,
        c_size_t,
        c_size_t,  # A, a_offset, a_ld
        c_void_p,
        c_size_t,
        c_size_t,  # B, b_offset, b_ld
        c_float,  # beta
        c_void_p,
        c_size_t,
        c_size_t,  # C, c_offset, c_ld
        c_void_p,
        ctypes.POINTER(c_void_p),  # queue, event (nullable)
    ]

    layout = CLBLAST_LAYOUT_ROW_MAJOR
    transA = CLBLAST_TRANSPOSE_YES if a_transpose else CLBLAST_TRANSPOSE_NO
    transB = CLBLAST_TRANSPOSE_YES if b_transpose else CLBLAST_TRANSPOSE_NO

    # Leading dimensions for row-major non-transposed A[M,K], B[K,N], C[M,N]
    a_ld = K if not a_transpose else M
    b_ld = N if not b_transpose else K
    c_ld = N

    # Offsets 0 and output event pointer (unused, but non-NULL to avoid buggy drivers)
    out_event = c_void_p(0)

    status = func(
        c_int(layout),
        c_int(transA),
        c_int(transB),
        c_size_t(M),
        c_size_t(N),
        c_size_t(K),
        c_float(alpha),
        c_void_p(int(a_buf_int_ptr)),
        c_size_t(0),
        c_size_t(a_ld),
        c_void_p(int(b_buf_int_ptr)),
        c_size_t(0),
        c_size_t(b_ld),
        c_float(beta),
        c_void_p(int(c_buf_int_ptr)),
        c_size_t(0),
        c_size_t(c_ld),
        c_void_p(int(queue_int_ptr)),
        ctypes.byref(out_event),
    )
    return int(status)
