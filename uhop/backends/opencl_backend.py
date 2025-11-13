"""
OpenCL backend
=================

Capabilities:
- Device selection (via UHOP_OPENCL_DEVICE_INDEX) and cached context/queue
- Program build-cache with optional persistent binary caching
- MatMul: correctness-first naive kernel by default; optional tiled kernel with validation; optional CLBlast when available
- Conv2D: fused conv2d+relu for N=1, tiled conv2d for general shapes, optional im2col+GEMM when CLBlast is enabled
"""

from __future__ import annotations

import threading
from typing import Optional, Sequence, Tuple

import numpy as _np

try:
    import pyopencl as cl  # type: ignore
except Exception:  # pragma: no cover - platform without OpenCL
    cl = None  # type: ignore

from ..utils.logging import get_logger as _get_logger

_log = _get_logger("uhop.opencl")


# ------------------------
# Caches and global state
# ------------------------
_program_cache = {}
_program_cache_lock = threading.Lock()
_buffer_cache = {}
_buffer_cache_lock = threading.Lock()
_ctx_cache_lock = threading.Lock()
_ctx_cache = None  # (ctx, queue)
_naive_kernel_cache = {}  # key: (vec,width) -> (program, kernel)


def _build_ctx_queue():
    """Create or reuse a process-wide OpenCL context and command queue.

    Respects UHOP_OPENCL_DEVICE_INDEX if provided (0-based across all devices).
    """
    if cl is None:
        raise RuntimeError(
            "pyopencl not available — install PyOpenCL and vendor drivers (e.g., NVIDIA/AMD/Intel OpenCL runtime)"
        )
    global _ctx_cache
    with _ctx_cache_lock:
        if _ctx_cache is not None:
            return _ctx_cache
        # Flatten devices across all platforms
        devices = []
        for plat in cl.get_platforms():
            try:
                devices.extend(plat.get_devices())
            except Exception:
                continue
        if not devices:
            raise RuntimeError(
                "No OpenCL devices found — ensure GPU drivers and OpenCL runtime are installed and visible to your user"
            )
        # Device selection via centralized config
        from .. import config as _cfg

        try:
            idx = int(_cfg.get("UHOP_OPENCL_DEVICE_INDEX") or 0)
        except Exception:
            idx = 0
        idx = max(0, min(idx, len(devices) - 1))
        dev = devices[idx]
        ctx = cl.Context(devices=[dev])
        q = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
        _ctx_cache = (ctx, q)
        return _ctx_cache


def is_opencl_available() -> bool:
    """Return True if PyOpenCL is importable and at least one device is available."""
    if cl is None:
        return False
    try:
        plats = cl.get_platforms()
        for p in plats:
            try:
                if p.get_devices():
                    return True
            except Exception:
                continue
        return False
    except Exception:
        return False


def _load_conv2d_relu_kernel_source() -> str:
    from pathlib import Path as _Path

    src_path = _Path(__file__).resolve().parents[1] / "kernels" / "opencl" / "conv2d_relu.cl"
    return src_path.read_text()


def _load_conv2d_tiled_source() -> str:
    from pathlib import Path as _Path

    src_path = _Path(__file__).resolve().parents[1] / "kernels" / "opencl" / "conv2d_tiled.cl"
    return src_path.read_text()


def _load_im2col_source() -> str:
    from pathlib import Path as _Path

    src_path = _Path(__file__).resolve().parents[1] / "kernels" / "opencl" / "im2col.cl"
    return src_path.read_text()


def _load_matmul_tiled_source() -> str:
    from pathlib import Path as _Path

    src_path = _Path(__file__).resolve().parents[1] / "kernels" / "opencl" / "matmul_tiled.cl"
    return src_path.read_text()


def _get_program(ctx, source: str, program_key: str, build_options: Optional[str] = None):
    """Compile/get a cached program for the context/source key."""
    if cl is None:
        raise RuntimeError("pyopencl not available")
    key = (id(ctx), program_key, build_options or "")
    with _program_cache_lock:
        if key in _program_cache:
            return _program_cache[key]
        # Try persistent binary cache via KernelRegistry; fallback to normal compile
        try:
            import hashlib

            from ..cache import KernelRegistry as _KernelRegistry

            dev = ctx.devices[0]
            dev_name = getattr(dev, "name", "unknown")
            h = hashlib.sha1((source + "\n" + (build_options or "")).encode("utf-8")).hexdigest()
            reg = _KernelRegistry()
            bin_path = reg.load_opencl_binary(dev_name, h)
            if bin_path:
                with open(bin_path, "rb") as f:
                    binaries = [f.read()]
                prg = cl.Program(ctx, [dev], [binaries[0]]).build(options=build_options)
            else:
                prg = cl.Program(ctx, source).build(options=build_options)
                try:
                    bins = prg.get_info(cl.program_info.BINARIES)
                    if bins and isinstance(bins, (list, tuple)) and len(bins) > 0:
                        reg.save_opencl_binary(dev_name, h, bins)  # type: ignore
                except Exception:
                    pass
        except Exception:
            prg = cl.Program(ctx, source).build(options=build_options)
        _program_cache[key] = prg
        return prg


def _get_registry_kernel(operator: str):
    """Fetch kernel file info and source for an operator from the KernelRegistry."""
    from ..kernels import BackendType as _BT
    from ..kernels import get_kernel_registry

    reg = get_kernel_registry()
    kinfo = reg.get_kernel_file(operator, _BT.OPENCL)
    if kinfo is None:
        raise ValueError(f"No OpenCL kernel registered for operator '{operator}'")
    src = reg.get_kernel_source(operator, _BT.OPENCL)
    if not src:
        raise RuntimeError(f"Failed to load OpenCL kernel source for '{operator}'")
    return kinfo, src


def _build_program_from_registry(ctx, operator: str):
    kinfo, src = _get_registry_kernel(operator)
    key = f"{operator}::" + "+".join(kinfo.kernel_names)
    return _get_program(ctx, src, key), kinfo


def _enqueue_registry_kernel(
    q,
    prg,
    kernel_name: str,
    gsz: Tuple[int, ...],
    lsz: Optional[Tuple[int, ...]],
    args: Sequence,
):
    kn = getattr(prg, kernel_name)
    kn.set_args(*args)
    evt = cl.enqueue_nd_range_kernel(q, kn, gsz, lsz)
    return evt


def opencl_matmul(a, b, *, ir=None):
    """Delegated to refactored MatmulOp module."""
    if cl is None:
        raise RuntimeError("pyopencl not available")
    from .opencl.matmul import MatmulOp

    res = MatmulOp().execute(a, b, ir=ir)
    return res.output


def opencl_relu(x):
    if cl is None:
        raise RuntimeError("pyopencl not available")
    ctx, q = _build_ctx_queue()
    x = _np.array(x, dtype=_np.float32).ravel()
    out = _np.empty_like(x)
    mf = cl.mem_flags
    from ..cache import OPENCL_BUFFER_POOL as _POOL

    a_buf = _POOL.get(ctx, x.nbytes, mf.READ_ONLY)
    out_buf = _POOL.get(ctx, out.nbytes, mf.WRITE_ONLY)
    cl.enqueue_copy(q, a_buf, x)
    prg_src = r"""
    __kernel void relu(__global const float* A,
                       __global float* Out,
                       const int N) {
        int i = get_global_id(0);
        if (i < N) {
            float v = A[i];
            Out[i] = v > 0.0f ? v : 0.0f;
        }
    }
    """
    prg = _get_program(ctx, prg_src, "relu")
    evt = prg.relu(q, (x.size,), None, a_buf, out_buf, _np.int32(x.size))
    cl.enqueue_copy(q, out, out_buf, wait_for=[evt])
    q.finish()
    return out.reshape(-1)


def _choose_conv_impl(
    ctx,
    N: int,
    C: int,
    H: int,
    W: int,
    Cout: int,
    KH: int,
    KW: int,
    stride: int,
    padding: int,
):
    # Simple heuristic; keep behavior compatible with earlier logic
    ksz = int(C * KH * KW)
    outH = int((H + 2 * padding - KH) // stride + 1)
    outW = int((W + 2 * padding - KW) // stride + 1)
    outHW = int(outH * outW)
    dev = ctx.devices[0] if ctx.devices else None
    vendor = (getattr(dev, "vendor", "") or "").lower()
    if KH == 1 and KW == 1:
        return "im2col"
    if ksz >= 256 and outHW >= 256:
        return "im2col"
    if N >= 4 or Cout >= 256:
        return "im2col"
    if ("nvidia" in vendor or "advanced micro devices" in vendor or "amd" in vendor) and (ksz >= 128 and outHW >= 128):
        return "im2col"
    return "tiled"


_conv2d_cl_source = r"""
__kernel void conv2d_device(
    const int N, const int C, const int H, const int W,
    const int Cout, const int Cin, const int KH, const int KW,
    __global const float* input, __global const float* weights,
    __global float* output
) {
    // global ids: x = out_x, y = out_y, z = (n * Cout + co)
    int out_x = get_global_id(0);
    int out_y = get_global_id(1);
    int z = get_global_id(2);
    int co = z % Cout;
    int n = z / Cout;

    int outH = H - KH + 1;
    int outW = W - KW + 1;
    if (out_x >= outW || out_y >= outH) return;

    float sum = 0.0f;
    // for each input channel
    for (int ci = 0; ci < Cin; ++ci) {
        for (int ky = 0; ky < KH; ++ky) {
            for (int kx = 0; kx < KW; ++kx) {
                int in_y = out_y + ky;
                int in_x = out_x + kx;
                int in_index = ((n * C + ci) * H + in_y) * W + in_x;
                int w_index = ((co * Cin + ci) * KH + ky) * KW + kx;
                sum += input[in_index] * weights[w_index];
            }
        }
    }
    int out_index = ((n * Cout + co) * outH + out_y) * outW + out_x;
    output[out_index] = sum;
}
"""


def opencl_conv2d(input_np, weight_np, stride=1, padding=0, fuse_relu: bool = False):
    """Delegated to refactored Conv2DOp module."""
    if cl is None:
        raise RuntimeError("pyopencl not available")
    from .opencl.conv2d import Conv2DOp

    res = Conv2DOp().execute(input_np, weight_np, stride=stride, padding=padding, fuse_relu=fuse_relu)
    return res.output


def opencl_conv2d_relu(input_np, weight_np, stride=1, padding=0):
    return opencl_conv2d(input_np, weight_np, stride=stride, padding=padding, fuse_relu=True)


def opencl_conv2d_backward_input(
    input_shape: Tuple[int, int, int, int],
    weight_np,
    grad_out_np,
    stride: int = 1,
    padding: int = 0,
):
    if cl is None:
        raise RuntimeError("pyopencl not available")
    import numpy as _np

    N, Cin, H, W = [int(x) for x in input_shape]
    weight_np = _np.array(weight_np, dtype=_np.float32, order="C")
    grad_out_np = _np.array(grad_out_np, dtype=_np.float32, order="C")
    Cout, Cin_w, KH, KW = weight_np.shape
    assert Cin_w == Cin
    outH = int((H + 2 * padding - KH) // stride + 1)
    outW = int((W + 2 * padding - KW) // stride + 1)
    assert grad_out_np.shape == (N, Cout, outH, outW)
    ctx, q = _build_ctx_queue()
    mf = cl.mem_flags
    from ..cache import OPENCL_BUFFER_POOL as _POOL

    grad_in = _np.zeros((N, Cin, H, W), dtype=_np.float32)
    go_buf = _POOL.get(ctx, grad_out_np.nbytes, mf.READ_ONLY)
    w_buf = _POOL.get(ctx, weight_np.nbytes, mf.READ_ONLY)
    gi_buf = _POOL.get(ctx, grad_in.nbytes, mf.WRITE_ONLY)
    cl.enqueue_copy(q, go_buf, grad_out_np)
    cl.enqueue_copy(q, w_buf, weight_np)
    prg, kinfo = _build_program_from_registry(ctx, "conv2d_input_grad")
    kernel_name = kinfo.kernel_names[0]
    total = int(N * Cin * H * W)
    gsz = (total,)
    lsz = None
    args = (
        go_buf,
        w_buf,
        gi_buf,
        _np.int32(N),
        _np.int32(Cin),
        _np.int32(H),
        _np.int32(W),
        _np.int32(Cout),
        _np.int32(KH),
        _np.int32(KW),
        _np.int32(outH),
        _np.int32(outW),
        _np.int32(stride),
        _np.int32(stride),
        _np.int32(padding),
        _np.int32(padding),
    )
    evt = _enqueue_registry_kernel(q, prg, kernel_name, gsz, lsz, args)
    cl.enqueue_copy(q, grad_in, gi_buf, wait_for=[evt])
    q.finish()
    return grad_in


def opencl_conv2d_backward_weight(
    input_np,
    grad_out_np,
    weight_shape: Tuple[int, int, int, int],
    stride: int = 1,
    padding: int = 0,
):
    if cl is None:
        raise RuntimeError("pyopencl not available")
    import numpy as _np

    input_np = _np.array(input_np, dtype=_np.float32, order="C")
    grad_out_np = _np.array(grad_out_np, dtype=_np.float32, order="C")
    N, Cin, H, W = input_np.shape
    Cout, Cin_w, KH, KW = [int(x) for x in weight_shape]
    assert Cin_w == Cin
    outH = int((H + 2 * padding - KH) // stride + 1)
    outW = int((W + 2 * padding - KW) // stride + 1)
    assert grad_out_np.shape == (N, Cout, outH, outW)
    ctx, q = _build_ctx_queue()
    mf = cl.mem_flags
    from ..cache import OPENCL_BUFFER_POOL as _POOL

    grad_w = _np.zeros((Cout, Cin, KH, KW), dtype=_np.float32)
    in_buf = _POOL.get(ctx, input_np.nbytes, mf.READ_ONLY)
    go_buf = _POOL.get(ctx, grad_out_np.nbytes, mf.READ_ONLY)
    gw_buf = _POOL.get(ctx, grad_w.nbytes, mf.WRITE_ONLY)
    cl.enqueue_copy(q, in_buf, input_np)
    cl.enqueue_copy(q, go_buf, grad_out_np)
    prg, kinfo = _build_program_from_registry(ctx, "conv2d_weight_grad")
    kernel_name = kinfo.kernel_names[0]
    total = int(Cout * Cin * KH * KW)
    gsz = (total,)
    lsz = None
    args = (
        in_buf,
        go_buf,
        gw_buf,
        _np.int32(N),
        _np.int32(Cin),
        _np.int32(H),
        _np.int32(W),
        _np.int32(Cout),
        _np.int32(KH),
        _np.int32(KW),
        _np.int32(outH),
        _np.int32(outW),
        _np.int32(stride),
        _np.int32(stride),
        _np.int32(padding),
        _np.int32(padding),
    )
    evt = _enqueue_registry_kernel(q, prg, kernel_name, gsz, lsz, args)
    cl.enqueue_copy(q, grad_w, gw_buf, wait_for=[evt])
    q.finish()
    return grad_w
