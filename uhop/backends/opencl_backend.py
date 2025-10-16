# uhop/backends/opencl_backend.py
"""
OpenCL backend with persistent program + buffer caching and device conv2d/conv2d_relu kernels.
Requires: pyopencl
"""

from __future__ import annotations
import threading
from typing import Tuple, Optional
import os

try:
    import pyopencl as cl
    import numpy as np  # type: ignore

    _OPENCL_AVAILABLE = True
except Exception:
    cl = None
    np = None
    _OPENCL_AVAILABLE = False

# Global caches (process-wide)
_program_cache = {}
_buffer_cache = {}  # key -> cl.Buffer
_program_cache_lock = threading.Lock()
_buffer_cache_lock = threading.Lock()
_DEVICE_INDEX: Optional[int] = None  # global GPU index across platforms


def is_opencl_available() -> bool:
    return _OPENCL_AVAILABLE


if _OPENCL_AVAILABLE:

    def _flatten_gpu_devices(platforms):
        devs = []
        for p in platforms:
            for d in p.get_devices():
                if d.type & cl.device_type.GPU:
                    devs.append((p, d))
        return devs

    def set_opencl_device(index: int):
        """Select GPU device by global index across all OpenCL platforms."""
        global _DEVICE_INDEX, _program_cache, _buffer_cache
        _DEVICE_INDEX = int(index)
        # Optional: clear caches to avoid mismatched device programs/buffers
        with _program_cache_lock:
            _program_cache.clear()
        with _buffer_cache_lock:
            _buffer_cache.clear()

    def _build_ctx_queue():
        # Choose a GPU device deterministically; allow override via env or setter
        env_idx = os.environ.get("UHOP_OPENCL_DEVICE_INDEX")
        chosen = _DEVICE_INDEX
        if env_idx is not None:
            try:
                chosen = int(env_idx)
            except Exception:
                pass
        try:
            plats = cl.get_platforms()
            gpu_devs = _flatten_gpu_devices(plats)
            if chosen is not None and 0 <= chosen < len(gpu_devs):
                _, d = gpu_devs[chosen]
                ctx = cl.Context(devices=[d])
                q = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
                return ctx, q
            # default: first GPU
            for p in plats:
                gpus = [d for d in p.get_devices() if d.type & cl.device_type.GPU]
                if gpus:
                    ctx = cl.Context(devices=[gpus[0]])
                    q = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
                    return ctx, q
        except Exception:
            pass
        # fallback: any device (may be CPU)
        ctx = cl.create_some_context(interactive=False)
        q = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
        return ctx, q

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

    # Load fused conv2d+relu kernel source from kernels directory
    def _load_conv2d_relu_kernel_source() -> str:
        from pathlib import Path as _Path
        src_path = _Path(__file__).resolve().parents[1] / "kernels" / "opencl" / "conv2d_relu.cl"
        return src_path.read_text()

    def _load_conv2d_tiled_source() -> str:
        from pathlib import Path as _Path
        src_path = _Path(__file__).resolve().parents[1] / "kernels" / "opencl" / "conv2d_tiled.cl"
        return src_path.read_text()

    def _get_program(ctx, source: str, program_key: str):
        """
        Compile or retrieve a cached program for the given context and source key.
        program_key should identify semantics (e.g., "conv2d_device" + device name).
        """
        key = (id(ctx), program_key)
        with _program_cache_lock:
            if key in _program_cache:
                return _program_cache[key]
            # Try persistent binary cache
            try:
                import hashlib
                from ..cache import KernelRegistry as _KernelRegistry
                dev = ctx.devices[0]
                dev_name = getattr(dev, "name", "unknown")
                h = hashlib.sha1(source.encode("utf-8")).hexdigest()
                reg = _KernelRegistry()
                bin_path = reg.load_opencl_binary(dev_name, h)
                if bin_path:
                    with open(bin_path, "rb") as f:
                        binaries = [f.read()]
                    prg = cl.Program(ctx, [dev], [binaries[0]]).build()
                else:
                    prg = cl.Program(ctx, source).build()
                    # Save binaries if available
                    try:
                        bins = prg.get_info(cl.program_info.BINARIES)
                        if bins and isinstance(bins, (list, tuple)) and len(bins) > 0:
                            reg.save_opencl_binary(dev_name, h, bins)  # type: ignore
                    except Exception:
                        pass
            except Exception:
                prg = cl.Program(ctx, source).build()
            _program_cache[key] = prg
            return prg

    def _maybe_get_buffer(ctx, q, arr):
        """
        Reuse a buffer if an identical-size buffer exists (simple heuristics).
        """
        key = (id(ctx), arr.nbytes)
        with _buffer_cache_lock:
            # reuse existing buffer if present
            if key in _buffer_cache:
                buf = _buffer_cache[key]
                return buf
            mf = cl.mem_flags
            buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=arr)
            _buffer_cache[key] = buf
            return buf

    def opencl_matmul(a, b):
        import numpy as _np

        ctx, q = _build_ctx_queue()
        a = _np.array(a, dtype=_np.float32)
        b = _np.array(b, dtype=_np.float32)
        m, k = a.shape
        k2, n = b.shape
        assert k == k2
        c = _np.zeros((m, n), dtype=_np.float32)
        mf = cl.mem_flags
        from ..cache import OPENCL_BUFFER_POOL as _POOL
        a_buf = _POOL.get(ctx, a.nbytes, mf.READ_ONLY)
        b_buf = _POOL.get(ctx, b.nbytes, mf.READ_ONLY)
        c_buf = _POOL.get(ctx, c.nbytes, mf.WRITE_ONLY)
        cl.enqueue_copy(q, a_buf, a)
        cl.enqueue_copy(q, b_buf, b)
        prg_src = r"""
        __kernel void matmul(const int M, const int N, const int K,
                             __global const float* A,
                             __global const float* B,
                             __global float* C) {
            int row = get_global_id(0);
            int col = get_global_id(1);
            if (row < M && col < N) {
                float s = 0.0f;
                for (int k = 0; k < K; ++k) {
                    s += A[row*K + k] * B[k*N + col];
                }
                C[row*N + col] = s;
            }
        }
        """
        prg = _get_program(ctx, prg_src, "matmul")
        evt = prg.matmul(q, (m, n), None, _np.int32(m), _np.int32(n), _np.int32(k), a_buf, b_buf, c_buf)
        # Non-blocking: read back and finish
        cl.enqueue_copy(q, c, c_buf, wait_for=[evt])
        q.finish()
        return c

    def opencl_relu(x):
        import numpy as _np

        ctx, q = _build_ctx_queue()
        x = _np.array(x, dtype=_np.float32).ravel()
        out = _np.empty_like(x)
        mf = cl.mem_flags
        from ..cache import OPENCL_BUFFER_POOL as _POOL
        a_buf = _POOL.get(ctx, x.nbytes, mf.READ_ONLY)
        out_buf = _POOL.get(ctx, out.nbytes, mf.WRITE_ONLY)
        cl.enqueue_copy(q, a_buf, x)
        prg_src = r"""
        __kernel void relu(__global const float* A, __global float* Out, const int N) {
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

    def opencl_conv2d(
        input_np, weight_np, stride=1, padding=0, fuse_relu: bool = False
    ):
        """
        Device-side conv2d via OpenCL. Accepts:
          input_np: (N, C, H, W)
          weight_np: (Cout, Cin, KH, KW)
        Returns output on host as numpy array (N, Cout, outH, outW).
        """
        import numpy as _np

        ctx, q = _build_ctx_queue()
        # ensure float32 contiguous
        input_np = _np.array(input_np, dtype=_np.float32, order="C")
        weight_np = _np.array(weight_np, dtype=_np.float32, order="C")
        N, C, H, W = input_np.shape
        Cout, Cin, KH, KW = weight_np.shape
        # If stride/padding are default (1/0) and batch is 1, use device fused kernel when fuse_relu
        if stride == 1 and padding == 0 and N == 1:
            outH = H - KH + 1
            outW = W - KW + 1
            out = _np.zeros((N, Cout, outH, outW), dtype=_np.float32)
            mf = cl.mem_flags
            if fuse_relu:
                # Kernel expects single-batch input [C_in,H_in,W_in] and outputs [C_out,H_out,W_out]
                inp0 = input_np[0]
                from ..cache import OPENCL_BUFFER_POOL as _POOL
                in_buf = _POOL.get(ctx, inp0.nbytes, mf.READ_ONLY)
                w_buf = _POOL.get(ctx, weight_np.nbytes, mf.READ_ONLY)
                bias = _np.zeros((Cout,), dtype=_np.float32)
                b_buf = _POOL.get(ctx, bias.nbytes, mf.READ_ONLY)
                out0 = _np.zeros((Cout, outH, outW), dtype=_np.float32)
                out_buf = _POOL.get(ctx, out0.nbytes, mf.WRITE_ONLY)
                # Upload inputs asynchronously
                cl.enqueue_copy(q, in_buf, inp0)
                cl.enqueue_copy(q, w_buf, weight_np)
                cl.enqueue_copy(q, b_buf, bias)
                prg = _get_program(ctx, _load_conv2d_relu_kernel_source(), "conv2d_relu_fused_file")
                kn = cl.Kernel(prg, "conv2d_relu")
                # global size: (W_out, H_out, C_out); use autotuned local size if available
                gsz = (int(outW), int(outH), int(Cout))
                lsz = None
                try:
                    from ..cache import UhopAutotune as _UhopAutotune
                    from . import opencl_matmul as _
                    # build a simple shape key
                    shape_key = f"N{N}_C{C}_H{H}_W{W}_Co{Cout}_KH{KH}_KW{KW}"
                    dev_name = ctx.devices[0].name if ctx.devices else "unknown"
                    lsz_t = _UhopAutotune().get_lsz("opencl", "conv2d_relu", "conv2d_relu", dev_name, shape_key)
                    if lsz_t and len(lsz_t) in (2, 3):
                        lsz = tuple(int(x) for x in lsz_t)
                except Exception:
                    lsz = None
                if lsz is None:
                    # Micro-autotune a few candidates
                    candidates = [(8,8,1), (16,8,1), (8,16,1)]
                    best = None
                    best_t = 1e9
                    for cand in candidates:
                        kn.set_args(
                            in_buf, w_buf, b_buf, out_buf,
                            _np.int32(C), _np.int32(H), _np.int32(W),
                            _np.int32(Cout), _np.int32(KH), _np.int32(KW),
                            _np.int32(1), _np.int32(0), _np.int32(0),
                            _np.int32(outH), _np.int32(outW)
                        )
                        evt = cl.enqueue_nd_range_kernel(q, kn, gsz, cand)
                        evt.wait()
                        try:
                            start = evt.profile.start
                            end = evt.profile.end
                            dt = (end - start) * 1e-9
                        except Exception:
                            dt = 0.0
                        if dt == 0.0:
                            # fallback rough timing
                            import time as _time
                            t0 = _time.perf_counter(); cl.enqueue_nd_range_kernel(q, kn, gsz, cand); q.finish(); dt = _time.perf_counter() - t0
                        if dt < best_t:
                            best_t = dt
                            best = cand
                    lsz = best or (8,8,1)
                # set args
                kn.set_args(
                    in_buf, w_buf, b_buf, out_buf,
                    _np.int32(C), _np.int32(H), _np.int32(W),
                    _np.int32(Cout), _np.int32(KH), _np.int32(KW),
                    _np.int32(1), _np.int32(0), _np.int32(0),
                    _np.int32(outH), _np.int32(outW)
                )
                evt = cl.enqueue_nd_range_kernel(q, kn, gsz, lsz)
                q.finish()
                cl.enqueue_copy(q, out0, out_buf, wait_for=[evt])
                q.finish()
                # on first run, persist lsz if not present
                try:
                    from ..cache import UhopAutotune as _UhopAutotune
                    shape_key = f"N{N}_C{C}_H{H}_W{W}_Co{Cout}_KH{KH}_KW{KW}"
                    dev_name = ctx.devices[0].name if ctx.devices else "unknown"
                    _UhopAutotune().set_lsz("opencl", "conv2d_relu", "conv2d_relu", dev_name, shape_key, list(lsz))
                except Exception:
                    pass
                out[0] = out0
                return out
            else:
                # Non-fused: use device conv2d kernel defined inline
                from ..cache import OPENCL_BUFFER_POOL as _POOL
                in_buf = _POOL.get(ctx, input_np.nbytes, mf.READ_ONLY)
                w_buf = _POOL.get(ctx, weight_np.nbytes, mf.READ_ONLY)
                out_buf = _POOL.get(ctx, out.nbytes, mf.WRITE_ONLY)
                cl.enqueue_copy(q, in_buf, input_np)
                cl.enqueue_copy(q, w_buf, weight_np)
                prg = _get_program(ctx, _conv2d_cl_source, "conv2d")
                kernel_name = "conv2d_device"
                global_size = (outW, outH, N * Cout)
                getattr(prg, kernel_name)(
                    q, global_size, None,
                    _np.int32(N), _np.int32(C), _np.int32(H), _np.int32(W),
                    _np.int32(Cout), _np.int32(Cin), _np.int32(KH), _np.int32(KW),
                    in_buf, w_buf, out_buf,
                )
                cl.enqueue_copy(q, out, out_buf)
                q.finish()
                return out
        else:
            # General path: use tiled OpenCL Conv2D kernel with local memory for any N/stride/pad
            outH = (H + 2 * padding - KH) // stride + 1
            outW = (W + 2 * padding - KW) // stride + 1
            out = _np.zeros((N, Cout, outH, outW), dtype=_np.float32)
            mf = cl.mem_flags
            from ..cache import OPENCL_BUFFER_POOL as _POOL
            in_buf = _POOL.get(ctx, input_np.nbytes, mf.READ_ONLY)
            w_buf = _POOL.get(ctx, weight_np.nbytes, mf.READ_ONLY)
            out_buf = _POOL.get(ctx, out.nbytes, mf.WRITE_ONLY)
            cl.enqueue_copy(q, in_buf, input_np)
            cl.enqueue_copy(q, w_buf, weight_np)
            prg = _get_program(ctx, _load_conv2d_tiled_source(), "conv2d_tiled")
            kn = cl.Kernel(prg, "conv2d_tiled")
            # Global size padded to multiples of tile sizes for valid local size usage
            gsz_x = int(((outW + 8 - 1) // 8) * 8)
            gsz_y = int(((outH + 8 - 1) // 8) * 8)
            gsz = (gsz_x, gsz_y, int(N * Cout))
            # Local tile sizes (8,8,1) consistent with kernel defines
            lsz = (8, 8, 1)
            # Dynamic local memory sizes for input tile and weight tile (account for stride)
            tile_in_w = (8 - 1) * stride + KW
            tile_in_h = (8 - 1) * stride + KH
            tile_in_bytes = int(tile_in_w * tile_in_h * 4)
            tile_w_bytes = int(KH * KW * 4)
            # Set args, including local memory placeholders
            kn.set_args(
                in_buf, w_buf, out_buf,
                _np.int32(N), _np.int32(C), _np.int32(H), _np.int32(W),
                _np.int32(Cout), _np.int32(KH), _np.int32(KW),
                _np.int32(outH), _np.int32(outW), _np.int32(stride), _np.int32(padding),
                cl.LocalMemory(tile_in_bytes),
                cl.LocalMemory(tile_w_bytes),
            )
            evt = cl.enqueue_nd_range_kernel(q, kn, gsz, lsz)
            cl.enqueue_copy(q, out, out_buf, wait_for=[evt])
            q.finish()
            if fuse_relu:
                # Apply ReLU with OpenCL kernel
                out = opencl_relu(out.reshape(-1)).reshape(N, Cout, outH, outW)
            return out

    def opencl_conv2d_relu(input_np, weight_np, stride=1, padding=0):
        return opencl_conv2d(input_np, weight_np, stride=stride, padding=padding, fuse_relu=True)

else:

    def opencl_matmul(*args, **kwargs):
        raise RuntimeError("pyopencl not available")

    def opencl_relu(*args, **kwargs):
        raise RuntimeError("pyopencl not available")

    def opencl_conv2d(*args, **kwargs):
        raise RuntimeError("pyopencl not available")
    def opencl_conv2d_relu(*args, **kwargs):
        raise RuntimeError("pyopencl not available")
