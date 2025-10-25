# uhop/backends/opencl_backend.py
"""
OpenCL backend.

Features:
- Device selection (env override + process-wide index)
- Program cache with optional persistence via KernelRegistry
- Simple buffer pool to reduce cl.Buffer churn
- Fused conv2d and conv2d+relu reference kernels

This module is used by the optimizer and the CLI demos when OpenCL is present.
"""

from __future__ import annotations
import threading
from typing import Optional, Tuple, Sequence
import os
import warnings

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
        # Choose a GPU device deterministically; allow override via env or
        # setter
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
                props = cl.command_queue_properties.PROFILING_ENABLE
                q = cl.CommandQueue(ctx, properties=props)
                return ctx, q
            # default: first GPU
            for p in plats:
                gpus = [
                    d for d in p.get_devices()
                    if d.type & cl.device_type.GPU
                ]
                if gpus:
                    ctx = cl.Context(devices=[gpus[0]])
                    props = cl.command_queue_properties.PROFILING_ENABLE
                    q = cl.CommandQueue(ctx, properties=props)
                    return ctx, q
        except Exception:
            pass
        # fallback: any device (may be CPU)
        ctx = cl.create_some_context(interactive=False)
        props = cl.command_queue_properties.PROFILING_ENABLE
        q = cl.CommandQueue(ctx, properties=props)
        return ctx, q

    def _choose_conv_impl(ctx, N, C, H, W, Cout, KH, KW, stride, padding) -> str:
        """Heuristic to choose between tiled conv and im2col+GEMM.

        Returns one of: 'tiled', 'im2col'.
        """
        # If CLBlast not available, stick to tiled
        try:
            from .clblast_integration import load_clblast as _load_clblast
            lib = _load_clblast()
        except Exception:
            lib = None
        if lib is None:
            return "tiled"

        # If CLBlast proved unstable on this device, stick to tiled to avoid fallback overhead
        try:
            from ..cache import UhopAutotune as _UhopAutotune
            dev_name = (ctx.devices[0].name if ctx.devices else "unknown")
            auto = _UhopAutotune()
            params = auto.get_params("opencl", "clblast", "sgemm", str(dev_name), "device") or {}
            if bool(params.get("unstable")):
                return "tiled"
        except Exception:
            pass

        # Simple heuristics:
        # - 1x1 kernels map perfectly to GEMM
        # - Large ksz (C*KH*KW) and large output area favor im2col
        # - Small images and 3x3 stride1 may favor tiled
        ksz = int(C * KH * KW)
        outH = int((H + 2 * padding - KH) // stride + 1)
        outW = int((W + 2 * padding - KW) // stride + 1)
        outHW = int(outH * outW)
        # Device vendor hint
        dev = ctx.devices[0] if ctx.devices else None
        vendor = (getattr(dev, "vendor", "") or "").lower()
        name = (getattr(dev, "name", "") or "").lower()

        # 1x1 -> im2col+GEMM
        if KH == 1 and KW == 1:
            return "im2col"
        # Large channels/kernels or big output -> im2col
        if ksz >= 256 and outHW >= 256:
            return "im2col"
        # If batch is large or Cout is large, GEMM tends to do better
        if N >= 4 or Cout >= 256:
            return "im2col"
        # Vendor-specific bias: prefer im2col on AMD/NVIDIA for mid-large shapes
        if ("nvidia" in vendor or "advanced micro devices" in vendor or "amd" in vendor) and (ksz >= 128 and outHW >= 128):
            return "im2col"
        # Default: tiled
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

    # Load fused conv2d+relu kernel source from kernels directory
    def _load_conv2d_relu_kernel_source() -> str:
        from pathlib import Path as _Path
        src_path = (
            _Path(__file__).resolve().parents[1]
            / "kernels" / "opencl" / "conv2d_relu.cl"
        )
        return src_path.read_text()

    def _load_conv2d_tiled_source() -> str:
        from pathlib import Path as _Path
        src_path = (
            _Path(__file__).resolve().parents[1]
            / "kernels" / "opencl" / "conv2d_tiled.cl"
        )
        return src_path.read_text()

    def _load_im2col_source() -> str:
        from pathlib import Path as _Path
        src_path = (
            _Path(__file__).resolve().parents[1]
            / "kernels" / "opencl" / "im2col.cl"
        )
        return src_path.read_text()

    def _get_program(ctx, source: str, program_key: str, build_options: Optional[str] = None):
        """
        Compile/get a cached program for the context/source key.
        program_key should identify semantics (e.g., conv2d_device + device).
        """
        key = (id(ctx), program_key, build_options or "")
        with _program_cache_lock:
            if key in _program_cache:
                return _program_cache[key]
            # Try persistent binary cache
            try:
                import hashlib
                from ..cache import KernelRegistry as _KernelRegistry
                dev = ctx.devices[0]
                dev_name = getattr(dev, "name", "unknown")
                # Include build options in source hash to disambiguate variants
                h = hashlib.sha1((source + "\n" + (build_options or "")).encode("utf-8")).hexdigest()
                reg = _KernelRegistry()
                bin_path = reg.load_opencl_binary(dev_name, h)
                if bin_path:
                    with open(bin_path, "rb") as f:
                        binaries = [f.read()]
                    prg = cl.Program(ctx, [dev], [binaries[0]]).build(options=build_options)
                else:
                    prg = cl.Program(ctx, source).build(options=build_options)
                    # Save binaries if available
                    try:
                        bins = prg.get_info(cl.program_info.BINARIES)
                        if (
                            bins and isinstance(bins, (list, tuple))
                            and len(bins) > 0
                        ):
                            reg.save_opencl_binary(
                                dev_name, h, bins
                            )  # type: ignore
                    except Exception:
                        pass
            except Exception:
                prg = cl.Program(ctx, source).build(options=build_options)
            _program_cache[key] = prg
            return prg

    def _load_matmul_tiled_source() -> str:
        from pathlib import Path as _Path
        src_path = (
            _Path(__file__).resolve().parents[1]
            / "kernels" / "opencl" / "matmul_tiled.cl"
        )
        return src_path.read_text()

    # Generic registry-driven loader/dispatcher
    def _get_registry_kernel(operator: str):
        """Fetch kernel file info and source for an operator from the KernelRegistry."""
        from ..kernels import get_kernel_registry, BackendType as _BT
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

    def _enqueue_registry_kernel(q, prg, kernel_name: str, gsz: Tuple[int, ...], lsz: Optional[Tuple[int, ...]], args: Sequence):
        kn = getattr(prg, kernel_name)
        kn.set_args(*args)
        evt = cl.enqueue_nd_range_kernel(q, kn, gsz, lsz)
        return evt

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

        # Optional policy: request CLBlast GEMM if available
        impl = os.environ.get("UHOP_OPENCL_MATMUL_IMPL", "tiled").lower()
        if impl == "clblast":
            try:
                from .clblast_integration import load_clblast, sgemm as _clblast_sgemm
                lib = load_clblast()
                if lib is not None:
                    # Skip immediately if device flagged unstable for CLBlast
                    try:
                        from ..cache import UhopAutotune as _UhopAutotune
                        dev_name = (ctx.devices[0].name if ctx.devices else "unknown")
                        auto = _UhopAutotune()
                        params = auto.get_params("opencl", "clblast", "sgemm", str(dev_name), "device") or {}
                        if bool(params.get("unstable")):
                            warnings.warn("UHOP: CLBlast marked unstable on this device; using tiled GEMM.")
                            raise RuntimeError("CLBlast unstable")
                    except Exception:
                        pass
                    a = _np.array(a, dtype=_np.float32, order="C")
                    b = _np.array(b, dtype=_np.float32, order="C")
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
                    status = _clblast_sgemm(lib, int(q.int_ptr), int(a_buf.int_ptr), int(b_buf.int_ptr), int(c_buf.int_ptr), m, n, k)
                    if status != 0:
                        warnings.warn(f"UHOP: CLBlastSgemm failed with status {status}; falling back to tiled GEMM.")
                        try:
                            from ..cache import UhopAutotune as _UhopAutotune
                            dev_name = (ctx.devices[0].name if ctx.devices else "unknown")
                            _UhopAutotune().set_params("opencl", "clblast", "sgemm", str(dev_name), "device", {"unstable": True})
                        except Exception:
                            pass
                    else:
                        cl.enqueue_copy(q, c, c_buf)
                        q.finish()
                        return c
                else:
                    warnings.warn("UHOP: CLBlast library not found; falling back to tiled GEMM.")
            except Exception as e:
                warnings.warn(f"UHOP: CLBlast path error: {e}; falling back to tiled GEMM.")
                # Mark unstable to avoid repeated attempts
                try:
                    from ..cache import UhopAutotune as _UhopAutotune
                    dev_name = (ctx.devices[0].name if ctx.devices else "unknown")
                    _UhopAutotune().set_params("opencl", "clblast", "sgemm", str(dev_name), "device", {"unstable": True})
                except Exception:
                    pass

        # Tiled GEMM with autotuning
        a = _np.array(a, dtype=_np.float32, order="C")
        b = _np.array(b, dtype=_np.float32, order="C")
        m, k = a.shape
        k2, n = b.shape
        assert k == k2
        c = _np.zeros((m, n), dtype=_np.float32)
        mf = cl.mem_flags
        from ..cache import OPENCL_BUFFER_POOL as _POOL
        from ..cache import UhopAutotune as _UhopAutotune

        a_buf = _POOL.get(ctx, a.nbytes, mf.READ_ONLY)
        b_buf = _POOL.get(ctx, b.nbytes, mf.READ_ONLY)
        c_buf = _POOL.get(ctx, c.nbytes, mf.WRITE_ONLY)
        cl.enqueue_copy(q, a_buf, a)
        cl.enqueue_copy(q, b_buf, b)

        # Autotune tile/local size per device and shape
        dev_name = ctx.devices[0].name if ctx.devices else "unknown"
        shape_key = f"M{m}_K{k}_N{n}"
        auto = _UhopAutotune()
        lsz_t = auto.get_lsz("opencl", "matmul", "matmul_tiled", dev_name, shape_key)
        params = auto.get_params("opencl", "matmul", "matmul_tiled", dev_name, shape_key) or {}
        persisted_tile = int(params.get("tile", 0) or 0)
        persisted_vec = int(params.get("vec", 0) or 0)

        # vectorization candidates opt-in via env, default VEC=1 only
        vec_env = os.environ.get("UHOP_OPENCL_VEC_CANDIDATES", "1")
        try:
            vec_cands = [int(v.strip()) for v in vec_env.split(",") if v.strip()]
            vec_cands = [v for v in vec_cands if v in (1, 2, 4, 8)]
        except Exception:
            vec_cands = [1]
        if not vec_cands:
            vec_cands = [1]

        candidates = []
        if lsz_t and persisted_tile:
            # Use persisted candidate
            candidates = [(tuple(int(x) for x in lsz_t), persisted_tile, persisted_vec or 1)]  # type: ignore
        else:
            # Try a few safe square tiles; pair with vec candidates
            tiles = [8, 16, 32]
            for t in tiles:
                for v in vec_cands:
                    candidates.append(((t, t), t, v))

        best = None
        best_t = 1e30
        best_tile = None
        best_vec = 1

        for cand in candidates:
            if isinstance(cand, tuple) and len(cand) == 3 and isinstance(cand[0], tuple):
                tile = int(cand[1])
                vec = int(cand[2])
            else:
                # backward compatibility when only lsz present
                tile = int(cand[0])
                vec = 1
            try:
                src = _load_matmul_tiled_source()
                # Include TILE and VEC defines in the cache key and build options
                build_opts = f"-D TILE={tile} -D VEC={vec}"
                prg = _get_program(ctx, src, f"matmul_tiled_T{tile}_V{vec}", build_options=build_opts)
                kn = cl.Kernel(prg, "matmul_tiled")
                # Global size padded to multiples of tile
                gsz = ((n + tile - 1) // tile * tile, (m + tile - 1) // tile * tile)
                lsz = (tile, tile)
                kn.set_args(a_buf, b_buf, c_buf, _np.int32(m), _np.int32(k), _np.int32(n))
                evt = cl.enqueue_nd_range_kernel(q, kn, gsz, lsz)
                evt.wait()
                try:
                    dt = (evt.profile.end - evt.profile.start) * 1e-9
                except Exception:
                    dt = 0.0
                if dt == 0.0:
                    import time as _time
                    t0 = _time.perf_counter()
                    cl.enqueue_nd_range_kernel(q, kn, gsz, lsz)
                    q.finish()
                    dt = _time.perf_counter() - t0
                if dt < best_t:
                    best_t = dt
                    best = (gsz, lsz)
                    best_tile = tile
                    best_vec = vec
                # Early exit if using persisted
                if lsz_t and persisted_tile:
                    break
            except Exception:
                continue

        if best is None:
            # Fallback: naive kernel inline
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
            prg = _get_program(ctx, prg_src, "matmul_naive")
            evt = prg.matmul(
                q,
                (m, n),
                None,
                _np.int32(m),
                _np.int32(n),
                _np.int32(k),
                a_buf,
                b_buf,
                c_buf,
            )
            cl.enqueue_copy(q, c, c_buf, wait_for=[evt])
            q.finish()
            return c

        # Persist the chosen local size and compile-time tile/vec
        if best and best_tile:
            try:
                auto.set_lsz("opencl", "matmul", "matmul_tiled", dev_name, shape_key, [best[1][0], best[1][1]])
                auto.set_params("opencl", "matmul", "matmul_tiled", dev_name, shape_key, {"tile": int(best_tile), "vec": int(best_vec)})
            except Exception:
                pass

        # Run with best config and copy back
        gsz, lsz = best
        # Reuse program for best_tile
        src = _load_matmul_tiled_source()
        prg = _get_program(ctx, src, f"matmul_tiled_T{best_tile}_V{best_vec}", build_options=f"-D TILE={int(best_tile)} -D VEC={int(best_vec)}")
        kn = cl.Kernel(prg, "matmul_tiled")
        kn.set_args(a_buf, b_buf, c_buf, _np.int32(m), _np.int32(k), _np.int32(n))
        evt = cl.enqueue_nd_range_kernel(q, kn, gsz, lsz)
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
        evt = prg.relu(
            q, (x.size,), None, a_buf, out_buf, _np.int32(x.size)
        )
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
    # If stride/padding are default (1/0) and batch is 1, use device fused
    # kernel when fuse_relu
        if stride == 1 and padding == 0 and N == 1:
            outH = H - KH + 1
            outW = W - KW + 1
            out = _np.zeros((N, Cout, outH, outW), dtype=_np.float32)
            mf = cl.mem_flags
            if fuse_relu:
                # Kernel expects single-batch input [C_in,H_in,W_in] and
                # outputs [C_out,H_out,W_out]
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
                prg = _get_program(
                    ctx, _load_conv2d_relu_kernel_source(),
                    "conv2d_relu_fused_file",
                )
                kn = cl.Kernel(prg, "conv2d_relu")
                # global size: (W_out, H_out, C_out); autotune local size
                # if available
                gsz = (int(outW), int(outH), int(Cout))
                lsz = None
                try:
                    from ..cache import UhopAutotune as _UhopAutotune
                    # build a simple shape key
                    shape_key = f"N{N}_C{C}_H{H}_W{W}_Co{Cout}_KH{KH}_KW{KW}"
                    dev_name = (
                        ctx.devices[0].name if ctx.devices else "unknown"
                    )
                    lsz_t = _UhopAutotune().get_lsz(
                        "opencl",
                        "conv2d_relu",
                        "conv2d_relu",
                        dev_name,
                        shape_key,
                    )
                    if lsz_t and len(lsz_t) in (2, 3):
                        lsz = tuple(int(x) for x in lsz_t)
                except Exception:
                    lsz = None
                if lsz is None:
                    # Micro-autotune a few candidates
                    candidates = [(8, 8, 1), (16, 8, 1), (8, 16, 1)]
                    best = None
                    best_t = 1e9
                    for cand in candidates:
                        kn.set_args(
                            in_buf,
                            w_buf,
                            b_buf,
                            out_buf,
                            _np.int32(C),
                            _np.int32(H),
                            _np.int32(W),
                            _np.int32(Cout),
                            _np.int32(KH),
                            _np.int32(KW),
                            _np.int32(1),
                            _np.int32(0),
                            _np.int32(0),
                            _np.int32(outH),
                            _np.int32(outW),
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
                            t0 = _time.perf_counter()
                            cl.enqueue_nd_range_kernel(q, kn, gsz, cand)
                            q.finish()
                            dt = _time.perf_counter() - t0
                        if dt < best_t:
                            best_t = dt
                            best = cand
                    lsz = best or (8, 8, 1)
                # set args
                kn.set_args(
                    in_buf,
                    w_buf,
                    b_buf,
                    out_buf,
                    _np.int32(C),
                    _np.int32(H),
                    _np.int32(W),
                    _np.int32(Cout),
                    _np.int32(KH),
                    _np.int32(KW),
                    _np.int32(1),
                    _np.int32(0),
                    _np.int32(0),
                    _np.int32(outH),
                    _np.int32(outW),
                )
                evt = cl.enqueue_nd_range_kernel(q, kn, gsz, lsz)
                q.finish()
                cl.enqueue_copy(q, out0, out_buf, wait_for=[evt])
                q.finish()
                # on first run, persist lsz if not present
                try:
                    from ..cache import UhopAutotune as _UhopAutotune
                    shape_key = f"N{N}_C{C}_H{H}_W{W}_Co{Cout}_KH{KH}_KW{KW}"
                    dev_name = (
                        ctx.devices[0].name if ctx.devices else "unknown"
                    )
                    _UhopAutotune().set_lsz(
                        "opencl",
                        "conv2d_relu",
                        "conv2d_relu",
                        dev_name,
                        shape_key,
                        list(lsz),
                    )
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
                    _np.int32(Cout),
                    _np.int32(Cin),
                    _np.int32(KH),
                    _np.int32(KW),
                    in_buf, w_buf, out_buf,
                )
                cl.enqueue_copy(q, out, out_buf)
                q.finish()
                return out
        else:
            # General path: use tiled OpenCL Conv2D kernel with local memory
            # for any N/stride/pad
            # Optional policy: im2col+GEMM when CLBlast is available and requested
            impl = os.environ.get("UHOP_OPENCL_CONV_IMPL", "auto").lower()
            if impl == "auto":
                impl = _choose_conv_impl(ctx, int(N), int(C), int(H), int(W), int(Cout), int(KH), int(KW), int(stride), int(padding))
            if impl in ("im2col_gemm", "im2col"):
                try:
                    from .clblast_integration import load_clblast, sgemm as _clblast_sgemm
                    lib = load_clblast()
                except Exception:
                    lib = None
                if lib is not None:
                    # Skip immediately if device flagged unstable for CLBlast
                    try:
                        from ..cache import UhopAutotune as _UhopAutotune
                        dev_name = (ctx.devices[0].name if ctx.devices else "unknown")
                        auto = _UhopAutotune()
                        params = auto.get_params("opencl", "clblast", "sgemm", str(dev_name), "device") or {}
                        if bool(params.get("unstable")):
                            warnings.warn("UHOP: CLBlast marked unstable on this device; using tiled conv2d.")
                            lib = None
                    except Exception:
                        pass
                    # Shapes
                    outH = int((H + 2 * padding - KH) // stride + 1)
                    outW = int((W + 2 * padding - KW) // stride + 1)
                    ksz = int(C * KH * KW)
                    out = _np.zeros((N, Cout, outH, outW), dtype=_np.float32)
                    # Prepare buffers
                    mf = cl.mem_flags
                    from ..cache import OPENCL_BUFFER_POOL as _POOL
                    in_buf = _POOL.get(ctx, input_np.nbytes, mf.READ_ONLY)
                    w_buf = _POOL.get(ctx, weight_np.nbytes, mf.READ_ONLY)
                    cl.enqueue_copy(q, in_buf, input_np)
                    cl.enqueue_copy(q, w_buf, weight_np)

                    # Build im2col kernel
                    prg = _get_program(ctx, _load_im2col_source(), "im2col")
                    kn = cl.Kernel(prg, "im2col_batched")

                    # Flatten weights as (K, C*R*S) row-major for GEMM
                    # No extra copy needed: interpret as row-major
                    # Device: reuse w_buf directly with appropriate strides; offsets=0 in our CLBlast wrapper
                    # Per-batch processing (simpler offsets management)
                    for b_idx in range(int(N)):
                        # Allocate per-batch columns and output matrices
                        cols_size = int(ksz * outH * outW * 4)
                        out2d_size = int(Cout * outH * outW * 4)
                        cols_buf = _POOL.get(ctx, cols_size, mf.READ_WRITE)
                        out2d_buf = _POOL.get(ctx, out2d_size, mf.WRITE_ONLY)
                        # Launch im2col for batch b_idx
                        gsz = (int(outW), int(outH), int(ksz))
                        kn.set_args(
                            in_buf, cols_buf,
                            _np.int32(N), _np.int32(C), _np.int32(H), _np.int32(W),
                            _np.int32(KH), _np.int32(KW),
                            _np.int32(outH), _np.int32(outW),
                            _np.int32(stride), _np.int32(padding),
                            _np.int32(b_idx)
                        )
                        cl.enqueue_nd_range_kernel(q, kn, gsz, None)

                        # GEMM: (K x ksz) * (ksz x outHW) => (K x outHW)
                        try:
                            status = _clblast_sgemm(
                                lib,
                                int(q.int_ptr),
                                int(w_buf.int_ptr),
                                int(cols_buf.int_ptr),
                                int(out2d_buf.int_ptr),
                                int(Cout), int(outH * outW), int(ksz),
                                1.0, 0.0,
                                False, False,
                            )
                            if status != 0:
                                warnings.warn(f"UHOP: CLBlastSgemm (im2col) failed with status {status}; using tiled conv2d.")
                                try:
                                    from ..cache import UhopAutotune as _UhopAutotune
                                    dev_name = (ctx.devices[0].name if ctx.devices else "unknown")
                                    _UhopAutotune().set_params("opencl", "clblast", "sgemm", str(dev_name), "device", {"unstable": True})
                                except Exception:
                                    pass
                                lib = None
                                break
                        except Exception as e:
                            warnings.warn(f"UHOP: CLBlastSgemm (im2col) raised {e}; using tiled conv2d.")
                            try:
                                from ..cache import UhopAutotune as _UhopAutotune
                                dev_name = (ctx.devices[0].name if ctx.devices else "unknown")
                                _UhopAutotune().set_params("opencl", "clblast", "sgemm", str(dev_name), "device", {"unstable": True})
                            except Exception:
                                pass
                            lib = None
                            break
                        # Copy out2d into the correct slice of out_buf as (K, outH, outW)
                        # We'll copy via host staging into out array for simplicity
                        tmp = _np.empty((Cout, outH, outW), dtype=_np.float32)
                        cl.enqueue_copy(q, tmp.reshape(Cout, outH * outW), out2d_buf)
                        out[b_idx] = tmp
                    q.finish()
                    if lib is not None:
                        if fuse_relu:
                            out = opencl_relu(out.reshape(-1)).reshape(N, Cout, outH, outW)
                        return out
                else:
                    warnings.warn("UHOP: CLBlast not available for im2col+GEMM; using tiled conv2d.")
            outH = (H + 2 * padding - KH) // stride + 1
            outW = (W + 2 * padding - KW) // stride + 1
            out = _np.zeros((N, Cout, outH, outW), dtype=_np.float32)
            mf = cl.mem_flags
            from ..cache import OPENCL_BUFFER_POOL as _POOL
            from ..cache import UhopAutotune as _UhopAutotune
            in_buf = _POOL.get(ctx, input_np.nbytes, mf.READ_ONLY)
            w_buf = _POOL.get(ctx, weight_np.nbytes, mf.READ_ONLY)
            out_buf = _POOL.get(ctx, out.nbytes, mf.WRITE_ONLY)
            cl.enqueue_copy(q, in_buf, input_np)
            cl.enqueue_copy(q, w_buf, weight_np)
            src_conv = _load_conv2d_tiled_source()
            # We'll build with -D TILE_W/TILE_H according to candidate lsz below
            # Initial placeholder build for kernel object creation; will re-build inside loop with options
            prg = None
            kn = None
            # Autotune local size (tile size) per device+shape, persist best
            dev_name = (ctx.devices[0].name if ctx.devices else "unknown")
            shape_key = f"N{N}_C{C}_H{H}_W{W}_Co{Cout}_KH{KH}_KW{KW}_S{stride}_P{padding}"
            auto = _UhopAutotune()
            lsz_t = auto.get_lsz("opencl", "conv2d", "conv2d_tiled", dev_name, shape_key)

            # Candidate local sizes (tile sizes). Must match kernel's tile expectations.
            candidates = []
            # vectorization candidates opt-in via env, default VEC=1 only
            vec_env = os.environ.get("UHOP_OPENCL_VEC_CANDIDATES", "1")
            try:
                vec_cands = [int(v.strip()) for v in vec_env.split(",") if v.strip()]
                vec_cands = [v for v in vec_cands if v in (1, 2, 4, 8)]
            except Exception:
                vec_cands = [1]
            if not vec_cands:
                vec_cands = [1]
            if lsz_t and len(lsz_t) in (2, 3):
                # persisted
                if len(lsz_t) == 2:
                    candidates = [(int(lsz_t[0]), int(lsz_t[1]), 1)]
                else:
                    candidates = [tuple(int(x) for x in lsz_t)]  # type: ignore
            else:
                # expand a bit and pair with vec candidates
                base_tiles = [(8, 8, 1), (16, 8, 1), (8, 16, 1), (16, 16, 1)]
                # Attach vec as a separate dimension in build options (kernel currently ignores it)
                candidates = [(tx, ty, tz, v) for (tx, ty, tz) in base_tiles for v in vec_cands]

            best = None
            best_t = 1e30
            # Try candidates
            for cand in candidates:
                # support both (tx,ty,tz) and (tx,ty,tz,vec)
                if len(cand) == 4:
                    tx, ty, tz, vec = int(cand[0]), int(cand[1]), int(cand[2]), int(cand[3])
                else:
                    tx, ty, tz = int(cand[0]), int(cand[1]), int(cand[2])
                    vec = 1
                # Build program for these tile sizes via -D defines
                try:
                    build_opts = f"-D TILE_W={tx} -D TILE_H={ty} -D VEC={vec}"
                    prg = _get_program(ctx, src_conv, f"conv2d_tiled_T{tx}x{ty}_V{vec}", build_options=build_opts)
                    kn = cl.Kernel(prg, "conv2d_tiled")
                except Exception:
                    continue
                # Global size padded to multiples of tile sizes
                gsz_x = int(((outW + tx - 1) // tx) * tx)
                gsz_y = int(((outH + ty - 1) // ty) * ty)
                gsz = (gsz_x, gsz_y, int(N * Cout))
                # Dynamic local memory sizes for input tile and weight tile
                tile_in_w = (tx - 1) * stride + KW
                tile_in_h = (ty - 1) * stride + KH
                tile_in_bytes = int(tile_in_w * tile_in_h * 4)
                tile_w_bytes = int(KH * KW * 4)
                # Set args
                kn.set_args(
                    in_buf, w_buf, out_buf,
                    _np.int32(N), _np.int32(C), _np.int32(H), _np.int32(W),
                    _np.int32(Cout), _np.int32(KH), _np.int32(KW),
                    _np.int32(outH),
                    _np.int32(outW),
                    _np.int32(stride),
                    _np.int32(padding),
                    cl.LocalMemory(tile_in_bytes),
                    cl.LocalMemory(tile_w_bytes),
                )
                try:
                    evt = cl.enqueue_nd_range_kernel(q, kn, gsz, (tx, ty, tz))
                    evt.wait()
                    try:
                        dt = (evt.profile.end - evt.profile.start) * 1e-9
                    except Exception:
                        dt = 0.0
                    if dt == 0.0:
                        import time as _time
                        t0 = _time.perf_counter()
                        cl.enqueue_nd_range_kernel(q, kn, gsz, (tx, ty, tz))
                        q.finish()
                        dt = _time.perf_counter() - t0
                    if dt < best_t:
                        best_t = dt
                        best = (gsz, (tx, ty, tz, vec), tile_in_bytes, tile_w_bytes)
                    # if persisted, break early
                    if lsz_t:
                        break
                except Exception:
                    continue

            if best is None:
                # Fallback to a safe default
                tx, ty, tz = 8, 8, 1
                build_opts = f"-D TILE_W={tx} -D TILE_H={ty} -D VEC=1"
                prg = _get_program(ctx, src_conv, f"conv2d_tiled_T{tx}x{ty}_V1", build_options=build_opts)
                kn = cl.Kernel(prg, "conv2d_tiled")
                gsz_x = int(((outW + tx - 1) // tx) * tx)
                gsz_y = int(((outH + ty - 1) // ty) * ty)
                gsz = (gsz_x, gsz_y, int(N * Cout))
                tile_in_w = (tx - 1) * stride + KW
                tile_in_h = (ty - 1) * stride + KH
                tile_in_bytes = int(tile_in_w * tile_in_h * 4)
                tile_w_bytes = int(KH * KW * 4)
                best = (gsz, (tx, ty, tz, 1), tile_in_bytes, tile_w_bytes)

            # Persist tuned local size
            if not lsz_t and best:
                try:
                    _, lsz_best, _, _ = best
                    # store only (tx,ty,tz) in lsz; persist vec separately
                    tx, ty, tz, vec = lsz_best
                    auto.set_lsz("opencl", "conv2d", "conv2d_tiled", dev_name, shape_key, [int(tx), int(ty), int(tz)])
                    auto.set_params("opencl", "conv2d", "conv2d_tiled", dev_name, shape_key, {"vec": int(vec)})
                except Exception:
                    pass

            # Final run with best config and copy back
            gsz, lsz, tile_in_bytes, tile_w_bytes = best
            # Ensure program/kernel is built with matching tile defines
            tx, ty, tz, vec = lsz
            build_opts = f"-D TILE_W={int(tx)} -D TILE_H={int(ty)} -D VEC={int(vec)}"
            prg = _get_program(ctx, src_conv, f"conv2d_tiled_T{int(tx)}x{int(ty)}_V{int(vec)}", build_options=build_opts)
            kn = cl.Kernel(prg, "conv2d_tiled")
            # Reset args with final LM sizes to be safe
            kn.set_args(
                in_buf, w_buf, out_buf,
                _np.int32(N), _np.int32(C), _np.int32(H), _np.int32(W),
                _np.int32(Cout), _np.int32(KH), _np.int32(KW),
                _np.int32(outH),
                _np.int32(outW),
                _np.int32(stride),
                _np.int32(padding),
                cl.LocalMemory(int(tile_in_bytes)),
                cl.LocalMemory(int(tile_w_bytes)),
            )
            lsz3 = (int(tx), int(ty), int(tz))
            evt = cl.enqueue_nd_range_kernel(q, kn, gsz, lsz3)
            cl.enqueue_copy(q, out, out_buf, wait_for=[evt])
            q.finish()
            if fuse_relu:
                # Apply ReLU with OpenCL kernel
                out = opencl_relu(out.reshape(-1)).reshape(N, Cout, outH, outW)
            return out

    def opencl_conv2d_relu(input_np, weight_np, stride=1, padding=0):
        return opencl_conv2d(
            input_np, weight_np, stride=stride, padding=padding, fuse_relu=True
        )

    def opencl_conv2d_backward_input(input_shape: Tuple[int, int, int, int], weight_np, grad_out_np, stride: int = 1, padding: int = 0):
        """Compute input gradient for conv2d given weight and grad_out.

        Args:
            input_shape: (N, Cin, H, W)
            weight_np: (Cout, Cin, KH, KW)
            grad_out_np: (N, Cout, outH, outW)
        Returns:
            grad_in: numpy array of shape input_shape
        """
        import numpy as _np

        N, Cin, H, W = [int(x) for x in input_shape]
        weight_np = _np.array(weight_np, dtype=_np.float32, order="C")
        grad_out_np = _np.array(grad_out_np, dtype=_np.float32, order="C")
        Cout, Cin_w, KH, KW = weight_np.shape
        assert Cin_w == Cin
        outH = int((H + 2 * padding - KH) // stride + 1)
        outW = int((W + 2 * padding - KW) // stride + 1)
        assert grad_out_np.shape == (N, Cout, outH, outW), f"grad_out shape {grad_out_np.shape} mismatch expected {(N, Cout, outH, outW)}"

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
            _np.int32(N), _np.int32(Cin), _np.int32(H), _np.int32(W),
            _np.int32(Cout), _np.int32(KH), _np.int32(KW),
            _np.int32(outH), _np.int32(outW),
            _np.int32(stride), _np.int32(stride),
            _np.int32(padding), _np.int32(padding),
        )
        evt = _enqueue_registry_kernel(q, prg, kernel_name, gsz, lsz, args)
        cl.enqueue_copy(q, grad_in, gi_buf, wait_for=[evt])
        q.finish()
        return grad_in

    def opencl_conv2d_backward_weight(input_np, grad_out_np, weight_shape: Tuple[int, int, int, int], stride: int = 1, padding: int = 0):
        """Compute weight gradient for conv2d given input and grad_out.

        Args:
            input_np: (N, Cin, H, W)
            grad_out_np: (N, Cout, outH, outW)
            weight_shape: (Cout, Cin, KH, KW)
        Returns:
            grad_w: numpy array of shape weight_shape
        """
        import numpy as _np

        input_np = _np.array(input_np, dtype=_np.float32, order="C")
        grad_out_np = _np.array(grad_out_np, dtype=_np.float32, order="C")
        N, Cin, H, W = input_np.shape
        Cout, Cin_w, KH, KW = [int(x) for x in weight_shape]
        assert Cin_w == Cin
        outH = int((H + 2 * padding - KH) // stride + 1)
        outW = int((W + 2 * padding - KW) // stride + 1)
        assert grad_out_np.shape == (N, Cout, outH, outW), f"grad_out shape {grad_out_np.shape} mismatch expected {(N, Cout, outH, outW)}"

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
            _np.int32(N), _np.int32(Cin), _np.int32(H), _np.int32(W),
            _np.int32(Cout), _np.int32(KH), _np.int32(KW),
            _np.int32(outH), _np.int32(outW),
            _np.int32(stride), _np.int32(stride),
            _np.int32(padding), _np.int32(padding),
        )
        evt = _enqueue_registry_kernel(q, prg, kernel_name, gsz, lsz, args)
        cl.enqueue_copy(q, grad_w, gw_buf, wait_for=[evt])
        q.finish()
        return grad_w

else:

    def opencl_matmul(*args, **kwargs):
        raise RuntimeError("pyopencl not available")

    def opencl_relu(*args, **kwargs):
        raise RuntimeError("pyopencl not available")

    def opencl_conv2d(*args, **kwargs):
        raise RuntimeError("pyopencl not available")

    def opencl_conv2d_relu(*args, **kwargs):
        raise RuntimeError("pyopencl not available")
