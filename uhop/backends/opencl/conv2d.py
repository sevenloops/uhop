"""Conv2D operation module (refactored).

Supports:
- Fused conv2d+relu for N=1
- Tiled conv2d
- im2col + CLBlast GEMM path
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as _np

try:
    import pyopencl as cl  # type: ignore
except Exception:  # pragma: no cover
    cl = None  # type: ignore

from ... import config as _cfg
from ...cache import OPENCL_BUFFER_POOL
from ...utils.logging import get_logger as _get_logger
from .autotune import OCLAutotune
from .clblast import current_device_name, load_clblast_safe
from .context import get_ctx_queue, is_opencl_available

_log = _get_logger("uhop.opencl.conv2d")


@dataclass
class Conv2DResult:
    output: _np.ndarray
    impl: str
    ms: float | None = None
    validated_err: float | None = None
    unstable: bool = False


class Conv2DOp:
    def __init__(self) -> None:
        self.auto = OCLAutotune()

    def execute(
        self, input_np: Any, weight_np: Any, stride: int = 1, padding: int = 0, fuse_relu: bool = False
    ) -> Conv2DResult:
        if not is_opencl_available():
            raise RuntimeError("OpenCL unavailable")
        ctx, q = get_ctx_queue()
        input_np = _np.array(input_np, dtype=_np.float32, order="C")
        weight_np = _np.array(weight_np, dtype=_np.float32, order="C")
        N, C, H, W = input_np.shape
        Cout, Cin, KH, KW = weight_np.shape
        outH = (H + 2 * padding - KH) // stride + 1
        outW = (W + 2 * padding - KW) // stride + 1
        dev_name = current_device_name()
        shape_key = f"N{N}_C{C}_H{H}_W{W}_Co{Cout}_KH{KH}_KW{KW}_S{stride}_P{padding}"

        # Fast fused N=1 path
        if stride == 1 and padding == 0 and N == 1 and fuse_relu:
            # Borrow helpers from legacy backend for kernel source and building
            from ..opencl_backend import _load_conv2d_relu_kernel_source

            src = _load_conv2d_relu_kernel_source()
            prg = cl.Program(ctx, src).build()
            mf = cl.mem_flags
            inp0 = input_np[0]
            out = _np.zeros((N, Cout, outH, outW), dtype=_np.float32)
            out0 = _np.zeros((Cout, outH, outW), dtype=_np.float32)
            in_buf = OPENCL_BUFFER_POOL.get(ctx, inp0.nbytes, mf.READ_ONLY)
            w_buf = OPENCL_BUFFER_POOL.get(ctx, weight_np.nbytes, mf.READ_ONLY)
            b = _np.zeros((Cout,), dtype=_np.float32)
            b_buf = OPENCL_BUFFER_POOL.get(ctx, b.nbytes, mf.READ_ONLY)
            y_buf = OPENCL_BUFFER_POOL.get(ctx, out0.nbytes, mf.WRITE_ONLY)
            cl.enqueue_copy(q, in_buf, inp0)
            cl.enqueue_copy(q, w_buf, weight_np)
            cl.enqueue_copy(q, b_buf, b)
            kn = cl.Kernel(prg, "conv2d_relu")
            gsz = (int(outW), int(outH), int(Cout))
            # Optional tuned local size
            lsz = None
            try:
                lsz_t = self.auto.get_lsz("opencl", "conv2d_relu", "conv2d_relu", dev_name, shape_key)
                if lsz_t and len(lsz_t) in (2, 3):
                    lsz = tuple(int(x) for x in lsz_t)
            except Exception:
                lsz = None
            kn.set_args(
                in_buf,
                w_buf,
                b_buf,
                y_buf,
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
            cl.enqueue_copy(q, out0, y_buf, wait_for=[evt])
            out[0] = out0
            # Record profiling if available
            try:
                dt_ms = (evt.profile.end - evt.profile.start) * 1e-6 if hasattr(evt, "profile") else 0.0
                gflops = None
                if dt_ms > 0:
                    # Approx FLOPs: N*outH*outW*Cout*Cin*KH*KW*2
                    flops = N * outH * outW * Cout * Cin * KH * KW * 2.0
                    gflops = flops / (dt_ms * 1e6)
                self.auto.record_profile("opencl", "conv2d", "conv2d_fused", dev_name, shape_key, gflops or 0.0, dt_ms)
                self.auto.set_params(
                    "opencl",
                    "conv2d",
                    "conv2d_fused",
                    dev_name,
                    shape_key,
                    {"tuned_at": float(evt.profile.end) if hasattr(evt, "profile") else None},
                )
            except Exception:
                pass
            return Conv2DResult(out, "fused", ms=dt_ms if "dt_ms" in locals() else None)

        # Decide implementation
        impl = str((_cfg.get("UHOP_OPENCL_CONV_IMPL") or "auto")).lower()
        if impl == "auto":
            from ..opencl_backend import _choose_conv_impl as _choose

            impl = _choose(ctx, int(N), int(C), int(H), int(W), int(Cout), int(KH), int(KW), int(stride), int(padding))
        # Guard: avoid im2col path for very large aggregate workloads that have been observed to trigger driver instability
        workload = N * Cout * outH * outW * Cin * KH * KW
        try:
            wl_thresh = int(_cfg.get("UHOP_OPENCL_CONV_LARGE_WORKLOAD") or 8_000_000)
        except Exception:
            wl_thresh = 8_000_000
        if impl in ("im2col_gemm", "im2col") and workload > wl_thresh:
            _log.warning(f"[conv2d] workload {workload} exceeds threshold {wl_thresh}; forcing tiled path")
            impl = "tiled"
        # Guard: im2col temp buffer sizes vs device global memory (very conservative)
        try:
            dev_global = int(ctx.devices[0].global_mem_size)
        except Exception:
            dev_global = 0
        if impl in ("im2col_gemm", "im2col") and dev_global > 0:
            cols_bytes = int(C * KH * KW * outH * outW * 4)
            out2d_bytes = int(Cout * outH * outW * 4)
            # If temp buffers would exceed 40% of global memory, avoid im2col
            if (cols_bytes + out2d_bytes) > int(dev_global * 0.4):
                _log.warning(
                    f"[conv2d] temp buffers {cols_bytes+out2d_bytes} exceed 40% of global mem {dev_global}; forcing tiled"
                )
                impl = "tiled"

        # im2col + CLBlast
        if impl in ("im2col_gemm", "im2col"):
            lib = load_clblast_safe()
            if lib is not None:
                # Skip if unstable for this device
                try:
                    params = self.auto.get_params("opencl", "clblast", "sgemm", dev_name, "device") or {}
                    if bool(params.get("unstable")):
                        lib = None
                except Exception:
                    pass
            if lib is not None:
                mf = cl.mem_flags
                in_buf = OPENCL_BUFFER_POOL.get(ctx, input_np.nbytes, mf.READ_ONLY)
                w_buf = OPENCL_BUFFER_POOL.get(ctx, weight_np.nbytes, mf.READ_ONLY)
                cl.enqueue_copy(q, in_buf, input_np)
                cl.enqueue_copy(q, w_buf, weight_np)
                from ..opencl_backend import _load_im2col_source

                src = _load_im2col_source()
                prg = cl.Program(ctx, src).build()
                kn = cl.Kernel(prg, "im2col_batched")
                out = _np.zeros((N, Cout, outH, outW), dtype=_np.float32)
                from ..clblast_integration import (
                    sgemm as _clblast_sgemm,  # type: ignore
                )

                ksz = int(C * KH * KW)
                # Choose a safe local size for im2col kernel by quick probe on first batch
                best_lsz = None
                try:
                    lsz_cands = [(8, 8, 1), (16, 4, 1), (4, 16, 1), None]
                    # Reduce candidate set for extreme stride/padding
                    if stride > 2 or padding > max(KH, KW) // 2:
                        lsz_cands = [(8, 8, 1), None]
                    cols_size_probe = int(ksz * outH * outW * 4)
                    cols_buf_probe = OPENCL_BUFFER_POOL.get(ctx, cols_size_probe, mf.READ_WRITE)
                    gsz_probe = (int(outW), int(outH), int(ksz))
                    # Use b_idx=0 for arg set
                    kn.set_args(
                        in_buf,
                        cols_buf_probe,
                        _np.int32(N),
                        _np.int32(C),
                        _np.int32(H),
                        _np.int32(W),
                        _np.int32(KH),
                        _np.int32(KW),
                        _np.int32(outH),
                        _np.int32(outW),
                        _np.int32(stride),
                        _np.int32(padding),
                        _np.int32(0),
                    )
                    best_t = 1e30
                    for lsz in lsz_cands:
                        try:
                            evt_probe = cl.enqueue_nd_range_kernel(q, kn, gsz_probe, lsz)
                            evt_probe.wait()
                            t = (
                                (evt_probe.profile.end - evt_probe.profile.start) * 1e-9
                                if hasattr(evt_probe, "profile")
                                else 0.0
                            )
                        except Exception:
                            t = 1e30
                        if 0 < t < best_t:
                            best_t = t
                            best_lsz = lsz
                except Exception:
                    best_lsz = None
                for b_idx in range(int(N)):
                    cols_size = int(ksz * outH * outW * 4)
                    cols_buf = OPENCL_BUFFER_POOL.get(ctx, cols_size, mf.READ_WRITE)
                    gsz = (int(outW), int(outH), int(ksz))
                    kn.set_args(
                        in_buf,
                        cols_buf,
                        _np.int32(N),
                        _np.int32(C),
                        _np.int32(H),
                        _np.int32(W),
                        _np.int32(KH),
                        _np.int32(KW),
                        _np.int32(outH),
                        _np.int32(outW),
                        _np.int32(stride),
                        _np.int32(padding),
                        _np.int32(b_idx),
                    )
                    # Capture im2col kernel timing via event profiling (using tuned best_lsz)
                    evt_col = cl.enqueue_nd_range_kernel(q, kn, gsz, best_lsz)
                    # Decide chunking across Cout to limit GEMM output buffer size
                    try:
                        max_chunk_bytes = int(_cfg.get("UHOP_OPENCL_CONV_CHUNK_MAX_BYTES") or (64 * 1024 * 1024))
                    except Exception:
                        max_chunk_bytes = 64 * 1024 * 1024
                    out2d_total_elems = int(Cout * outH * outW)
                    out2d_total_bytes = int(out2d_total_elems * 4)
                    chunked = out2d_total_bytes > max_chunk_bytes
                    co_step = int(Cout) if not chunked else max(1, int(max_chunk_bytes // (outH * outW * 4)))
                    if co_step < 1:
                        co_step = 1
                    # GEMM timing aggregates
                    import time as _time

                    gemm_ms_total = 0.0
                    gemm_ms_min = None
                    gemm_ms_max = None
                    tmp = _np.empty((Cout, outH, outW), dtype=_np.float32)
                    if not chunked:
                        try:
                            t0 = _time.perf_counter()
                            status = _clblast_sgemm(
                                lib,
                                int(q.int_ptr),
                                int(w_buf.int_ptr),
                                int(cols_buf.int_ptr),
                                # Allocate full out2d on-the-fly to device for copy-back
                                int(OPENCL_BUFFER_POOL.get(ctx, out2d_total_bytes, mf.WRITE_ONLY).int_ptr),
                                int(Cout),
                                int(outH * outW),
                                int(ksz),
                                1.0,
                                0.0,
                                False,
                                False,
                                0,
                                0,
                                0,
                            )
                            gemm_dt = (_time.perf_counter() - t0) * 1000.0
                            gemm_ms_total += gemm_dt
                            gemm_ms_min = gemm_dt if gemm_ms_min is None else min(gemm_ms_min, gemm_dt)
                            gemm_ms_max = gemm_dt if gemm_ms_max is None else max(gemm_ms_max, gemm_dt)
                            if status != 0:
                                lib = None
                                break
                        except Exception:
                            lib = None
                            break
                        # Copy back result in one go
                        out2d_buf = OPENCL_BUFFER_POOL.get(ctx, out2d_total_bytes, mf.READ_WRITE)
                        evt_copy = cl.enqueue_copy(q, tmp.reshape(Cout, outH * outW), out2d_buf)
                    else:
                        # Chunk across Cout: allocate smaller C buffer per chunk and copy to host slice
                        for co0 in range(0, int(Cout), co_step):
                            m_chunk = min(co_step, int(Cout) - co0)
                            out2d_chunk_bytes = int(m_chunk * outH * outW * 4)
                            c_chunk_buf = OPENCL_BUFFER_POOL.get(ctx, out2d_chunk_bytes, mf.WRITE_ONLY)
                            # Offsets in elements for A (weights) and C (output rows)
                            a_ld = int(ksz)
                            c_ld = int(outH * outW)
                            a_off = int(co0 * a_ld)
                            c_off = 0  # writing to chunk buffer has no offset
                            try:
                                t0 = _time.perf_counter()
                                status = _clblast_sgemm(
                                    lib,
                                    int(q.int_ptr),
                                    int(w_buf.int_ptr),
                                    int(cols_buf.int_ptr),
                                    int(c_chunk_buf.int_ptr),
                                    int(m_chunk),
                                    int(outH * outW),
                                    int(ksz),
                                    1.0,
                                    0.0,
                                    False,
                                    False,
                                    int(a_off),
                                    0,
                                    int(c_off),
                                )
                                gemm_dt = (_time.perf_counter() - t0) * 1000.0
                                gemm_ms_total += gemm_dt
                                gemm_ms_min = gemm_dt if gemm_ms_min is None else min(gemm_ms_min, gemm_dt)
                                gemm_ms_max = gemm_dt if gemm_ms_max is None else max(gemm_ms_max, gemm_dt)
                                if status != 0:
                                    lib = None
                                    break
                            except Exception:
                                lib = None
                                break
                            # Copy this chunk back and place into tmp slice
                            host_chunk = _np.empty((m_chunk, outH * outW), dtype=_np.float32)
                            cl.enqueue_copy(q, host_chunk, c_chunk_buf)
                            tmp[co0 : co0 + m_chunk] = host_chunk.reshape(m_chunk, outH, outW)
                        evt_copy = None
                    out[b_idx] = tmp
                q.finish()
                if lib is not None:
                    if fuse_relu:
                        out = _np.maximum(out, 0)
                    # Record profiling aggregate using last evt_col if available
                    try:
                        dt_ms = 0.0
                        try:
                            dt_ms = (
                                (evt_col.profile.end - evt_col.profile.start) * 1e-6
                                if hasattr(evt_col, "profile")
                                else 0.0
                            )
                        except Exception:
                            dt_ms = 0.0
                        # GEMM timing: prefer aggregated gemm_ms_total if measured via CPU timer in chunking path
                        gemm_ms = gemm_ms_total if "gemm_ms_total" in locals() and gemm_ms_total > 0 else None
                        copy_ms = None
                        try:
                            copy_ms = (
                                (evt_copy.profile.end - evt_copy.profile.start) * 1e-6
                                if hasattr(evt_copy, "profile")
                                else None
                            )
                        except Exception:
                            copy_ms = None
                        # Compute aggregate GFLOPS based on im2col+gemm total if gemm_ms known, else use im2col dt
                        total_ms_for_gflops = (gemm_ms or 0.0) + dt_ms
                        gflops = None
                        if total_ms_for_gflops > 0:
                            flops = N * outH * outW * Cout * Cin * KH * KW * 2.0
                            gflops = flops / (total_ms_for_gflops * 1e6)
                        self.auto.record_profile(
                            "opencl", "conv2d", "conv2d_im2col", dev_name, shape_key, gflops or 0.0, total_ms_for_gflops
                        )
                        self.auto.set_params(
                            "opencl",
                            "conv2d",
                            "conv2d_im2col",
                            dev_name,
                            shape_key,
                            {
                                "tuned_at": None,
                                "im2col_ms": dt_ms,
                                "gemm_ms": gemm_ms,
                                "copy_ms": copy_ms,
                                "lsz": list(best_lsz) if isinstance(best_lsz, tuple) else None,
                                "chunked": bool(chunked),
                                "chunk_count": int((int(Cout) + co_step - 1) // co_step) if chunked else 1,
                                "gemm_ms_min": float(gemm_ms_min) if gemm_ms_min is not None else None,
                                "gemm_ms_max": float(gemm_ms_max) if gemm_ms_max is not None else None,
                            },
                        )
                    except Exception:
                        pass
                    return Conv2DResult(out, "im2col_clblast")
            # fallthrough if lib unavailable

        # Tiled conv2d
        from ..opencl_backend import _load_conv2d_tiled_source

        src_conv = _load_conv2d_tiled_source()
        mf = cl.mem_flags
        out = _np.zeros((N, Cout, outH, outW), dtype=_np.float32)
        in_buf = OPENCL_BUFFER_POOL.get(ctx, input_np.nbytes, mf.READ_ONLY)
        w_buf = OPENCL_BUFFER_POOL.get(ctx, weight_np.nbytes, mf.READ_ONLY)
        out_buf = OPENCL_BUFFER_POOL.get(ctx, out.nbytes, mf.WRITE_ONLY)
        cl.enqueue_copy(q, in_buf, input_np)
        cl.enqueue_copy(q, w_buf, weight_np)
        # Simple candidate grid with vectorization choices
        try:
            raw = _cfg.get("UHOP_OPENCL_VEC_CANDIDATES") or [1]
            if isinstance(raw, str):
                vec_cands = [int(v.strip()) for v in raw.split(",") if v.strip()]
            else:
                vec_cands = [int(v) for v in raw]
            vec_cands = [v for v in vec_cands if v in (1, 2, 4, 8)] or [1]
        except Exception:
            vec_cands = [1]
        base_tiles = [(8, 8, 1), (16, 8, 1), (8, 16, 1), (16, 16, 1)]
        # TODO: apply perf heuristic using device metadata (local mem / compute units)
        best = None
        best_t = 1e30
        for tx, ty, tz in base_tiles:
            for vec in vec_cands:
                try:
                    prg = cl.Program(ctx, src_conv).build(options=f"-D TILE_W={tx} -D TILE_H={ty} -D VEC={vec}")
                    kn = cl.Kernel(prg, "conv2d_tiled")
                except Exception:
                    continue
                gsz_x = int(((outW + tx - 1) // tx) * tx)
                gsz_y = int(((outH + ty - 1) // ty) * ty)
                gsz = (gsz_x, gsz_y, int(N * Cout))
                tile_in_w = (tx - 1) * stride + KW
                tile_in_h = (ty - 1) * stride + KH
                tile_in_bytes = int(tile_in_w * tile_in_h * 4)
                tile_w_bytes = int(KH * KW * 4)
                # Local memory safety: skip if exceeding device local_mem_size
                try:
                    dev_local = ctx.devices[0].local_mem_size
                    if (tile_in_bytes + tile_w_bytes) > int(dev_local * 0.9):  # conservative 90%
                        continue
                except Exception:
                    pass
                kn.set_args(
                    in_buf,
                    w_buf,
                    out_buf,
                    _np.int32(N),
                    _np.int32(C),
                    _np.int32(H),
                    _np.int32(W),
                    _np.int32(Cout),
                    _np.int32(KH),
                    _np.int32(KW),
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
                    dt = (evt.profile.end - evt.profile.start) * 1e-9
                except Exception:
                    dt = 1e9
                if dt < best_t:
                    best_t = dt
                    best = (tx, ty, tz, vec, gsz, tile_in_bytes, tile_w_bytes)
        if best is None:
            raise RuntimeError("Failed to compile tiled conv2d candidates")
        tx, ty, tz, vec, gsz, tile_in_bytes, tile_w_bytes = best
        prg = cl.Program(ctx, src_conv).build(options=f"-D TILE_W={tx} -D TILE_H={ty} -D VEC={vec}")
        kn = cl.Kernel(prg, "conv2d_tiled")
        kn.set_args(
            in_buf,
            w_buf,
            out_buf,
            _np.int32(N),
            _np.int32(C),
            _np.int32(H),
            _np.int32(W),
            _np.int32(Cout),
            _np.int32(KH),
            _np.int32(KW),
            _np.int32(outH),
            _np.int32(outW),
            _np.int32(stride),
            _np.int32(padding),
            cl.LocalMemory(int(tile_in_bytes)),
            cl.LocalMemory(int(tile_w_bytes)),
        )
        evt = cl.enqueue_nd_range_kernel(q, kn, gsz, (tx, ty, tz))
        cl.enqueue_copy(q, out, out_buf, wait_for=[evt])
        q.finish()
        # Profiling record
        try:
            dt_ms = (evt.profile.end - evt.profile.start) * 1e-6 if hasattr(evt, "profile") else 0.0
            gflops = None
            if dt_ms > 0:
                flops = N * outH * outW * Cout * Cin * KH * KW * 2.0
                gflops = flops / (dt_ms * 1e6)
            self.auto.record_profile("opencl", "conv2d", "conv2d_tiled", dev_name, shape_key, gflops or 0.0, dt_ms)
            self.auto.set_params(
                "opencl",
                "conv2d",
                "conv2d_tiled",
                dev_name,
                shape_key,
                {
                    "tile_w": tx,
                    "tile_h": ty,
                    "vec": vec,
                    "tuned_at": float(evt.profile.end) if hasattr(evt, "profile") else None,
                },
            )
        except Exception:
            pass
        if fuse_relu:
            out = _np.maximum(out, 0)
        return Conv2DResult(out, "tiled", ms=dt_ms if "dt_ms" in locals() else None)


__all__ = ["Conv2DOp", "Conv2DResult"]
