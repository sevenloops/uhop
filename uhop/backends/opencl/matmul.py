"""Matmul operation module (refactored).

Provides MatmulOp with tune() and execute() methods, wrapping naive, tiled,
and optional CLBlast paths. Original logic extracted from monolithic file.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as _np

try:
    import pyopencl as cl  # type: ignore
except Exception:  # pragma: no cover
    cl = None  # type: ignore

from ... import config as _cfg
from ...cache import OPENCL_BUFFER_POOL, KernelRegistry
from ...utils.logging import get_logger as _get_logger
from ..opencl_backend import _load_matmul_tiled_source as _load_matmul_tiled_source
from .autotune import OCLAutotune
from .clblast import current_device_name, load_clblast_safe
from .context import get_ctx_queue, is_opencl_available
from .validation import _compute_local_size, prevalidate_matmul

_log = _get_logger("uhop.opencl.matmul")


@dataclass
class MatmulResult:
    output: _np.ndarray
    impl: str
    gflops: float | None = None
    ms: float | None = None
    validated_err: float | None = None
    unstable: bool = False


class MatmulOp:
    def __init__(self):
        self.auto = OCLAutotune()
        # Enable autotuned tiled path by default unless explicitly forced naive
        self._force_naive = str(
            os.environ.get("UHOP_OPENCL_FORCE_NAIVE") or _cfg.get("UHOP_OPENCL_FORCE_NAIVE") or "0"
        ).lower() in ("1", "true", "yes", "on")

    # --- Tuning helpers -------------------------------------------------
    def _tune_tiled(
        self, ctx, q, A: _np.ndarray, B: _np.ndarray, N: int, M: int, K: int, dev_name: str, shape_key: str, flip: bool
    ) -> Optional[dict]:
        """Brute-force scan tile and vector sizes, record best performing params.

        Returns params dict with chosen tile/vec and timing, or None if failed.
        Records profile metrics and sets autotune params with 'tuned_at'.
        """
        if self.auto.disabled:
            return None
        # Capture previous params so we can annotate retune origin and clear suggestion flags
        prev_params = self.auto.get_params("opencl", "matmul", "matmul_tiled", dev_name, shape_key)
        prev_tuned_at = prev_params.get("tuned_at") if isinstance(prev_params, dict) else None
        vec_cands = [1, 2, 4]
        best = None
        best_t = 1e30
        # track best parameters; avoid unused variable warnings
        # best_meta = {}
        for tile in [8, 16, 32]:
            for vec in vec_cands:
                try:
                    # Use centralized loader so tests can monkeypatch failures
                    src = _load_matmul_tiled_source()
                except Exception:
                    continue
                build_opts = f"-D TILE={tile} -D VEC={vec}" + (" -D GWS_FLIP=1" if flip else "")
                try:
                    prg = cl.Program(ctx, src).build(options=build_opts)
                except Exception:
                    continue
                try:
                    kn = cl.Kernel(prg, "matmul_tiled")
                except Exception:
                    continue
                if flip:
                    gsz = ((N + tile - 1) // tile * tile, (K + tile - 1) // tile * tile)
                else:
                    gsz = ((K + tile - 1) // tile * tile, (N + tile - 1) // tile * tile)
                mf = cl.mem_flags
                try:
                    a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
                    b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
                    c_buf = cl.Buffer(ctx, mf.WRITE_ONLY, size=N * K * 4)
                except Exception:
                    continue
                try:
                    kn.set_args(a_buf, b_buf, c_buf, _np.int32(N), _np.int32(M), _np.int32(K))
                    lsz = _compute_local_size(tile, q.device)
                    evt = cl.enqueue_nd_range_kernel(q, kn, gsz, lsz)
                    evt.wait()
                    dt = (evt.profile.end - evt.profile.start) * 1e-9 if hasattr(evt, "profile") else 0.0
                except Exception:
                    # Skip invalid candidates (e.g., oversized local size)
                    continue
                if dt < best_t:
                    best_t = dt
                    best = (tile, vec, kn, a_buf, b_buf, c_buf, gsz)
                    # best_meta = {"tile": tile, "vec": vec, "dt_s": dt}
        if best is None:
            return None
        # Prevalidate best kernel before committing params
        tile, vec, kn, a_buf, b_buf, c_buf, gsz = best
        errv = prevalidate_matmul(ctx, q, kn, tile, flip)
        # Treat NaN validation error as unstable
        if _np.isnan(errv):
            errv = 1e9
        params = {"tile": tile, "vec": vec, "prevalidate_err": errv, "tuned_at": time.time()}
        # Clear any previous retune suggestion now that we're retuning
        params["retune_suggested"] = False
        if prev_tuned_at is not None:
            params["retuned_from"] = prev_tuned_at
        if errv > 1e-2:
            params["unstable"] = True
            self.auto.set_params("opencl", "matmul", "matmul_tiled", dev_name, shape_key, params)
            return params
        # Record performance
        gflops = (2.0 * N * M * K) / (best_t * 1e9) if best_t > 0 else 0.0
        try:
            self.auto.record_profile("opencl", "matmul", "matmul_tiled", dev_name, shape_key, gflops, best_t * 1000.0)
        except Exception:
            pass
        self.auto.set_params("opencl", "matmul", "matmul_tiled", dev_name, shape_key, params)
        return params

    def _naive_kernel(self, ctx, vec: int) -> Tuple[cl.Kernel, cl.Program]:  # type: ignore
        src = r"""
        __kernel void matmul_naive(
            __global const float* A,
            __global const float* B,
            __global float* C,
            const int N,
            const int M,
            const int K
        ) {
            int row = get_global_id(1);
            int col_base = get_global_id(0) * {VEC};
            if (row < N) {
                for (int v=0; v<{VEC}; ++v) {
                    int col = col_base + v;
                    if (col < K) {
                        float acc = 0.0f;
                        for (int m2=0; m2<M; ++m2) {
                            acc += A[row*M + m2] * B[m2*K + col];
                        }
                        C[row*K + col] = acc;
                    }
                }
            }
        }
        """.replace(
            "{VEC}", str(int(vec))
        )
        prg = KernelRegistry().load_opencl_binary("device", "naive" + str(vec))  # placeholder bin reuse attempt
        if not isinstance(prg, str):
            prg_obj = cl.Program(ctx, src).build()
        else:
            prg_obj = cl.Program(ctx, src).build()
        kn = cl.Kernel(prg_obj, "matmul_naive")
        return kn, prg_obj

    def execute(self, A: Any, B: Any) -> MatmulResult:
        if not is_opencl_available():
            raise RuntimeError("OpenCL unavailable")
        ctx, q = get_ctx_queue()
        A = _np.array(A, dtype=_np.float32, order="C")
        B = _np.array(B, dtype=_np.float32, order="C")
        # Correct dimension mapping: A[N,M] * B[M,K] => C[N,K]
        N, M = A.shape
        M2, K = B.shape
        assert M == M2, "matmul inner dims mismatch"
        C = _np.zeros((N, K), dtype=_np.float32)
        # Prefer explicit env override; otherwise default to tiled unless naive is forced
        impl_env = os.environ.get("UHOP_OPENCL_MATMUL_IMPL")
        impl = str(impl_env if impl_env else ("naive" if self._force_naive else "tiled")).lower()
        dev_name = current_device_name()
        shape_key = f"N{N}_M{M}_K{K}"

        # CLBlast path
        if impl == "clblast":
            lib = load_clblast_safe()
            if lib is not None:
                try:
                    from ..clblast_integration import (
                        sgemm as _clblast_sgemm,  # type: ignore
                    )

                    mf = cl.mem_flags
                    a_buf = OPENCL_BUFFER_POOL.get(ctx, A.nbytes, mf.READ_ONLY)
                    b_buf = OPENCL_BUFFER_POOL.get(ctx, B.nbytes, mf.READ_ONLY)
                    c_buf = OPENCL_BUFFER_POOL.get(ctx, C.nbytes, mf.WRITE_ONLY)
                    cl.enqueue_copy(q, a_buf, A)
                    cl.enqueue_copy(q, b_buf, B)
                    status = _clblast_sgemm(
                        lib, int(q.int_ptr), int(a_buf.int_ptr), int(b_buf.int_ptr), int(c_buf.int_ptr), M, N, K
                    )
                    if status == 0:
                        cl.enqueue_copy(q, C, c_buf)
                        q.finish()
                        return MatmulResult(C, impl)
                except Exception as e:
                    _log.warning(f"CLBlast failure: {e}; falling back to tiled")
                    self.auto.set_params("opencl", "clblast", "sgemm", dev_name, "device", {"unstable": True})
                impl = "tiled"

        # Tiled path
        if impl == "tiled":
            flip = bool(_cfg.get("UHOP_OPENCL_FLIP_GWS"))
            params = self.auto.get_params("opencl", "matmul", "matmul_tiled", dev_name, shape_key)
            # Decide if we need to retune
            if self.auto.needs_retune("opencl", "matmul", "matmul_tiled", dev_name, shape_key):
                params = self._tune_tiled(ctx, q, A, B, N, M, K, dev_name, shape_key, flip)
            if not params or params.get("unstable"):
                impl = "naive"
            else:
                tile = int(params.get("tile", 16))
                vec = int(params.get("vec", 1))
                try:
                    # Use centralized loader so tests can monkeypatch failures
                    src = _load_matmul_tiled_source()
                except Exception as e:
                    _log.warning(f"tiled source load failed {e}; fallback naive")
                    impl = "naive"
                    src = None  # type: ignore
                build_opts = f"-D TILE={tile} -D VEC={vec}" + (" -D GWS_FLIP=1" if flip else "")
                try:
                    if src is None:
                        raise RuntimeError("no source for tiled kernel")
                    prg = cl.Program(ctx, src).build(options=build_opts)
                    kn = cl.Kernel(prg, "matmul_tiled")
                except Exception as e:
                    _log.warning(f"tiled build failed {e}; fallback naive")
                    impl = "naive"
                else:
                    if flip:
                        gsz = ((N + tile - 1) // tile * tile, (K + tile - 1) // tile * tile)
                    else:
                        gsz = ((K + tile - 1) // tile * tile, (N + tile - 1) // tile * tile)
                    mf = cl.mem_flags
                    a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
                    b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
                    c_buf = cl.Buffer(ctx, mf.WRITE_ONLY, size=C.nbytes)
                    try:
                        kn.set_args(a_buf, b_buf, c_buf, _np.int32(N), _np.int32(M), _np.int32(K))
                        lsz = _compute_local_size(tile, q.device)
                        evt = cl.enqueue_nd_range_kernel(q, kn, gsz, lsz)
                        cl.enqueue_copy(q, C, c_buf, wait_for=[evt])
                        q.finish()
                        # Guard against driver bugs: if output contains NaN/Inf, mark unstable and fallback to naive
                        if not _np.isfinite(C).all():
                            try:
                                self.auto.set_params(
                                    "opencl", "matmul", "matmul_tiled", dev_name, shape_key, {"unstable": True}
                                )
                            except Exception:
                                pass
                            impl = "naive"
                        else:
                            dt = (evt.profile.end - evt.profile.start) * 1e-9 if hasattr(evt, "profile") else 0.0
                            gflops = (2.0 * N * M * K) / (dt * 1e9) if dt > 0 else 0.0
                            try:
                                self.auto.record_profile(
                                    "opencl", "matmul", "matmul_tiled", dev_name, shape_key, gflops, dt * 1000.0
                                )
                            except Exception:
                                pass
                            return MatmulResult(
                                C,
                                "tiled",
                                ms=dt * 1000.0,
                                validated_err=params.get("prevalidate_err"),
                                gflops=gflops if dt > 0 else None,
                            )
                    except Exception as e:
                        _log.warning(f"tiled exec failed {e}; fallback naive")
                        impl = "naive"

        # Naive path (default or fallback)
        vec = 1
        kn, prg = self._naive_kernel(ctx, vec)
        mf = cl.mem_flags
        a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
        b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
        c_buf = cl.Buffer(ctx, mf.WRITE_ONLY, size=C.nbytes)
        gsz = ((K + vec - 1) // vec, N)
        kn.set_args(a_buf, b_buf, c_buf, _np.int32(N), _np.int32(M), _np.int32(K))
        # Use a conservative local size to avoid driver bugs on some devices.
        # Some AMD drivers return INVALID_WORK_GROUP_SIZE if local size is None here
        # or produce NaNs for larger inferred sizes.
        evt = cl.enqueue_nd_range_kernel(q, kn, gsz, (1, 1))
        cl.enqueue_copy(q, C, c_buf, wait_for=[evt])
        q.finish()
        try:
            dt_ms = (evt.profile.end - evt.profile.start) * 1e-6 if hasattr(evt, "profile") else 0.0
        except Exception:
            dt_ms = 0.0
        try:
            gflops = (2.0 * N * M * K) / (dt_ms * 1e6) if dt_ms > 0 else 0.0
            self.auto.record_profile("opencl", "matmul", "matmul_naive", dev_name, shape_key, gflops, dt_ms)
        except Exception:
            gflops = None
        return MatmulResult(C, "naive", ms=dt_ms, gflops=gflops)


__all__ = ["MatmulOp", "MatmulResult"]
