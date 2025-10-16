# uhop/optimizer.py
"""
UHOP optimizer with per-shape backend caching, reduced dispatch overhead, and integration with backend caches.
Priority:
  1) Torch backend (CUDA / ROCm-built PyTorch / MPS)
  2) Triton backend (triton)
  3) OpenCL backend (pyopencl)
  4) AI-generated kernels (CUDA via pycuda if available, else Python via sandbox)
  5) NumPy baseline (fallback)
"""
from __future__ import annotations
import importlib.util
import os
from functools import wraps
from pathlib import Path
from typing import Callable, Any, Optional, Tuple
import numpy as np

from .hardware import detect_hardware, HardwareProfile
from .cache import UhopCache
from .core.benchmark import benchmark_callable
from .sandbox import run_generated_python
from .ai_codegen.generator import AICodegen

# backends
from .backends import (
    is_torch_available,
    torch_matmul,
    torch_conv2d,
    torch_relu,
    is_triton_available,
    triton_matmul,
    triton_conv2d,
    triton_relu,
    is_opencl_available,
    opencl_matmul,
    opencl_conv2d,
    opencl_relu,
)

CACHE = UhopCache()

# (removed unused launch autotune stubs)


def _shape_key_from_args(op_name: str, args: tuple) -> str:
    """
    Create a string key representing operation + shapes so we can cache backend decisions per shape.
    """
    shapes = []
    for a in args:
        try:
            # numpy array or torch tensor; use shape tuple
            shapes.append(tuple(np.shape(a)))
        except Exception:
            shapes.append(str(type(a)))
    return f"{op_name}|" + "|".join([str(s) for s in shapes])


class UHopOptimizer:
    def __init__(self):
        self.hw = detect_hardware()
        self.cache = CACHE
        self.codegen = AICodegen()
        # JIT registry: first successful backend per op_name (session-only)
        self._jit_registry = {}
        # optional runtime flags
        try:
            import pycuda  # type: ignore  # noqa: F401

            self._pycuda_available = True
        except Exception:
            self._pycuda_available = False

    def _run_opencl_generated_kernel(
        self,
        op_name: str,
        source_path: Path,
        kernel_name: str,
        *args,
        **kwargs,
    ):
        """
        Execute an AI-generated OpenCL kernel for a small set of known ops.
        Supported ops and expected inputs:
          - matmul: (A: np.ndarray[M,K], B: np.ndarray[K,N]) -> np.ndarray[M,N]
          - relu: (X: np.ndarray[...]) -> np.ndarray[...]
          - conv2d: (X: np.ndarray[N,C,H,W], W: np.ndarray[Cout,Cin,KH,KW], stride=1, padding=0)
        """
        try:
            import pyopencl as cl  # type: ignore
            import numpy as _np
        except Exception as e:
            raise RuntimeError(f"ai_opencl requested but pyopencl not available: {e}")

        # Build program
        ctx = None
        try:
            plats = cl.get_platforms()
            for p in plats:
                gpus = [d for d in p.get_devices() if d.type & cl.device_type.GPU]
                if gpus:
                    ctx = cl.Context(devices=[gpus[0]])
                    break
            if ctx is None:
                ctx = cl.create_some_context(interactive=False)
        except Exception:
            ctx = cl.create_some_context(interactive=False)
        q = cl.CommandQueue(ctx)

        src = source_path.read_text()
        prg = cl.Program(ctx, src).build()

        if op_name == "matmul":
            A, B = args[:2]
            A = _np.array(A, dtype=_np.float32, order="C")
            B = _np.array(B, dtype=_np.float32, order="C")
            M, K = A.shape
            K2, N = B.shape
            assert K == K2, "Inner dims must match"
            C = _np.empty((M, N), dtype=_np.float32)
            mf = cl.mem_flags
            a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
            b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
            c_buf = cl.Buffer(ctx, mf.WRITE_ONLY, C.nbytes)
            kn = cl.Kernel(prg, kernel_name)
            kn.set_args(_np.int32(M), _np.int32(N), _np.int32(K), a_buf, b_buf, c_buf)
            gsz = (int(M), int(N))
            cl.enqueue_nd_range_kernel(q, kn, gsz, None)
            q.finish()
            cl.enqueue_copy(q, C, c_buf)
            q.finish()
            return C

        if op_name == "relu":
            X = _np.array(args[0], dtype=_np.float32, order="C")
            flat = X.reshape(-1)
            Y = _np.empty_like(flat)
            mf = cl.mem_flags
            x_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=flat)
            y_buf = cl.Buffer(ctx, mf.WRITE_ONLY, Y.nbytes)
            kn = cl.Kernel(prg, kernel_name)
            N = _np.int32(flat.size)
            kn.set_args(N, x_buf, y_buf)
            cl.enqueue_nd_range_kernel(q, kn, (int(N),), None)
            q.finish()
            cl.enqueue_copy(q, Y, y_buf)
            q.finish()
            return Y.reshape(X.shape)

        if op_name == "conv2d":
            X, Wt = args[:2]
            stride = int(kwargs.get("stride", 1))
            padding = int(kwargs.get("padding", 0))
            X = _np.array(X, dtype=_np.float32, order="C")
            Wt = _np.array(Wt, dtype=_np.float32, order="C")
            N, C_in, H, W = X.shape
            C_out, Cin2, KH, KW = Wt.shape
            assert C_in == Cin2
            outH = (H + 2 * padding - KH) // stride + 1
            outW = (W + 2 * padding - KW) // stride + 1
            Y = _np.empty((N, C_out, outH, outW), dtype=_np.float32)
            mf = cl.mem_flags
            x_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=X)
            w_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=Wt)
            y_buf = cl.Buffer(ctx, mf.WRITE_ONLY, Y.nbytes)
            # Expect a kernel like `generated_conv2d` with signature used in CLI smoke test
            kn = cl.Kernel(prg, kernel_name)
            kn.set_args(
                _np.int32(N), _np.int32(C_in), _np.int32(H), _np.int32(W),
                _np.int32(C_out), _np.int32(KH), _np.int32(KW),
                _np.int32(stride), _np.int32(padding),
                x_buf, w_buf, y_buf,
                _np.int32(outH), _np.int32(outW),
            )
            gsz = (int(outW), int(outH), int(N * C_out))
            cl.enqueue_nd_range_kernel(q, kn, gsz, None)
            q.finish()
            cl.enqueue_copy(q, Y, y_buf)
            q.finish()
            return Y

        raise RuntimeError(f"ai_opencl not implemented for op: {op_name}")

    # Helper to compile-run CUDA source using PyCUDA SourceModule (if available)
    def _run_cuda_source_via_pycuda(
        self,
        source_path: Path,
        kernel_name: str,
        a: np.ndarray,
        b: Optional[np.ndarray] = None,
    ):
        if not self._pycuda_available:
            raise RuntimeError("pycuda not available")
        from pycuda.compiler import SourceModule  # type: ignore
        import pycuda.driver as cuda  # type: ignore
        import numpy as _np

        src = source_path.read_text()
        mod = SourceModule(src)
        func = mod.get_function(kernel_name)
        if b is None:
            raise RuntimeError("CUDA kernel expects two inputs (a,b)")
        A = _np.array(a, dtype=_np.float32)
        B = _np.array(b, dtype=_np.float32)
        N, M = A.shape
        M2, K = B.shape
        assert M == M2
        C = _np.empty((N, K), dtype=_np.float32)
        A_gpu = cuda.mem_alloc(A.nbytes)
        B_gpu = cuda.mem_alloc(B.nbytes)
        C_gpu = cuda.mem_alloc(C.nbytes)
        cuda.memcpy_htod(A_gpu, A)
        cuda.memcpy_htod(B_gpu, B)
        block = (16, 16, 1)
        grid = ((K + block[0] - 1) // block[0], (N + block[1] - 1) // block[1])
        func(
            A_gpu,
            B_gpu,
            C_gpu,
            np.int32(N),
            np.int32(M),
            np.int32(K),
            block=block,
            grid=grid,
        )
        cuda.memcpy_dtoh(C, C_gpu)
        return C

    def optimize(self, op_name: str, sandbox_timeout: int = 8):
        """
        Decorator that optimizes a fallback Python function `fn(*args, **kwargs)` for op_name.
        Results are cached per operation + input shapes to skip detection & branching overhead.
        """

        def decorator(fn: Callable):
            @wraps(fn)
            def wrapper(*args, **kwargs):
                # shape-aware cache key
                shape_key = _shape_key_from_args(op_name, args)
                # Try shape-specific cache first, then a generic op-level entry (used by CLI for AI kernels)
                cached = self.cache.get(shape_key) or self.cache.get(op_name)
                # If a JIT preferred backend was recorded for this op, try it first (avoids repeated checks)
                jit_pref = self._jit_registry.get(op_name)
                if jit_pref and not cached:
                    try:
                        if jit_pref == "torch":
                            if op_name == "matmul": return torch_matmul(*args)
                            if op_name == "conv2d": return torch_conv2d(*args, **kwargs)
                            if op_name == "relu":   return torch_relu(*args)
                        if jit_pref == "triton":
                            if op_name == "matmul": return triton_matmul(*args)
                            if op_name == "relu":   return triton_relu(*args)
                        if jit_pref == "opencl":
                            if op_name == "matmul": return opencl_matmul(*args)
                            if op_name == "conv2d": return opencl_conv2d(*args, **kwargs)
                            if op_name == "relu":   return opencl_relu(*args)
                    except Exception:
                        # fallback to normal path if JIT pref fails (device lost, etc.)
                        pass
                # If we have a backend decision cached for this shape, try it first (fast path)
                if cached:
                    backend = cached.get("backend")
                    try:
                        if backend == "torch":
                            if op_name == "matmul":
                                return torch_matmul(*args)
                            if op_name == "conv2d":
                                return torch_conv2d(*args, **kwargs)
                            if op_name == "relu":
                                return torch_relu(*args)
                        if backend == "triton":
                            if op_name == "matmul":
                                return triton_matmul(*args)
                            if op_name == "relu":
                                return triton_relu(*args)
                        if backend == "opencl":
                            if op_name == "matmul":
                                return opencl_matmul(*args)
                            if op_name == "conv2d":
                                # prefer fused path if stored
                                fuse = cached.get("fused", False)
                                return opencl_conv2d(*args, fuse_relu=fuse, **kwargs)
                            if op_name == "relu":
                                return opencl_relu(*args)
                        if backend == "ai_opencl":
                            path = cached.get("path")
                            kernel_name = cached.get("kernel_name", f"generated_{op_name}")
                            return self._run_opencl_generated_kernel(
                                op_name, Path(path), kernel_name, *args, **kwargs
                            )
                        if backend == "ai_cuda":
                            path = cached.get("path")
                            kernel_name = cached.get("kernel_name", f"{op_name}_kernel")
                            return self._run_cuda_source_via_pycuda(
                                Path(path), kernel_name, *args[:2]
                            )
                        if backend == "ai_python":
                            path = cached.get("path")
                            fn_name = cached.get("function", f"generated_{op_name}")
                            return run_generated_python(
                                path, fn_name, *args, timeout=sandbox_timeout
                            )
                    except Exception:
                        # cached backend failed (e.g., device removed); drop cache and fall through
                        self.cache.set(shape_key, None)

                # Fast-path detection order: Torch -> Triton -> OpenCL
                # Note: we avoid expensive per-call checks later if a backend is chosen and cached.
                # 1) Torch (GPU-only). Skip if no GPU to avoid slow CPU vs NumPy baselines.
                try:
                    if is_torch_available():
                        try:
                            import torch  # type: ignore
                            has_gpu = bool(getattr(torch, "cuda", None) and torch.cuda.is_available()) or (
                                getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
                            )
                        except Exception:
                            has_gpu = False
                        if not has_gpu:
                            raise RuntimeError("torch GPU not available; skipping torch backend")
                        # If inputs are tensors, torch backend will avoid copies (torch_backend handles)
                        if op_name == "matmul":
                            res = torch_matmul(*args)
                            self.cache.set(
                                shape_key,
                                {"backend": "torch", "hardware": self.hw.__dict__},
                            )
                            self._jit_registry.setdefault(op_name, "torch")
                            return res
                        if op_name == "conv2d":
                            res = torch_conv2d(*args, **kwargs)
                            self.cache.set(
                                shape_key,
                                {"backend": "torch", "hardware": self.hw.__dict__},
                            )
                            self._jit_registry.setdefault(op_name, "torch")
                            return res
                        if op_name == "relu":
                            res = torch_relu(*args)
                            self.cache.set(
                                shape_key,
                                {"backend": "torch", "hardware": self.hw.__dict__},
                            )
                            self._jit_registry.setdefault(op_name, "torch")
                            return res
                except Exception:
                    pass

                # 2) Triton
                try:
                    if is_triton_available():
                        if op_name == "matmul":
                            res = triton_matmul(*args)
                            self.cache.set(
                                shape_key,
                                {"backend": "triton", "hardware": self.hw.__dict__},
                            )
                            self._jit_registry.setdefault(op_name, "triton")
                            return res
                        if op_name == "relu":
                            res = triton_relu(*args)
                            self._jit_registry.setdefault(op_name, "triton")
                            return res
                except Exception:
                    pass

                # 3) OpenCL (skip very small sizes to avoid transfer/launch overhead dominating)
                try:
                    if is_opencl_available():
                        # For conv2d we attempt fused conv2d+relu when beneficial
                        if op_name == "matmul":
                            a, b = args[:2]
                            try:
                                m, k = np.shape(a)
                                k2, n = np.shape(b)
                                prob = m * n * k
                            except Exception:
                                prob = 0
                            if prob < 256 * 256 * 256:
                                raise RuntimeError("problem too small for OpenCL")
                            res = opencl_matmul(*args)
                            self.cache.set(
                                shape_key,
                                {"backend": "opencl", "hardware": self.hw.__dict__},
                            )
                            self._jit_registry.setdefault(op_name, "opencl")
                            return res
                        if op_name == "conv2d":
                            # try fused conv2d + relu if caller likely has relu next: we can't always know,
                            # but we prefer fused when input shapes are moderate sized (heuristic)
                            fuse = kwargs.get("fuse_relu", False)
                            res = opencl_conv2d(*args, fuse_relu=fuse, **kwargs)
                            self.cache.set(
                                shape_key,
                                {
                                    "backend": "opencl",
                                    "hardware": self.hw.__dict__,
                                    "fused": fuse,
                                },
                            )
                            self._jit_registry.setdefault(op_name, "opencl")
                            return res
                        if op_name == "relu":
                            x = np.array(args[0])
                            if x.size < (1 << 20):  # < ~1M elements
                                raise RuntimeError("relu size too small for OpenCL")
                            res = opencl_relu(*args)
                            self._jit_registry.setdefault(op_name, "opencl")
                            return res
                except Exception:
                    pass

                # 4) NumPy baseline and AI generation fallback.
                # For small problems where GPU is skipped, use baseline immediately.
                if op_name == "matmul":
                    try:
                        a, b = args[:2]
                        m, k = np.shape(a)
                        k2, n = np.shape(b)
                        if m * n * k < 256 * 256 * 256:
                            self.cache.set(shape_key, {"backend": "numpy", "hardware": self.hw.__dict__})
                            return fn(*args, **kwargs)
                    except Exception:
                        pass
                if op_name == "relu":
                    try:
                        if np.array(args[0]).size < (1 << 20):
                            self.cache.set(shape_key, {"backend": "numpy", "hardware": self.hw.__dict__})
                            return fn(*args, **kwargs)
                    except Exception:
                        pass

                # Benchmark baseline otherwise
                try:
                    t_base = benchmark_callable(
                        lambda: fn(*args, **kwargs), runs=3, warmup=2, sync_fn=None
                    )
                except Exception:
                    t_base = float("inf")

                # Attempt AI generation (CUDA if pycuda available else python)
                t_ai = float("inf")
                ai_path = None
                ai_backend = None
                try:
                    target = "cuda" if self._pycuda_available else "python"
                    ai_path = self.codegen.generate(op_name, target=target)
                    if target == "cuda":
                        try:
                            out = self._run_cuda_source_via_pycuda(
                                ai_path, f"{op_name}_kernel", *args[:2]
                            )
                            # benchmark with simple sync function (none required for cpu-bound)
                            t_ai = benchmark_callable(
                                lambda: self._run_cuda_source_via_pycuda(
                                    ai_path, f"{op_name}_kernel", *args[:2]
                                ),
                                runs=2,
                                warmup=1,
                            )
                            ai_backend = "ai_cuda"
                        except Exception:
                            ai_path = None
                            t_ai = float("inf")
                    else:
                        try:
                            fn_name = f"generated_{op_name}"
                            out = run_generated_python(
                                str(ai_path), fn_name, *args, timeout=sandbox_timeout
                            )
                            t_ai = benchmark_callable(
                                lambda: run_generated_python(
                                    str(ai_path),
                                    fn_name,
                                    *args,
                                    timeout=sandbox_timeout,
                                ),
                                runs=2,
                                warmup=1,
                            )
                            ai_backend = "ai_python"
                        except Exception:
                            ai_path = None
                            t_ai = float("inf")
                except Exception:
                    ai_path = None
                    t_ai = float("inf")

                # Choose winner
                if t_ai < t_base and ai_path:
                    if ai_backend == "ai_cuda":
                        self.cache.set(
                            shape_key,
                            {
                                "backend": "ai_cuda",
                                "path": str(ai_path),
                                "kernel_name": f"{op_name}_kernel",
                                "hardware": self.hw.__dict__,
                            },
                        )
                        return self._run_cuda_source_via_pycuda(
                            ai_path, f"{op_name}_kernel", *args[:2]
                        )
                    else:
                        self.cache.set(
                            shape_key,
                            {
                                "backend": "ai_python",
                                "path": str(ai_path),
                                "function": f"generated_{op_name}",
                                "hardware": self.hw.__dict__,
                            },
                        )
                        return run_generated_python(
                            str(ai_path),
                            f"generated_{op_name}",
                            *args,
                            timeout=sandbox_timeout,
                        )
                else:
                    self.cache.set(
                        shape_key,
                        {
                            "backend": "numpy",
                            "path": None,
                            "function": fn.__name__,
                            "hardware": self.hw.__dict__,
                        },
                    )
                    return fn(*args, **kwargs)

            return wrapper

        return decorator


# convenience global optimizer for decorators
_GLOBAL_OPT = UHopOptimizer()


def optimize(op_name: str):
    return _GLOBAL_OPT.optimize(op_name)
