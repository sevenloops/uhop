# uhop/optimizer.py
"""
UHOP optimizer: multi-backend decision flow (simple and robust).

Order of preference:
  1) Torch accelerator (CUDA/MPS) on macOS or when available
  2) Triton (matmul/relu when installed)
  3) OpenCL (matmul/conv2d/relu)
  4) Torch CPU
  5) AI-generated kernel (CUDA via pycuda or Python) if faster than baseline
  6) NumPy baseline

Cache is per op-name (MVP). Optional strict validation can gate AI kernels.
"""
from __future__ import annotations

import os
import platform
from functools import wraps
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .hardware import detect_hardware
from .cache import UhopCache
from .core.benchmark import benchmark_callable
from .sandbox import run_generated_python
from .ai_codegen.generator import AICodegen
from .validation import validate_kernel

# backends
from .backends import (
    is_torch_available,
    torch_matmul,
    torch_conv2d,
    torch_relu,
    is_triton_available,
    triton_matmul,
    triton_relu,
    is_opencl_available,
    opencl_matmul,
    opencl_conv2d,
    opencl_relu,
)


def _torch_has_accelerator() -> bool:
    try:
        import torch  # type: ignore
        return (
            bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
            or (
                getattr(torch.backends, "mps", None) is not None
                and torch.backends.mps.is_available()
            )
        )
    except Exception:
        return False


CACHE = UhopCache()


class UHopOptimizer:
    def __init__(self):
        self.hw = detect_hardware()
        self.cache = CACHE
        self.codegen = AICodegen()
        try:
            import pycuda  # type: ignore  # noqa: F401
            self._pycuda_available = True
        except Exception:
            self._pycuda_available = False

    # Helper: compile-run CUDA via PyCUDA SourceModule
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
        n, m = A.shape
        m2, k = B.shape
        assert m == m2
        C = _np.empty((n, k), dtype=_np.float32)
        a_gpu = cuda.mem_alloc(A.nbytes)
        b_gpu = cuda.mem_alloc(B.nbytes)
        c_gpu = cuda.mem_alloc(C.nbytes)
        cuda.memcpy_htod(a_gpu, A)
        cuda.memcpy_htod(b_gpu, B)
        block = (16, 16, 1)
        grid = ((k + block[0] - 1) // block[0], (n + block[1] - 1) // block[1])
        func(
            a_gpu,
            b_gpu,
            c_gpu,
            _np.int32(n),
            _np.int32(m),
            _np.int32(k),
            block=block,
            grid=grid,
        )
        cuda.memcpy_dtoh(C, c_gpu)
        return C

    def optimize(self, op_name: str, sandbox_timeout: int = 8):
        """
        Decorator that optimizes a fallback Python function for op_name.
        Baseline fn must accept array-like inputs and return a NumPy array.
        """

        def decorator(fn: Callable):
            def _sig_from_val(v):
                try:
                    import torch  # type: ignore
                    if isinstance(v, torch.Tensor):
                        dev = getattr(v.device, "type", "cpu")
                        return (
                            "torch",
                            tuple(int(x) for x in v.shape),
                            str(v.dtype).replace("torch.", ""),
                            dev,
                        )
                except Exception:
                    pass
                try:
                    import numpy as _np
                    if isinstance(v, _np.ndarray):
                        return (
                            "numpy",
                            tuple(int(x) for x in v.shape),
                            str(v.dtype),
                        )
                except Exception:
                    pass
                # Fallback to type name
                return (type(v).__name__,)

            @wraps(fn)
            def wrapper(*args, **kwargs):
                # Quick override for tests/dev: force calling the Python
                # baseline implementation and skip backend selection.
                if os.environ.get("UHOP_FORCE_BASELINE", "0") not in (
                    "0",
                    "false",
                    "False",
                    "",
                    None,
                ):
                    return fn(*args, **kwargs)

                if len(args) < 1:
                    raise ValueError("need at least one positional arg")
                a = args[0]
                b = args[1] if len(args) > 1 else None
                # Optional per-shape/dtype caching
                per_shape = os.environ.get("UHOP_CACHE_PER_SHAPE", "0")
                per_shape = per_shape not in ("0", "false", "False", None)
                if per_shape:
                    parts = [str(_sig_from_val(a))]
                    if b is not None:
                        parts.append(str(_sig_from_val(b)))
                    cache_key = f"{op_name}|" + ";".join(parts)
                else:
                    cache_key = op_name

                # 0) Cached backend first
                rec = self.cache.get(cache_key)
                if rec:
                    backend = rec.get("backend")
                    try:
                        if backend == "torch":
                            if op_name == "matmul":
                                return torch_matmul(a, b)
                            if op_name == "conv2d":
                                return torch_conv2d(
                                    a,
                                    b,
                                    stride=kwargs.get("stride", 1),
                                    padding=kwargs.get("padding", 0),
                                )
                            if op_name == "relu":
                                return torch_relu(a)
                        if backend == "triton":
                            if op_name == "matmul":
                                return triton_matmul(a, b)
                            if op_name == "relu":
                                return triton_relu(a)
                        if backend == "opencl":
                            if op_name == "matmul":
                                return opencl_matmul(a, b)
                            if op_name == "conv2d":
                                return opencl_conv2d(
                                    a,
                                    b,
                                    stride=kwargs.get("stride", 1),
                                    padding=kwargs.get("padding", 0),
                                )
                            if op_name == "relu":
                                return opencl_relu(a)
                        if backend == "ai_cuda":
                            path = rec.get("path")
                            kname = rec.get("kernel_name", f"{op_name}_kernel")
                            arr0 = np.array(a)
                            arr1 = np.array(b) if b is not None else None
                            return self._run_cuda_source_via_pycuda(
                                Path(path), kname, arr0, arr1
                            )
                        if backend == "ai_python":
                            path = rec.get("path")
                            fname = rec.get(
                                "function", f"generated_{op_name}"
                            )
                            arr0 = np.array(a)
                            arr1 = np.array(b) if b is not None else None
                            return run_generated_python(
                                str(path),
                                fname,
                                arr0,
                                arr1,
                                timeout=sandbox_timeout,
                            )
                    except Exception:
                        pass

                # 0.5) Environment override order
                # Example: UHOP_BACKEND_PREFERENCE="opencl,torch,triton,cpu,numpy"
                pref = os.environ.get("UHOP_BACKEND_PREFERENCE")
                if pref:
                    order = [p.strip().lower() for p in pref.split(',') if p.strip()]

                    def _try_backend(name: str):
                        try:
                            if name in ("torch", "cpu") and is_torch_available():
                                if op_name == "matmul":
                                    return torch_matmul(a, b)
                                if op_name == "conv2d":
                                    return torch_conv2d(
                                        a,
                                        b,
                                        stride=kwargs.get("stride", 1),
                                        padding=kwargs.get("padding", 0),
                                    )
                                if op_name == "relu":
                                    return torch_relu(a)
                            if name == "triton" and is_triton_available():
                                if op_name == "matmul":
                                    return triton_matmul(a, b)
                                if op_name == "relu":
                                    return triton_relu(a)
                            if name == "opencl" and is_opencl_available():
                                if op_name == "matmul":
                                    return opencl_matmul(a, b)
                                if op_name == "conv2d":
                                    return opencl_conv2d(
                                        a,
                                        b,
                                        stride=kwargs.get("stride", 1),
                                        padding=kwargs.get("padding", 0),
                                    )
                                if op_name == "relu":
                                    return opencl_relu(a)
                            if name in ("numpy", "baseline"):
                                # Force baseline
                                return fn(*args, **kwargs)
                        except Exception:
                            return None
                        return None

                    for name in order:
                        res = _try_backend(name)
                        if res is not None:
                            # cache the chosen backend if it's a recognized one
                            chosen = name
                            if chosen in ("baseline", "numpy"):
                                chosen = "numpy"
                            self.cache.set(
                                cache_key,
                                {
                                    "backend": chosen,
                                    "hardware": self.hw.__dict__,
                                },
                            )
                            return res

                # 1) Torch accelerator preferred on macOS (MPS)
                try:
                    if (
                        platform.system() == "Darwin"
                        and is_torch_available()
                        and _torch_has_accelerator()
                    ):
                        if op_name == "matmul":
                            res = torch_matmul(a, b)
                            self.cache.set(
                                cache_key,
                                {
                                    "backend": "torch",
                                    "hardware": self.hw.__dict__,
                                },
                            )
                            return res
                        if op_name == "conv2d":
                            res = torch_conv2d(
                                a,
                                b,
                                stride=kwargs.get("stride", 1),
                                padding=kwargs.get("padding", 0),
                            )
                            self.cache.set(
                                cache_key,
                                {
                                    "backend": "torch",
                                    "hardware": self.hw.__dict__,
                                },
                            )
                            return res
                        if op_name == "relu":
                            res = torch_relu(a)
                            self.cache.set(
                                cache_key,
                                {
                                    "backend": "torch",
                                    "hardware": self.hw.__dict__,
                                },
                            )
                            return res
                except Exception:
                    pass

                # 2) Triton
                try:
                    if is_triton_available():
                        if op_name == "matmul":
                            res = triton_matmul(a, b)
                            self.cache.set(
                                cache_key,
                                {
                                    "backend": "triton",
                                    "hardware": self.hw.__dict__,
                                },
                            )
                            return res
                        if op_name == "relu":
                            res = triton_relu(a)
                            return res
                except Exception:
                    pass

                # 3) OpenCL
                try:
                    if is_opencl_available():
                        if op_name == "matmul":
                            res = opencl_matmul(a, b)
                            self.cache.set(
                                cache_key,
                                {
                                    "backend": "opencl",
                                    "hardware": self.hw.__dict__,
                                },
                            )
                            return res
                        if op_name == "conv2d":
                            res = opencl_conv2d(
                                a,
                                b,
                                stride=kwargs.get("stride", 1),
                                padding=kwargs.get("padding", 0),
                            )
                            self.cache.set(
                                cache_key,
                                {
                                    "backend": "opencl",
                                    "hardware": self.hw.__dict__,
                                },
                            )
                            return res
                        if op_name == "relu":
                            res = opencl_relu(a)
                            return res
                except Exception:
                    pass

                # 4) Torch fallback (CPU allowed)
                try:
                    if is_torch_available():
                        if op_name == "matmul":
                            res = torch_matmul(a, b)
                            self.cache.set(
                                cache_key,
                                {
                                    "backend": "torch",
                                    "hardware": self.hw.__dict__,
                                },
                            )
                            return res
                        if op_name == "conv2d":
                            res = torch_conv2d(
                                a,
                                b,
                                stride=kwargs.get("stride", 1),
                                padding=kwargs.get("padding", 0),
                            )
                            self.cache.set(
                                cache_key,
                                {
                                    "backend": "torch",
                                    "hardware": self.hw.__dict__,
                                },
                            )
                            return res
                        if op_name == "relu":
                            res = torch_relu(a)
                            self.cache.set(
                                cache_key,
                                {
                                    "backend": "torch",
                                    "hardware": self.hw.__dict__,
                                },
                            )
                            return res
                except Exception:
                    pass

                # 5) Baseline vs AI generation
                try:
                    t_base = benchmark_callable(
                        lambda: fn(*args, **kwargs),
                        runs=3,
                    )
                except Exception:
                    t_base = float("inf")

                t_ai = float("inf")
                ai_path = None
                ai_backend = None
                try:
                    target = "cuda" if self._pycuda_available else "python"
                    ai_path = self.codegen.generate(op_name, target=target)
                    if target == "cuda":
                        try:
                            arr0 = np.array(a)
                            arr1 = np.array(b) if b is not None else None
                            _ = self._run_cuda_source_via_pycuda(
                                ai_path, f"{op_name}_kernel", arr0, arr1
                            )
                            t_ai = benchmark_callable(
                                lambda: self._run_cuda_source_via_pycuda(
                                    ai_path, f"{op_name}_kernel", arr0, arr1
                                ),
                                runs=2,
                            )
                            ai_backend = "ai_cuda"
                        except Exception:
                            ai_path = None
                            t_ai = float("inf")
                    else:
                        try:
                            fn_name = f"generated_{op_name}"
                            arr0 = np.array(a)
                            arr1 = np.array(b) if b is not None else None
                            _ = run_generated_python(
                                str(ai_path),
                                fn_name,
                                arr0,
                                arr1,
                                timeout=sandbox_timeout,
                            )
                            t_ai = benchmark_callable(
                                lambda: run_generated_python(
                                    str(ai_path),
                                    fn_name,
                                    arr0,
                                    arr1,
                                    timeout=sandbox_timeout,
                                ),
                                runs=2,
                            )
                            ai_backend = "ai_python"
                        except Exception:
                            ai_path = None
                            t_ai = float("inf")
                except Exception:
                    ai_path = None
                    t_ai = float("inf")

                # Strict validation (optional)
                strict = os.environ.get("UHOP_STRICT_VALIDATE", "0")
                strict = strict not in ("0", "false", "False", None)
                is_valid = True
                if strict and ai_path:
                    try:
                        if op_name == "matmul" and b is not None:
                            a0 = np.array(a)
                            b0 = np.array(b)
                            spec = [(a0.shape, a0.dtype), (b0.shape, b0.dtype)]
                            if ai_backend == "ai_cuda":
                                runner = self._run_cuda_source_via_pycuda
                                is_valid = validate_kernel(
                                    lambda x, y: runner(
                                        ai_path,
                                        f"{op_name}_kernel",
                                        x,
                                        y,
                                    ),
                                    lambda x, y: fn(x, y),
                                    spec,
                                    nargs=2,
                                    tests=2,
                                )
                            elif ai_backend == "ai_python":
                                is_valid = validate_kernel(
                                    lambda x, y: run_generated_python(
                                        str(ai_path),
                                        f"generated_{op_name}",
                                        x,
                                        y,
                                        timeout=sandbox_timeout,
                                    ),
                                    lambda x, y: fn(x, y),
                                    spec,
                                    nargs=2,
                                    tests=2,
                                )
                    except Exception:
                        is_valid = False

                if t_ai < t_base and ai_path and is_valid:
                    if ai_backend == "ai_cuda":
                        self.cache.set(
                            cache_key,
                            {
                                "backend": "ai_cuda",
                                "path": str(ai_path),
                                "kernel_name": f"{op_name}_kernel",
                                "hardware": self.hw.__dict__,
                            },
                        )
                        arr0 = np.array(a)
                        arr1 = np.array(b) if b is not None else None
                        return self._run_cuda_source_via_pycuda(
                            ai_path, f"{op_name}_kernel", arr0, arr1
                        )
                    else:
                        self.cache.set(
                            cache_key,
                            {
                                "backend": "ai_python",
                                "path": str(ai_path),
                                "function": f"generated_{op_name}",
                                "hardware": self.hw.__dict__,
                            },
                        )
                        arr0 = np.array(a)
                        arr1 = np.array(b) if b is not None else None
                        return run_generated_python(
                            str(ai_path), f"generated_{op_name}", arr0, arr1,
                            timeout=sandbox_timeout,
                        )

                # 6) Baseline
                self.cache.set(
                    cache_key,
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
