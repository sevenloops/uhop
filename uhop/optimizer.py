# uhop/optimizer.py
"""
UHOP optimizer: multi-backend decision flow.

Priority:
  1) Torch backend (CUDA / ROCm-built PyTorch / MPS) via torch_backend
  2) Triton backend (triton)
  3) OpenCL backend (pyopencl)
  4) AI-generated kernels (CUDA via pycuda if available, else Python via sandbox)
  5) NumPy baseline (fallback)

Caches chosen backend and associated metadata in UhopCache.
"""
import importlib.util
import os
from functools import wraps
from pathlib import Path
from typing import Callable, Any, Optional
import numpy as np

from .hardware import detect_hardware, HardwareProfile
from .cache import UhopCache
from .core.benchmark import benchmark_callable
from .sandbox import run_generated_python
from .ai_codegen.generator import AICodegen

# backends
from .backends import (
    is_torch_available, torch_matmul, torch_conv2d, torch_relu,
    is_triton_available, triton_matmul, triton_conv2d, triton_relu,
    is_opencl_available, opencl_matmul, opencl_conv2d, opencl_conv2d_relu, opencl_relu,
)
from .backends.torch_backend import torch_has_accelerator

CACHE = UhopCache()

class UHopOptimizer:
    def __init__(self):
        self.hw = detect_hardware()
        self.cache = CACHE
        self.codegen = AICodegen()
        # detect optional runtimes
        try:
            import pycuda  # type: ignore  # noqa: F401
            self._pycuda_available = True
        except Exception:
            self._pycuda_available = False
        # cache for compiled AI OpenCL programs
        self._ai_opencl_cache = {}

    # Helper to import python module by path and call function
    def _import_and_call(self, path: str, fn_name: str, *args):
        spec = importlib.util.spec_from_file_location("uhop_generated", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore
        fn = getattr(mod, fn_name)
        return fn(*args)

    # Helper to compile-run CUDA source using PyCUDA SourceModule (if available)
    def _run_cuda_source_via_pycuda(self, source_path: Path, kernel_name: str, a: np.ndarray, b: Optional[np.ndarray] = None):
        if not self._pycuda_available:
            raise RuntimeError("pycuda not available")
        from pycuda.compiler import SourceModule  # type: ignore
        import pycuda.driver as cuda  # type: ignore
        import numpy as _np
        src = source_path.read_text()
        mod = SourceModule(src)
        func = mod.get_function(kernel_name)
        # For matmul kernel signature we follow convention (A, B, C, N, M, K)
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
        func(A_gpu, B_gpu, C_gpu, np.int32(N), np.int32(M), np.int32(K), block=block, grid=grid)
        cuda.memcpy_dtoh(C, C_gpu)
        return C

    def _run_opencl_generated_matmul(self, source_path: Path, kernel_name: str, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compile and run an AI-generated OpenCL matmul kernel with signature:
        __kernel void generated_matmul(const int M, const int N, const int K,
                                       __global const float* A,
                                       __global const float* B,
                                       __global float* C);
        """
        import numpy as _np
        try:
            import pyopencl as cl  # type: ignore
        except Exception as e:
            raise RuntimeError(f"pyopencl not available for ai_opencl backend: {e}")
        a = _np.array(a, dtype=_np.float32)
        b = _np.array(b, dtype=_np.float32)
        M, K = a.shape
        K2, N = b.shape
        assert K == K2
        C = _np.empty((M, N), dtype=_np.float32)
        # compile/cache
        cache_key = (str(source_path), kernel_name)
        entry = self._ai_opencl_cache.get(cache_key)
        if entry is None:
            src = Path(source_path).read_text()
            # Prefer a GPU device
            plats = cl.get_platforms()
            ctx = None
            for p in plats:
                gpus = [d for d in p.get_devices() if d.type & cl.device_type.GPU]
                if gpus:
                    ctx = cl.Context(devices=[gpus[0]])
                    break
            if ctx is None:
                ctx = cl.create_some_context(interactive=False)
            q = cl.CommandQueue(ctx)
            prg = cl.Program(ctx, src).build()
            kn = cl.Kernel(prg, kernel_name)
            entry = (ctx, q, prg, kn)
            self._ai_opencl_cache[cache_key] = entry
        ctx, q, prg, kn = entry
        mf = cl.mem_flags
        a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
        b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
        c_buf = cl.Buffer(ctx, mf.WRITE_ONLY, C.nbytes)
        kn.set_args(_np.int32(M), _np.int32(N), _np.int32(K), a_buf, b_buf, c_buf)
        gsz = (int(M), int(N))
        cl.enqueue_nd_range_kernel(q, kn, gsz, None)
        cl.enqueue_copy(q, C, c_buf)
        q.finish()
        return C

    def optimize(self, op_name: str, sandbox_timeout: int = 8):
        """
        Decorator that optimizes a fallback Python function `fn(a,b)` for op_name.
        """
        def decorator(fn: Callable):
            cache_key = op_name

            @wraps(fn)
            def wrapper(*args, **kwargs):
                # Validate args
                if len(args) < 1:
                    raise ValueError("Expected at least one positional arg (e.g., a, optionally b).")
                # Make numpy copies for consistent benchmarks
                a = np.array(args[0])
                b = np.array(args[1]) if len(args) > 1 else None

                # 0) If cached & backend recorded, try to use cache first
                rec = self.cache.get(cache_key)
                if rec:
                    backend = rec.get("backend")
                    try:
                        if backend == "torch":
                            # Use torch wrapper; for conv2d we need weight argument
                            if op_name == "matmul":
                                return torch_matmul(a, b)
                            if op_name == "conv2d":
                                return torch_conv2d(a, b, stride=kwargs.get("stride",1), padding=kwargs.get("padding",0))
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
                                return opencl_conv2d(a, b, stride=kwargs.get("stride",1), padding=kwargs.get("padding",0))
                            if op_name == "relu":
                                return opencl_relu(a)
                        if backend == "ai_cuda":
                            # path available
                            path = rec.get("path")
                            kernel_name = rec.get("kernel_name", f"{op_name}_kernel")
                            return self._run_cuda_source_via_pycuda(Path(path), kernel_name, a, b)
                        if backend == "ai_opencl":
                            path = rec.get("path")
                            kernel_name = rec.get("kernel_name", f"generated_{op_name}")
                            if op_name == "matmul" and path:
                                return self._run_opencl_generated_matmul(Path(path), kernel_name, a, b)
                            # other ops not yet supported via ai_opencl cached path; fall through
                        if backend == "ai_python":
                            path = rec.get("path")
                            fn_name = rec.get("function", f"generated_{op_name}")
                            return run_generated_python(path, fn_name, a, b, timeout=sandbox_timeout)
                    except Exception:
                        # if cached backend failed, fall through to detection/regeneration
                        pass

                # 1) Try GPU-accelerated backends first (prefer GPU over CPU)
                # 1a) Triton (typically NVIDIA GPUs)
                try:
                    if is_triton_available():
                        if op_name == "matmul":
                            res = triton_matmul(a, b)
                            self.cache.set(cache_key, {"backend":"triton", "hardware": self.hw.__dict__})
                            return res
                        if op_name == "relu":
                            res = triton_relu(a)
                            return res
                except Exception:
                    pass

                # 1b) OpenCL (AMD/Intel GPUs)
                try:
                    if is_opencl_available():
                        if op_name == "matmul":
                            res = opencl_matmul(a, b)
                            self.cache.set(cache_key, {"backend":"opencl", "hardware": self.hw.__dict__})
                            return res
                        if op_name == "conv2d":
                            res = opencl_conv2d(a, b, stride=kwargs.get("stride",1), padding=kwargs.get("padding",0))
                            self.cache.set(cache_key, {"backend":"opencl", "hardware": self.hw.__dict__})
                            return res
                        if op_name == "conv2d_relu":
                            res = opencl_conv2d_relu(a, b, stride=kwargs.get("stride",1), padding=kwargs.get("padding",0))
                            self.cache.set(cache_key, {"backend":"opencl", "hardware": self.hw.__dict__})
                            return res
                        if op_name == "relu":
                            res = opencl_relu(a)
                            return res
                except Exception:
                    pass

                # 1c) Torch GPU/MPS
                try:
                    if is_torch_available() and torch_has_accelerator():
                        if op_name == "matmul":
                            res = torch_matmul(a, b)
                            self.cache.set(cache_key, {"backend":"torch", "hardware": self.hw.__dict__})
                            return res
                        if op_name == "conv2d":
                            res = torch_conv2d(a, b, stride=kwargs.get("stride",1), padding=kwargs.get("padding",0))
                            self.cache.set(cache_key, {"backend":"torch", "hardware": self.hw.__dict__})
                            return res
                        if op_name == "relu":
                            res = torch_relu(a)
                            self.cache.set(cache_key, {"backend":"torch", "hardware": self.hw.__dict__})
                            return res
                except Exception:
                    pass

                # 2) Torch CPU fallback if no accelerator or GPU backend succeeded
                try:
                    if is_torch_available():
                        if op_name == "matmul":
                            res = torch_matmul(a, b)
                            self.cache.set(cache_key, {"backend":"torch", "hardware": self.hw.__dict__})
                            return res
                        if op_name == "conv2d":
                            res = torch_conv2d(a, b, stride=kwargs.get("stride",1), padding=kwargs.get("padding",0))
                            self.cache.set(cache_key, {"backend":"torch", "hardware": self.hw.__dict__})
                            return res
                        if op_name == "relu":
                            res = torch_relu(a)
                            self.cache.set(cache_key, {"backend":"torch", "hardware": self.hw.__dict__})
                            return res
                except Exception:
                    pass

                # 4) Benchmark NumPy baseline
                try:
                    t_base = benchmark_callable(lambda: fn(*args, **kwargs), args=(), runs=3)
                except Exception:
                    t_base = float("inf")

                # 5) Try AI generation
                t_ai = float("inf")
                ai_path = None
                ai_backend = None
                try:
                    # decide target: prefer CUDA if pycuda available, else python sandbox
                    target = "cuda" if self._pycuda_available else "python"
                    ai_path = self.codegen.generate(op_name, target=target)
                    # run once to validate
                    if target == "cuda":
                        try:
                            out = self._run_cuda_source_via_pycuda(ai_path, f"{op_name}_kernel", a, b)
                            # benchmark ai (quick runs)
                            t_ai = benchmark_callable(lambda: self._run_cuda_source_via_pycuda(ai_path, f"{op_name}_kernel", a, b), runs=2)
                            ai_backend = "ai_cuda"
                        except Exception:
                            ai_path = None
                            t_ai = float("inf")
                    else:
                        # python sandbox
                        try:
                            fn_name = f"generated_{op_name}"
                            out = run_generated_python(str(ai_path), fn_name, a, b, timeout=sandbox_timeout)
                            t_ai = benchmark_callable(lambda: run_generated_python(str(ai_path), fn_name, a, b, timeout=sandbox_timeout), runs=2)
                            ai_backend = "ai_python"
                        except Exception:
                            ai_path = None
                            t_ai = float("inf")
                except Exception:
                    ai_path = None
                    t_ai = float("inf")

                # 6) Choose winner
                if t_ai < t_base and ai_path:
                    # cache and return ai result
                    if ai_backend == "ai_cuda":
                        self.cache.set(cache_key, {"backend":"ai_cuda", "path": str(ai_path), "kernel_name": f"{op_name}_kernel", "hardware": self.hw.__dict__})
                        return self._run_cuda_source_via_pycuda(ai_path, f"{op_name}_kernel", a, b)
                    else:
                        self.cache.set(cache_key, {"backend":"ai_python", "path": str(ai_path), "function": f"generated_{op_name}", "hardware": self.hw.__dict__})
                        return run_generated_python(str(ai_path), f"generated_{op_name}", a, b, timeout=sandbox_timeout)
                else:
                    # fallback to baseline
                    self.cache.set(cache_key, {"backend":"numpy", "path": None, "function": fn.__name__, "hardware": self.hw.__dict__})
                    return fn(*args, **kwargs)

            return wrapper
        return decorator

# convenience global optimizer for decorators
_GLOBAL_OPT = UHopOptimizer()

def optimize(op_name: str):
    return _GLOBAL_OPT.optimize(op_name)
