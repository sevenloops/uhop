# uhop/optimizer.py
"""
High-level developer-facing optimizer and decorator.
It tries hardware backends first (CUDA via PyCUDA), then attempts AI-generated kernels,
benchmarks, caches the winner, and uses a sandbox when executing AI-generated Python kernels.

Usage:
    from uhop import UHopOptimizer, optimize
    hop = UHopOptimizer()
    @hop.optimize("matmul")
    def matmul_np(A, B):
        return A @ B
"""

import os
import time
import numpy as np
from functools import wraps
from pathlib import Path

from .hardware import detect_hardware
from .cache import UhopCache
from .core.benchmark import benchmark_callable
from .core.executor import CudaExecutor, CpuExecutor
from .core.compiler import CUDACompiler
from .ai_codegen import generator as ai_gen
from .sandbox import run_generated_function

CACHE = UhopCache()

class UHopOptimizer:
    def __init__(self, mode: str = "auto", cache_dir: str = None):
        self.mode = mode
        self.hw = detect_hardware()
        self.cache = CACHE

    def compile_and_run_cuda_source(self, cuda_source: str, kernel_name: str, A, B):
        # compile via pycuda SourceModule (which compiles to PTX under the hood)
        from pycuda.compiler import SourceModule
        mod = SourceModule(cuda_source)
        execr = CudaExecutor(module_obj=mod)
        return execr.run_matmul(kernel_name, A, B)

    def try_cuda_kernel_file(self, cu_path: str, kernel_name: str, A, B):
        src = Path(cu_path).read_text()
        try:
            return self.compile_and_run_cuda_source(src, kernel_name, A, B)
        except Exception as e:
            raise

    def optimize(self, op_name: str, sandbox_timeout: int = 8):
        def decorator(fn):
            cache_key = op_name

            @wraps(fn)
            def wrapper(*args, **kwargs):
                if len(args) < 2:
                    raise ValueError("Expected (A, B, ...)")
                A, B = args[0], args[1]
                A_np = np.array(A, dtype=np.float32)
                B_np = np.array(B, dtype=np.float32)

                # 1) cached record?
                rec = self.cache.get(cache_key)
                if rec:
                    # if backend is cuda and a module file exists, prefer it
                    if rec.get("backend") == "cuda" and rec.get("path"):
                        try:
                            res = self.try_cuda_kernel_file(rec["path"], rec.get("kernel_name", "matmul_kernel"), A_np, B_np)
                            return res
                        except Exception:
                            # cache invalid -> continue
                            pass
                    if rec.get("backend") == "numpy":
                        return fn(A_np, B_np)

                # 2) try native CUDA backend if available
                try:
                    import pycuda
                    # run built-in CUDA kernel shipped with project
                    built_in = Path(__file__).resolve().parent / "kernels" / "cuda" / f"{op_name}.cu"
                    if built_in.exists():
                        try:
                            res = self.try_cuda_kernel_file(str(built_in), f"{op_name}_kernel", A_np, B_np)
                            self.cache.set(cache_key, {"backend":"cuda", "path": str(built_in), "kernel_name": f"{op_name}_kernel", "hardware": self.hw.__dict__})
                            return res
                        except Exception:
                            pass
                except Exception:
                    # pycuda not installed or no cuda device
                    pass

                # 3) benchmark numpy baseline
                try:
                    t_base = benchmark_callable(lambda: fn(A_np, B_np), runs=3)
                except Exception:
                    t_base = float("inf")

                # 4) try AI-generated kernel (CUDA C) using AICodegen
                ai_path = None
                try:
                    codegen = ai_gen.AICodegen()
                    prompt = self._prompt_for(op_name)
                    ai_path = codegen.generate(prompt, out_name=f"ai_{op_name}.cu")
                    # verify by compiling & running a single trial
                    res = self.try_cuda_kernel_file(str(ai_path), f"{op_name}_kernel", A_np, B_np)
                    t_ai = benchmark_callable(lambda: self.try_cuda_kernel_file(str(ai_path), f"{op_name}_kernel", A_np, B_np), runs=2)
                except Exception:
                    ai_path = None
                    t_ai = float("inf")

                winner = "numpy"
                if t_ai < t_base and ai_path:
                    winner = "ai"

                if winner == "ai":
                    self.cache.set(cache_key, {"backend":"cuda", "path": str(ai_path), "kernel_name": f"{op_name}_kernel", "hardware": self.hw.__dict__})
                    return self.try_cuda_kernel_file(str(ai_path), f"{op_name}_kernel", A_np, B_np)
                else:
                    # fallback to numpy baseline
                    self.cache.set(cache_key, {"backend":"numpy", "path": None, "kernel_name": fn.__name__, "hardware": self.hw.__dict__})
                    return fn(A_np, B_np)
            return wrapper
        return decorator

    def _prompt_for(self, op_name: str) -> str:
        from .ai_codegen import prompt_templates
        if op_name == "matmul":
            return prompt_templates.MATMUL_CUDA_PROMPT
        if op_name == "relu":
            return prompt_templates.RELU_CUDA_PROMPT
        if op_name == "conv2d":
            return prompt_templates.CONV2D_CUDA_PROMPT
        return prompt_templates.MATMUL_CUDA_PROMPT

# convenience
_GLOBAL_OPT = UHopOptimizer()
def optimize(op_name: str):
    return _GLOBAL_OPT.optimize(op_name)
