# uhop/core/compiler.py
"""
Compilation utilities.

We provide runtime compilation using PyCUDA (SourceModule) and a helper to call nvcc (optional).
If PyCUDA is available, SourceModule compiles the CUDA C source in-memory.
"""

import os
import subprocess
from pathlib import Path
from typing import Optional, Dict
import hashlib

class NVCCCompiler:
    @staticmethod
    def compile_cu_to_ptx(source_path: str, arch: str = "sm_70", out_path: Optional[str] = None) -> str:
        src = Path(source_path)
        out = Path(out_path) if out_path else src.with_suffix(".ptx")
        cmd = [
            "nvcc",
            "-ptx",
            f"-arch={arch}",
            str(src),
            "-o",
            str(out)
        ]
        subprocess.run(cmd, check=True)
        return str(out)

class CUDACompiler:
    _registry: Dict[str, any] = {}

    @staticmethod
    def compile_via_pycuda(source_code: str):
        """
        Return a PyCUDA SourceModule compiled object if pycuda is available.
        Otherwise raise ImportError.
        """
        try:
            from pycuda.compiler import SourceModule
        except Exception as e:
            raise ImportError("pycuda not available") from e
        h = hashlib.sha1(source_code.encode("utf-8")).hexdigest()
        if h in CUDACompiler._registry:
            return CUDACompiler._registry[h]
        mod = SourceModule(source_code, no_extern_c=True)
        CUDACompiler._registry[h] = mod
        return mod
