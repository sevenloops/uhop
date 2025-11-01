# uhop/core/executor.py
"""
Executor wrappers:
- CudaExecutor uses PyCUDA to run kernels compiled from CUDA source.
- CpuExecutor runs Python/NumPy fallback.
"""

import numpy as np


class CudaExecutor:
    def __init__(self, source_code: str = None, module_obj=None):
        """
        Either provide compiled module_obj (pycuda SourceModule)
        or provide source_code (will compile via pycuda).
        """
        self.source_code = source_code
        self.module = module_obj
        if self.module is None and self.source_code is not None:
            from pycuda.compiler import SourceModule

            self.module = SourceModule(self.source_code)

    def run_matmul(
        self, func_name: str, A: np.ndarray, B: np.ndarray, block=(16, 16, 1)
    ):
        import numpy as _np
        import pycuda.driver as cuda

        if self.module is None:
            raise RuntimeError("CUDA module not compiled")
        func = self.module.get_function(func_name)
        A = _np.array(A).astype(_np.float32)
        B = _np.array(B).astype(_np.float32)
        N, M = A.shape
        M2, K = B.shape
        assert M == M2
        C = _np.empty((N, K), dtype=_np.float32)
        A_gpu = cuda.mem_alloc(A.nbytes)
        B_gpu = cuda.mem_alloc(B.nbytes)
        C_gpu = cuda.mem_alloc(C.nbytes)
        cuda.memcpy_htod(A_gpu, A)
        cuda.memcpy_htod(B_gpu, B)
        # grid dims
        grid = ((K + block[0] - 1) // block[0], (N + block[1] - 1) // block[1])
        func(
            A_gpu,
            B_gpu,
            C_gpu,
            _np.int32(N),
            _np.int32(M),
            _np.int32(K),
            block=block,
            grid=grid,
        )
        cuda.memcpy_dtoh(C, C_gpu)
        return C


class CpuExecutor:
    @staticmethod
    def matmul(A, B):
        return np.array(A) @ np.array(B)


# Removed legacy kernel_cache and stubbed device persistence code (not used in tests/examples)
