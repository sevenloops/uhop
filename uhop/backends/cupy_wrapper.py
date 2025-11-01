"""
Minimal CuPy wrapper for compiling & launching RawModule CUDA kernels.
This wrapper is intentionally small: it compiles a CUDA kernel string and
exposes a simple `build_and_launch` helper used by the autotuner.
"""

import time

try:
    import cupy as cp  # type: ignore
except Exception:
    cp = None  # defensive; callers should check


def ensure_cupy():
    if cp is None:
        raise RuntimeError(
            "CuPy is required for the CUDA backend. Install cupy (pip install cupy-cuda11x or cupy)."
        )


class CupyKernel:
    def __init__(self, source: str, kernel_name: str):
        ensure_cupy()
        # RawModule accepts .cu source and compiles with NVCC (by default)
        self._module = cp.RawModule(code=source, backend="nvcc")
        self._fn = self._module.get_function(kernel_name)

    def launch(self, grid, block, args, stream=None):
        """Launch the compiled kernel. args must be cupy arrays or scalars."""
        if stream is None:
            self._fn(grid, block, tuple(args))
        else:
            self._fn(grid, block, tuple(args), stream=stream)


def time_kernel_run(kernel: CupyKernel, grid, block, args, warmups=3, runs=10):
    """Time kernel execution using cupy.cuda.get_current_stream() synchronization."""
    ensure_cupy()
    stream = cp.cuda.get_current_stream()
    # Warmup
    for _ in range(warmups):
        kernel.launch(grid, block, args, stream=stream)
    # timed runs
    stream.synchronize()
    start = time.perf_counter()
    for _ in range(runs):
        kernel.launch(grid, block, args, stream=stream)
    stream.synchronize()
    end = time.perf_counter()
    return (end - start) / runs


def arrays_to_device(*arrays, dtype=None):
    ensure_cupy()
    dev = []
    for a in arrays:
        if isinstance(a, cp.ndarray):
            dev.append(a)
        else:
            dev.append(cp.array(a, dtype=dtype))
    return dev


def device_name():
    ensure_cupy()
    dev = cp.cuda.runtime.getDevice()
    name = cp.cuda.runtime.getDeviceProperties(dev).get("name", "cuda")
    return name
