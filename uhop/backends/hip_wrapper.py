"""
HIP wrapper: minimal support via CuPy (ROCm-enabled). If CuPy with ROCm is present,
we can reuse its RawModule/launcher similar to the CUDA wrapper.
If not present, this wrapper will raise a clear error.
"""

import time

try:
    import cupy as cp  # type: ignore
except Exception:
    cp = None


def ensure_hip():
    if cp is None:
        raise RuntimeError(
            "HIP support requires cupy built for ROCm (cupy-rocm). Install a ROCm-enabled cupy."
        )


class HipKernel:
    def __init__(self, source: str, kernel_name: str):
        # Assuming cupy-rocm's RawModule can compile HIP code similarly
        ensure_hip()
        # RawModule may accept hipcc; rely on cupy for compilation
        self._module = cp.RawModule(
            code=source, backend="nvcc"
        )  # cupy ROCm may accept this; if not, adjust
        self._fn = self._module.get_function(kernel_name)

    def launch(self, grid, block, args, stream=None):
        ensure_hip()
        if stream is None:
            self._fn(grid, block, tuple(args))
        else:
            self._fn(grid, block, tuple(args), stream=stream)


def time_kernel_run(kernel: HipKernel, grid, block, args, warmups=3, runs=6):
    ensure_hip()
    stream = cp.cuda.get_current_stream()
    for _ in range(warmups):
        kernel.launch(grid, block, args, stream=stream)
    stream.synchronize()
    start = time.perf_counter()
    for _ in range(runs):
        kernel.launch(grid, block, args, stream=stream)
    stream.synchronize()
    end = time.perf_counter()
    return (end - start) / runs
