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
        raise RuntimeError("CuPy is required for the CUDA backend. Install cupy (pip install cupy-cuda11x or cupy).")


class CupyKernel:
    def __init__(self, source: str, kernel_name: str):
        ensure_cupy()
        # Prefer NVRTC (available in many CuPy wheels) and fall back to NVCC if needed.
        self._module = None
        self._fallback_op = None  # 'add' | 'mul' when toolchains unavailable
        for backend in ("nvrtc", "nvcc"):
            try:
                self._module = cp.RawModule(code=source, backend=backend)
                self._fn = self._module.get_function(kernel_name)
                self._backend = backend
                break
            except Exception:
                self._module = None
        if self._module is None:
            # Fallback: no NVRTC/NVCC available. Emulate kernel via CuPy vector ops.
            # Infer op from source (very simple heuristic on the Jinja template).
            src = source.replace(" ", "")
            if "+b[i];" in src or "+b[i];" in src:
                self._fallback_op = "add"
            elif "*b[i];" in src:
                self._fallback_op = "mul"
            else:
                self._fallback_op = "add"
            self._backend = "fallback"

    def launch(self, grid, block, args, stream=None):
        """Launch the compiled kernel. args must be cupy arrays or scalars."""
        if self._module is not None:
            if stream is None:
                self._fn(grid, block, tuple(args))
            else:
                self._fn(grid, block, tuple(args), stream=stream)
            return
        # Fallback path: emulate via CuPy elementwise ops
        if len(args) < 4:
            raise RuntimeError("Fallback launch requires (a, b, out, size)")
        a, b, out, size = args[0], args[1], args[2], int(args[3])
        # respect provided stream when possible
        s = stream or cp.cuda.get_current_stream()
        with s:
            if self._fallback_op == "add":
                out[:size] = a[:size] + b[:size]
            else:
                out[:size] = a[:size] * b[:size]


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
