"""
Kernel compilation and execution backend for AI-generated kernels.

Provides compilation, validation, and profiling for generated kernels
across multiple backends (OpenCL, CUDA, etc.).
"""

import os  # noqa: F401
import time
from pathlib import Path  # noqa: F401
from typing import Any, Callable, Dict, List, Optional, Tuple  # noqa: F401
import numpy as np

from ..validation import validate_callable, gen_cases
from ..hardware import detect_hardware


try:
    import pyopencl as cl  # type: ignore
except ImportError:
    cl = None

try:
    import cupy as cp  # type: ignore
except ImportError:
    cp = None


class KernelCompiler:
    """Compiles and executes AI-generated kernels."""

    def __init__(self, backend: Optional[str] = None):
        """
        Initialize kernel compiler.

        Parameters
        ----------
        backend: str
            Backend to use (opencl, cuda). Auto-detected if None.
        """
        self.backend = backend or self._detect_backend()
        self.context = None
        self.queue = None
        self._initialize_backend()

    def _detect_backend(self) -> str:
        """Detect available backend."""
        hardware = detect_hardware()

        if hardware.kind == "cuda" and cp is not None:
            return "cuda"
        elif hardware.kind.startswith("opencl") and cl is not None:
            return "opencl"
        else:
            return "opencl"  # Default fallback

    def _initialize_backend(self):
        """Initialize the selected backend."""
        if self.backend == "opencl" and cl is not None:
            try:
                platform = cl.get_platforms()[0]
                devices = platform.get_devices(device_type=cl.device_type.GPU)
                if not devices:
                    devices = platform.get_devices(device_type=cl.device_type.CPU)
                self.context = cl.Context(devices)
                self.queue = cl.CommandQueue(self.context, properties=cl.command_queue_properties.PROFILING_ENABLE)
            except Exception as e:
                print(f"OpenCL initialization failed: {e}")
                self.context = None
                self.queue = None

    def compile_opencl_kernel(self, kernel_code: str, kernel_name: str) -> Optional[Any]:
        """Compile OpenCL kernel."""
        if self.context is None or self.queue is None:
            return None

        try:
            program = cl.Program(self.context, kernel_code).build()
            kernel = getattr(program, kernel_name)
            return kernel
        except Exception as e:
            print(f"OpenCL compilation failed: {e}")
            return None

    def compile_cuda_kernel(self, kernel_code: str, kernel_name: str) -> Optional[Any]:
        """Compile CUDA kernel."""
        if cp is None:
            return None

        try:
            # For CUDA, we'll use CuPy's RawModule
            module = cp.RawModule(code=kernel_code)
            kernel = module.get_function(kernel_name)
            return kernel
        except Exception as e:
            print(f"CUDA compilation failed: {e}")
            return None

    def compile_kernel(self, kernel_code: str, kernel_name: str) -> Optional[Any]:
        """Compile kernel for the current backend."""
        if self.backend == "opencl":
            return self.compile_opencl_kernel(kernel_code, kernel_name)
        elif self.backend == "cuda":
            return self.compile_cuda_kernel(kernel_code, kernel_name)
        else:
            return None

    def execute_matmul_opencl(
        self, kernel: Any, A: np.ndarray, B: np.ndarray
    ) -> Tuple[Optional[np.ndarray], float, Dict[str, Any]]:
        """Execute matmul kernel on OpenCL."""
        if self.queue is None:
            return None, float("inf"), {}

        try:
            m, k = A.shape
            k2, n = B.shape

            if k != k2:
                raise ValueError("Matrix dimensions don't match for multiplication")

            # Create buffers
            mf = cl.mem_flags
            A_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A.astype(np.float32))
            B_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B.astype(np.float32))
            C_buf = cl.Buffer(self.context, mf.WRITE_ONLY, m * n * 4)  # 4 bytes per float

            # Set kernel arguments
            kernel.set_arg(0, np.int32(m))
            kernel.set_arg(1, np.int32(n))
            kernel.set_arg(2, np.int32(k))
            kernel.set_arg(3, A_buf)
            kernel.set_arg(4, B_buf)
            kernel.set_arg(5, C_buf)

            # Execute kernel
            global_size = (n, m)
            local_size = None

            # Warmup
            cl.enqueue_nd_range_kernel(self.queue, kernel, global_size, local_size)
            self.queue.finish()

            # Timed execution
            start_time = time.perf_counter()
            event = cl.enqueue_nd_range_kernel(self.queue, kernel, global_size, local_size)
            event.wait()
            end_time = time.perf_counter()

            execution_time = end_time - start_time

            # Read result
            C = np.empty((m, n), dtype=np.float32)
            cl.enqueue_copy(self.queue, C, C_buf)

            # Calculate performance metrics
            flops = 2 * m * n * k  # 2 operations per element
            gflops = flops / (execution_time * 1e9)

            bandwidth = (A.nbytes + B.nbytes + C.nbytes) / (execution_time * 1e9)  # GB/s

            metrics = {
                "execution_time_ms": execution_time * 1000,
                "gflops": gflops,
                "bandwidth_gbs": bandwidth,
                "backend": "opencl",
            }

            return C, execution_time, metrics

        except Exception as e:
            print(f"OpenCL execution failed: {e}")
            return None, float("inf"), {}

    def execute_matmul_cuda(
        self, kernel: Any, A: np.ndarray, B: np.ndarray
    ) -> Tuple[Optional[np.ndarray], float, Dict[str, Any]]:
        """Execute matmul kernel on CUDA."""
        if cp is None:
            return None, float("inf"), {}

        try:
            m, k = A.shape
            k2, n = B.shape

            if k != k2:
                raise ValueError("Matrix dimensions don't match for multiplication")

            # Copy to GPU
            A_gpu = cp.asarray(A.astype(np.float32))
            B_gpu = cp.asarray(B.astype(np.float32))
            C_gpu = cp.zeros((m, n), dtype=cp.float32)

            # Set grid and block sizes
            block_size = (16, 16)
            grid_size = ((n + block_size[0] - 1) // block_size[0], (m + block_size[1] - 1) // block_size[1])

            # Warmup
            kernel(grid_size, block_size, (A_gpu, B_gpu, C_gpu, m, n, k))
            cp.cuda.get_current_stream().synchronize()

            # Timed execution
            start_time = time.perf_counter()
            kernel(grid_size, block_size, (A_gpu, B_gpu, C_gpu, m, n, k))
            cp.cuda.get_current_stream().synchronize()
            end_time = time.perf_counter()

            execution_time = end_time - start_time

            # Copy result back
            C = cp.asnumpy(C_gpu)

            # Calculate performance metrics
            flops = 2 * m * n * k
            gflops = flops / (execution_time * 1e9)
            bandwidth = (A.nbytes + B.nbytes + C.nbytes) / (execution_time * 1e9)

            metrics = {
                "execution_time_ms": execution_time * 1000,
                "gflops": gflops,
                "bandwidth_gbs": bandwidth,
                "backend": "cuda",
            }

            return C, execution_time, metrics

        except Exception as e:
            print(f"CUDA execution failed: {e}")
            return None, float("inf"), {}

    def execute_matmul(
        self, kernel: Any, kernel_name: str, A: np.ndarray, B: np.ndarray
    ) -> Tuple[Optional[np.ndarray], float, Dict[str, Any]]:
        """Execute matmul kernel."""
        if self.backend == "opencl":
            return self.execute_matmul_opencl(kernel, A, B)
        elif self.backend == "cuda":
            return self.execute_matmul_cuda(kernel, A, B)
        else:
            return None, float("inf"), {}

    def validate_matmul_kernel(
        self, kernel: Any, kernel_name: str, input_shapes: List[Tuple[int, ...]]
    ) -> Tuple[bool, float, float, Dict[str, Any]]:
        """Validate matmul kernel against reference implementation."""

        def reference_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
            return A @ B

        def kernel_runner(A: np.ndarray, B: np.ndarray) -> np.ndarray:
            result, _, _ = self.execute_matmul(kernel, kernel_name, A, B)
            if result is None:
                raise RuntimeError("Kernel execution failed")
            return result

        # Generate test cases
        input_specs = [{"shape": input_shapes[0], "dtype": np.float32}, {"shape": input_shapes[1], "dtype": np.float32}]

        test_cases = gen_cases(input_specs, extra_edge_cases=True)

        # Run validation
        validation_result = validate_callable(
            candidate=kernel_runner, reference=reference_matmul, input_specs=input_specs, cases=test_cases, strict=True
        )

        # Performance profiling
        perf_metrics = {}
        if validation_result.ok:
            # Use larger matrices for performance measurement
            large_A = np.random.randn(256, 256).astype(np.float32)
            large_B = np.random.randn(256, 256).astype(np.float32)

            _, execution_time, metrics = self.execute_matmul(kernel, kernel_name, large_A, large_B)
            if execution_time < float("inf"):
                perf_metrics = metrics

        return validation_result.ok, validation_result.max_abs_err, validation_result.max_rel_err, perf_metrics


class KernelValidator:
    """High-level kernel validation and profiling."""

    def __init__(self, compiler: Optional[KernelCompiler] = None):
        self.compiler = compiler or KernelCompiler()

    def validate_and_profile_kernel(
        self, kernel_code: str, kernel_name: str, operation: str, input_shapes: List[Tuple[int, ...]]
    ) -> Dict[str, Any]:
        """
        Validate and profile a generated kernel.

        Returns comprehensive validation and profiling results.
        """
        result = {
            "compile_success": False,
            "validation_success": False,
            "execution_time_ms": float("inf"),
            "gflops": 0.0,
            "bandwidth_gbs": 0.0,
            "max_abs_error": float("inf"),
            "max_rel_error": float("inf"),
            "backend": self.compiler.backend,
            "errors": [],
        }

        # Compile kernel
        kernel = self.compiler.compile_kernel(kernel_code, kernel_name)
        if kernel is None:
            result["errors"].append("Compilation failed")
            return result

        result["compile_success"] = True

        # Validate kernel
        if operation == "matmul":
            try:
                valid, max_abs_err, max_rel_err, perf_metrics = self.compiler.validate_matmul_kernel(
                    kernel, kernel_name, input_shapes
                )

                result["validation_success"] = valid
                result["max_abs_error"] = max_abs_err
                result["max_rel_error"] = max_rel_err

                if perf_metrics:
                    result.update(perf_metrics)

            except Exception as e:
                result["errors"].append(f"Validation failed: {e}")
        else:
            result["errors"].append(f"Unsupported operation: {operation}")

        return result
