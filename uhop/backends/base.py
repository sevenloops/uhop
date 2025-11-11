"""
Base Backend Interface for UHOP Multi-Backend System.

Defines the abstract interface that all backend implementations must follow,
including kernel selection, vendor library integration, and fallback logic.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class KernelSource(Enum):
    """Source of a kernel implementation."""

    MANUAL = "manual"  # Hand-written optimized kernel
    VENDOR = "vendor"  # Vendor library (cuDNN, MIOpen, etc.)
    AUTOGEN = "autogen"  # Auto-generated (Triton, AI, etc.)
    FALLBACK = "fallback"  # Generic fallback implementation


@dataclass
class KernelInfo:
    """Information about a kernel implementation."""

    op_name: str
    backend: str
    source: KernelSource
    kernel_fn: Optional[Callable] = None
    vendor_lib: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BackendCapabilities:
    """
    Tracks capabilities and availability of a backend.
    """

    def __init__(self, name: str):
        self.name = name
        self.available = False
        self.vendor_libs: Dict[str, bool] = {}  # lib_name -> available
        self.device_count = 0
        self.device_names: List[str] = []
        self.compute_capability: Optional[str] = None
        self.memory_gb: float = 0.0
        self.error_msg: Optional[str] = None

    def __repr__(self) -> str:
        if not self.available:
            return f"<BackendCapabilities({self.name}, unavailable: {self.error_msg})>"
        return (
            f"<BackendCapabilities({self.name}, "
            f"devices={self.device_count}, "
            f"vendor_libs={list(self.vendor_libs.keys())})>"
        )


class Backend(ABC):
    """
    Abstract base class for all UHOP backend implementations.

    Each backend must implement:
    - Device/library detection
    - Kernel lookup with fallback logic
    - Kernel execution with proper error handling
    """

    def __init__(self, name: str):
        self.name = name
        self.capabilities = BackendCapabilities(name)
        self._manual_kernels: Dict[str, Callable] = {}
        self._vendor_kernels: Dict[str, Callable] = {}
        self._autogen_kernels: Dict[str, Callable] = {}
        self._initialized = False

    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the backend and detect capabilities.

        Returns:
            True if backend is available and usable, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def check_vendor_libs(self) -> Dict[str, bool]:
        """
        Check availability of vendor libraries (cuDNN, MIOpen, etc.).

        Returns:
            Dict mapping library name to availability status.
        """
        raise NotImplementedError

    @abstractmethod
    def get_supported_ops(self) -> List[str]:
        """
        Get list of operators supported by this backend.

        Returns:
            List of operator names (e.g., ['matmul', 'conv2d', 'relu']).
        """
        raise NotImplementedError

    def register_manual_kernel(self, op_name: str, kernel_fn: Callable):
        """Register a manually-written kernel for an operator."""
        self._manual_kernels[op_name] = kernel_fn
        logger.debug(f"[{self.name}] Registered manual kernel for '{op_name}'")

    def register_vendor_kernel(self, op_name: str, kernel_fn: Callable, vendor_lib: str):
        """Register a vendor library kernel."""
        self._vendor_kernels[op_name] = kernel_fn
        logger.debug(f"[{self.name}] Registered vendor kernel for '{op_name}' via {vendor_lib}")

    def register_autogen_kernel(self, op_name: str, kernel_fn: Callable):
        """Register an auto-generated kernel."""
        self._autogen_kernels[op_name] = kernel_fn
        logger.debug(f"[{self.name}] Registered autogen kernel for '{op_name}'")

    def get_kernel(self, op_name: str, prefer_source: Optional[KernelSource] = None) -> Optional[KernelInfo]:
        """
        Get best available kernel for an operator using fallback logic.

        Fallback order (unless prefer_source is specified):
        1. Manual kernel (hand-optimized)
        2. Vendor library kernel (cuDNN, MIOpen, etc.)
        3. Auto-generated kernel (Triton, AI)
        4. Generic fallback

        Args:
            op_name: Name of the operator
            prefer_source: Optional preference for kernel source

        Returns:
            KernelInfo if kernel is available, None otherwise
        """
        if not self._initialized:
            if not self.initialize():
                logger.warning(f"[{self.name}] Backend not initialized for '{op_name}'")
                return None

        # If specific source is preferred, try that first
        if prefer_source == KernelSource.MANUAL and op_name in self._manual_kernels:
            return KernelInfo(
                op_name=op_name,
                backend=self.name,
                source=KernelSource.MANUAL,
                kernel_fn=self._manual_kernels[op_name],
            )
        elif prefer_source == KernelSource.VENDOR and op_name in self._vendor_kernels:
            return KernelInfo(
                op_name=op_name,
                backend=self.name,
                source=KernelSource.VENDOR,
                kernel_fn=self._vendor_kernels[op_name],
            )
        elif prefer_source == KernelSource.AUTOGEN and op_name in self._autogen_kernels:
            return KernelInfo(
                op_name=op_name,
                backend=self.name,
                source=KernelSource.AUTOGEN,
                kernel_fn=self._autogen_kernels[op_name],
            )

        # Standard fallback order
        if op_name in self._manual_kernels:
            return KernelInfo(
                op_name=op_name,
                backend=self.name,
                source=KernelSource.MANUAL,
                kernel_fn=self._manual_kernels[op_name],
                metadata={"priority": "highest"},
            )

        if op_name in self._vendor_kernels:
            return KernelInfo(
                op_name=op_name,
                backend=self.name,
                source=KernelSource.VENDOR,
                kernel_fn=self._vendor_kernels[op_name],
                metadata={"priority": "high"},
            )

        if op_name in self._autogen_kernels:
            return KernelInfo(
                op_name=op_name,
                backend=self.name,
                source=KernelSource.AUTOGEN,
                kernel_fn=self._autogen_kernels[op_name],
                metadata={"priority": "medium"},
            )

        # No kernel available
        logger.debug(f"[{self.name}] No kernel available for '{op_name}'")
        return None

    def execute(self, op_name: str, *args, **kwargs) -> Any:
        """
        Execute an operator with automatic kernel selection.

        Args:
            op_name: Operator name
            *args: Positional arguments for the kernel
            **kwargs: Keyword arguments for the kernel

        Returns:
            Kernel execution result

        Raises:
            RuntimeError: If no kernel is available or execution fails
        """
        kernel_info = self.get_kernel(op_name)
        if kernel_info is None or kernel_info.kernel_fn is None:
            raise RuntimeError(f"No kernel available for operator '{op_name}' on backend '{self.name}'")

        try:
            result = kernel_info.kernel_fn(*args, **kwargs)
            logger.debug(f"[{self.name}] Executed '{op_name}' via {kernel_info.source.value}")
            return result
        except Exception as e:
            logger.error(f"[{self.name}] Failed to execute '{op_name}' " f"via {kernel_info.source.value}: {e}")
            raise

    def benchmark_kernel(
        self, op_name: str, *args, warmup: int = 3, iterations: int = 10, **kwargs
    ) -> Dict[str, float]:
        """
        Benchmark a kernel implementation.

        Args:
            op_name: Operator name
            *args: Arguments for the kernel
            warmup: Number of warmup iterations
            iterations: Number of benchmark iterations
            **kwargs: Keyword arguments for the kernel

        Returns:
            Dict with timing statistics (mean, median, min, max, std)
        """
        import statistics
        import time

        kernel_info = self.get_kernel(op_name)
        if kernel_info is None or kernel_info.kernel_fn is None:
            raise RuntimeError(f"No kernel for '{op_name}' on '{self.name}'")

        # Warmup
        for _ in range(warmup):
            _ = kernel_info.kernel_fn(*args, **kwargs)

        # Synchronize if needed (backend-specific)
        self._synchronize()

        # Benchmark
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            _ = kernel_info.kernel_fn(*args, **kwargs)
            self._synchronize()
            times.append(time.perf_counter() - start)

        return {
            "mean": statistics.mean(times),
            "median": statistics.median(times),
            "min": min(times),
            "max": max(times),
            "std": statistics.stdev(times) if len(times) > 1 else 0.0,
            "iterations": iterations,
            "source": kernel_info.source.value,
        }

    def _synchronize(self):
        """
        Synchronize device execution (backend-specific).
        Override in subclasses for GPU backends.
        """
        pass

    def get_device_info(self) -> Dict[str, Any]:
        """Get detailed device information."""
        return {
            "name": self.name,
            "available": self.capabilities.available,
            "device_count": self.capabilities.device_count,
            "device_names": self.capabilities.device_names,
            "compute_capability": self.capabilities.compute_capability,
            "memory_gb": self.capabilities.memory_gb,
            "vendor_libs": self.capabilities.vendor_libs,
        }

    def __repr__(self) -> str:
        status = "available" if self.capabilities.available else "unavailable"
        return f"<{self.__class__.__name__}({self.name}, {status})>"


class BackendManager:
    """
    Manages multiple backends and provides unified kernel selection.
    """

    def __init__(self):
        self._backends: Dict[str, Backend] = {}
        self._initialized = False

    def register_backend(self, backend: Backend):
        """Register a backend."""
        self._backends[backend.name] = backend
        logger.info(f"Registered backend: {backend.name}")

    def initialize_all(self):
        """Initialize all registered backends."""
        for name, backend in self._backends.items():
            try:
                success = backend.initialize()
                if success:
                    logger.info(f"Initialized backend: {name}")
                else:
                    logger.warning(f"Failed to initialize backend: {name}")
            except Exception as e:
                logger.error(f"Error initializing backend {name}: {e}")
        self._initialized = True

    def get_backend(self, name: str) -> Optional[Backend]:
        """Get a backend by name."""
        return self._backends.get(name)

    def list_available_backends(self) -> List[str]:
        """List all available (initialized) backends."""
        if not self._initialized:
            self.initialize_all()
        return [name for name, backend in self._backends.items() if backend.capabilities.available]

    def get_best_backend_for_op(
        self, op_name: str, preferred_backends: Optional[List[str]] = None
    ) -> Optional[Backend]:
        """
        Get the best available backend for an operator.

        Args:
            op_name: Operator name
            preferred_backends: Optional list of backends in priority order

        Returns:
            Best available backend, or None
        """
        if not self._initialized:
            self.initialize_all()

        # Use provided preference or query registry
        if preferred_backends is None:
            from uhop.core.op_registry import get_registry

            registry = get_registry()
            preferred_backends = registry.get_backends_for_op(op_name)

        # Try backends in order of preference
        for backend_name in preferred_backends:
            backend = self._backends.get(backend_name)
            if backend and backend.capabilities.available:
                kernel = backend.get_kernel(op_name)
                if kernel is not None:
                    return backend

        return None

    def get_coverage_summary(self) -> Dict[str, Any]:
        """Get summary of operator coverage across backends."""
        summary = {
            "total_backends": len(self._backends),
            "available_backends": len(self.list_available_backends()),
            "backend_details": {},
        }

        for name, backend in self._backends.items():
            if backend.capabilities.available:
                supported = backend.get_supported_ops()
                summary["backend_details"][name] = {
                    "supported_ops": len(supported),
                    "vendor_libs": backend.capabilities.vendor_libs,
                }

        return summary


# Global backend manager instance
_global_manager = None


def get_backend_manager() -> BackendManager:
    """Get the global backend manager."""
    global _global_manager
    if _global_manager is None:
        _global_manager = BackendManager()
    return _global_manager
