"""
Unified Operator Registry for UHOP Multi-Backend System.

Tracks all supported deep learning operators, their categories,
backend support, and priority ordering for optimal kernel selection.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class OperatorCategory(Enum):
    """Categories of deep learning operators."""

    LINEAR_ALGEBRA = "linear_algebra"
    CONVOLUTION = "convolution"
    ATTENTION = "attention"
    NORMALIZATION = "normalization"
    ACTIVATION = "activation"
    POOLING = "pooling"
    SOFTMAX = "softmax"
    RNN = "rnn"
    ELEMENTWISE = "elementwise"
    REDUCTION = "reduction"
    FUSED = "fused"


@dataclass
class OperatorSpec:
    """Specification for a single operator."""

    name: str
    category: OperatorCategory
    backends: List[str]  # Ordered by priority
    vendor_libs: Dict[str, str] = field(default_factory=dict)  # backend -> lib name
    importance: str = "medium"  # low, medium, high, critical
    description: str = ""
    manual_kernel_available: bool = False
    autogen_supported: bool = True

    def supports_backend(self, backend: str) -> bool:
        """Check if operator supports a given backend."""
        return backend in self.backends

    def get_vendor_lib(self, backend: str) -> Optional[str]:
        """Get vendor library name for backend (e.g., 'cudnn' for cuda)."""
        return self.vendor_libs.get(backend)


class OperatorRegistry:
    """
    Global registry of all supported operators.

    Provides lookup, filtering, and backend matching capabilities.
    """

    def __init__(self):
        self._registry: Dict[str, OperatorSpec] = {}
        self._initialize_operators()

    def _initialize_operators(self):
        """Initialize the operator registry with all supported ops."""

        # Linear Algebra Operations
        self.register(
            OperatorSpec(
                name="matmul",
                category=OperatorCategory.LINEAR_ALGEBRA,
                backends=[
                    "cuda",
                    "cudnn",
                    "rocm",
                    "triton",
                    "opencl",
                    "metal",
                    "vulkan",
                    "cpu",
                ],
                vendor_libs={
                    "cuda": "cublas",
                    "cudnn": "cudnn",
                    "rocm": "rocblas",
                    "metal": "mps",
                },
                importance="critical",
                description="Matrix multiplication - core of MLPs and Transformers",
                manual_kernel_available=True,
            )
        )

        self.register(
            OperatorSpec(
                name="bmm",
                category=OperatorCategory.LINEAR_ALGEBRA,
                backends=["cuda", "cudnn", "rocm", "triton", "opencl", "metal", "cpu"],
                vendor_libs={
                    "cuda": "cublas",
                    "cudnn": "cudnn",
                    "rocm": "rocblas",
                    "metal": "mps",
                },
                importance="high",
                description="Batched matrix multiplication",
                manual_kernel_available=True,
            )
        )

        self.register(
            OperatorSpec(
                name="einsum",
                category=OperatorCategory.LINEAR_ALGEBRA,
                backends=["cuda", "triton", "cpu"],
                importance="medium",
                description="Einstein summation convention",
                autogen_supported=True,
            )
        )

        # Convolution Operations
        self.register(
            OperatorSpec(
                name="conv1d",
                category=OperatorCategory.CONVOLUTION,
                backends=["cuda", "cudnn", "rocm", "opencl", "metal", "cpu"],
                vendor_libs={"cudnn": "cudnn", "rocm": "miopen", "metal": "mps"},
                importance="high",
                description="1D convolution for sequence models",
                manual_kernel_available=False,
            )
        )

        self.register(
            OperatorSpec(
                name="conv2d",
                category=OperatorCategory.CONVOLUTION,
                backends=[
                    "cuda",
                    "cudnn",
                    "rocm",
                    "triton",
                    "opencl",
                    "metal",
                    "vulkan",
                    "cpu",
                ],
                vendor_libs={"cudnn": "cudnn", "rocm": "miopen", "metal": "mps"},
                importance="critical",
                description="2D convolution - core CNN operation",
                manual_kernel_available=True,
            )
        )

        self.register(
            OperatorSpec(
                name="conv3d",
                category=OperatorCategory.CONVOLUTION,
                backends=["cuda", "cudnn", "rocm", "metal", "cpu"],
                vendor_libs={"cudnn": "cudnn", "rocm": "miopen", "metal": "mps"},
                importance="medium",
                description="3D convolution for video/volumetric data",
                manual_kernel_available=False,
            )
        )

        # Attention Operations
        self.register(
            OperatorSpec(
                name="scaled_dot_product_attention",
                category=OperatorCategory.ATTENTION,
                backends=["cuda", "cudnn", "rocm", "triton", "cpu"],
                vendor_libs={"cudnn": "cudnn"},
                importance="critical",
                description="Scaled dot-product attention (FlashAttention-style)",
                manual_kernel_available=True,
            )
        )

        self.register(
            OperatorSpec(
                name="multi_head_attention",
                category=OperatorCategory.ATTENTION,
                backends=["cuda", "triton", "cpu"],
                importance="critical",
                description="Multi-head attention for Transformers",
                autogen_supported=True,
            )
        )

        # Normalization Operations
        self.register(
            OperatorSpec(
                name="layernorm",
                category=OperatorCategory.NORMALIZATION,
                backends=["cuda", "cudnn", "rocm", "triton", "opencl", "metal", "cpu"],
                vendor_libs={"cudnn": "cudnn", "rocm": "miopen"},
                importance="critical",
                description="Layer normalization for stable training",
                manual_kernel_available=True,
            )
        )

        self.register(
            OperatorSpec(
                name="batchnorm",
                category=OperatorCategory.NORMALIZATION,
                backends=["cuda", "cudnn", "rocm", "opencl", "metal", "cpu"],
                vendor_libs={"cudnn": "cudnn", "rocm": "miopen", "metal": "mps"},
                importance="high",
                description="Batch normalization",
                manual_kernel_available=True,
            )
        )

        self.register(
            OperatorSpec(
                name="groupnorm",
                category=OperatorCategory.NORMALIZATION,
                backends=["cuda", "triton", "cpu"],
                importance="medium",
                description="Group normalization",
            )
        )

        self.register(
            OperatorSpec(
                name="rmsnorm",
                category=OperatorCategory.NORMALIZATION,
                backends=["cuda", "triton", "cpu"],
                importance="high",
                description="RMS normalization (used in LLaMA)",
                manual_kernel_available=True,
            )
        )

        # Activation Functions
        self.register(
            OperatorSpec(
                name="relu",
                category=OperatorCategory.ACTIVATION,
                backends=[
                    "cuda",
                    "cudnn",
                    "rocm",
                    "triton",
                    "opencl",
                    "metal",
                    "vulkan",
                    "cpu",
                ],
                vendor_libs={"cudnn": "cudnn", "rocm": "miopen"},
                importance="high",
                description="ReLU activation",
                manual_kernel_available=True,
            )
        )

        self.register(
            OperatorSpec(
                name="gelu",
                category=OperatorCategory.ACTIVATION,
                backends=["cuda", "cudnn", "rocm", "triton", "opencl", "metal", "cpu"],
                vendor_libs={"cudnn": "cudnn"},
                importance="critical",
                description="GELU activation (Transformers)",
                manual_kernel_available=True,
            )
        )

        self.register(
            OperatorSpec(
                name="silu",
                category=OperatorCategory.ACTIVATION,
                backends=["cuda", "triton", "opencl", "metal", "cpu"],
                importance="high",
                description="SiLU/Swish activation",
                manual_kernel_available=True,
            )
        )

        self.register(
            OperatorSpec(
                name="sigmoid",
                category=OperatorCategory.ACTIVATION,
                backends=["cuda", "cudnn", "rocm", "triton", "opencl", "metal", "cpu"],
                vendor_libs={"cudnn": "cudnn"},
                importance="medium",
                description="Sigmoid activation",
            )
        )

        self.register(
            OperatorSpec(
                name="tanh",
                category=OperatorCategory.ACTIVATION,
                backends=["cuda", "cudnn", "rocm", "triton", "opencl", "metal", "cpu"],
                vendor_libs={"cudnn": "cudnn"},
                importance="medium",
                description="Tanh activation",
            )
        )

        # Pooling Operations
        self.register(
            OperatorSpec(
                name="maxpool2d",
                category=OperatorCategory.POOLING,
                backends=["cuda", "cudnn", "rocm", "opencl", "metal", "cpu"],
                vendor_libs={"cudnn": "cudnn", "rocm": "miopen", "metal": "mps"},
                importance="high",
                description="2D max pooling",
                manual_kernel_available=True,
            )
        )

        self.register(
            OperatorSpec(
                name="avgpool2d",
                category=OperatorCategory.POOLING,
                backends=["cuda", "cudnn", "rocm", "opencl", "metal", "cpu"],
                vendor_libs={"cudnn": "cudnn", "rocm": "miopen", "metal": "mps"},
                importance="high",
                description="2D average pooling",
            )
        )

        self.register(
            OperatorSpec(
                name="adaptiveavgpool2d",
                category=OperatorCategory.POOLING,
                backends=["cuda", "triton", "cpu"],
                importance="medium",
                description="Adaptive average pooling",
            )
        )

        # Softmax Operations
        self.register(
            OperatorSpec(
                name="softmax",
                category=OperatorCategory.SOFTMAX,
                backends=["cuda", "cudnn", "rocm", "triton", "opencl", "metal", "cpu"],
                vendor_libs={"cudnn": "cudnn", "rocm": "miopen"},
                importance="critical",
                description="Softmax for attention and classification",
                manual_kernel_available=True,
            )
        )

        self.register(
            OperatorSpec(
                name="logsoftmax",
                category=OperatorCategory.SOFTMAX,
                backends=["cuda", "cudnn", "rocm", "triton", "cpu"],
                vendor_libs={"cudnn": "cudnn"},
                importance="medium",
                description="Log-softmax for numerical stability",
            )
        )

        # RNN Operations
        self.register(
            OperatorSpec(
                name="rnn",
                category=OperatorCategory.RNN,
                backends=["cuda", "cudnn", "rocm", "cpu"],
                vendor_libs={"cudnn": "cudnn", "rocm": "miopen"},
                importance="medium",
                description="Basic RNN cell",
            )
        )

        self.register(
            OperatorSpec(
                name="lstm",
                category=OperatorCategory.RNN,
                backends=["cuda", "cudnn", "rocm", "cpu"],
                vendor_libs={"cudnn": "cudnn", "rocm": "miopen"},
                importance="high",
                description="LSTM cell",
            )
        )

        self.register(
            OperatorSpec(
                name="gru",
                category=OperatorCategory.RNN,
                backends=["cuda", "cudnn", "rocm", "cpu"],
                vendor_libs={"cudnn": "cudnn"},
                importance="medium",
                description="GRU cell",
            )
        )

        # Elementwise Operations
        self.register(
            OperatorSpec(
                name="add",
                category=OperatorCategory.ELEMENTWISE,
                backends=["cuda", "rocm", "triton", "opencl", "metal", "vulkan", "cpu"],
                importance="high",
                description="Elementwise addition",
                manual_kernel_available=True,
            )
        )

        self.register(
            OperatorSpec(
                name="mul",
                category=OperatorCategory.ELEMENTWISE,
                backends=["cuda", "rocm", "triton", "opencl", "metal", "vulkan", "cpu"],
                importance="high",
                description="Elementwise multiplication",
            )
        )

        self.register(
            OperatorSpec(
                name="div",
                category=OperatorCategory.ELEMENTWISE,
                backends=["cuda", "rocm", "triton", "opencl", "metal", "cpu"],
                importance="medium",
                description="Elementwise division",
            )
        )

        # Reduction Operations
        self.register(
            OperatorSpec(
                name="sum",
                category=OperatorCategory.REDUCTION,
                backends=["cuda", "rocm", "triton", "opencl", "metal", "cpu"],
                importance="high",
                description="Sum reduction",
            )
        )

        self.register(
            OperatorSpec(
                name="mean",
                category=OperatorCategory.REDUCTION,
                backends=["cuda", "rocm", "triton", "opencl", "metal", "cpu"],
                importance="high",
                description="Mean reduction",
            )
        )

        self.register(
            OperatorSpec(
                name="max",
                category=OperatorCategory.REDUCTION,
                backends=["cuda", "rocm", "triton", "opencl", "metal", "cpu"],
                importance="medium",
                description="Max reduction",
            )
        )

        # Fused Operations
        self.register(
            OperatorSpec(
                name="fused_bias_gelu",
                category=OperatorCategory.FUSED,
                backends=["cuda", "triton", "cpu"],
                importance="high",
                description="Fused bias addition + GELU",
                manual_kernel_available=True,
            )
        )

        self.register(
            OperatorSpec(
                name="fused_add_layernorm",
                category=OperatorCategory.FUSED,
                backends=["cuda", "triton", "cpu"],
                importance="critical",
                description="Fused residual add + layer norm",
                manual_kernel_available=True,
            )
        )

        self.register(
            OperatorSpec(
                name="fused_matmul_bias_silu",
                category=OperatorCategory.FUSED,
                backends=["cuda", "triton", "cpu"],
                importance="high",
                description="Fused matmul + bias + SiLU",
                manual_kernel_available=True,
            )
        )

        self.register(
            OperatorSpec(
                name="fused_conv2d_relu",
                category=OperatorCategory.FUSED,
                backends=["cuda", "cudnn", "opencl", "cpu"],
                vendor_libs={"cudnn": "cudnn"},
                importance="high",
                description="Fused conv2d + ReLU",
                manual_kernel_available=True,
            )
        )

    def register(self, spec: OperatorSpec):
        """Register a new operator specification."""
        self._registry[spec.name] = spec

    def get(self, name: str) -> Optional[OperatorSpec]:
        """Get operator specification by name."""
        return self._registry.get(name)

    def list_all(self) -> List[str]:
        """List all registered operator names."""
        return sorted(self._registry.keys())

    def list_by_category(self, category: OperatorCategory) -> List[str]:
        """List operators in a specific category."""
        return sorted(
            [name for name, spec in self._registry.items() if spec.category == category]
        )

    def list_by_backend(self, backend: str) -> List[str]:
        """List operators supported by a backend."""
        return sorted(
            [
                name
                for name, spec in self._registry.items()
                if spec.supports_backend(backend)
            ]
        )

    def list_by_importance(self, importance: str) -> List[str]:
        """List operators by importance level."""
        return sorted(
            [
                name
                for name, spec in self._registry.items()
                if spec.importance == importance
            ]
        )

    def get_backends_for_op(self, op_name: str) -> List[str]:
        """Get prioritized list of backends for an operator."""
        spec = self.get(op_name)
        return spec.backends if spec else []

    def get_coverage_matrix(self) -> Dict[str, Dict[str, bool]]:
        """
        Generate a coverage matrix showing which backends support which ops.

        Returns:
            Dict mapping op_name -> {backend_name: supported}
        """
        all_backends = set()
        for spec in self._registry.values():
            all_backends.update(spec.backends)

        matrix = {}
        for name, spec in self._registry.items():
            matrix[name] = {
                backend: backend in spec.backends for backend in sorted(all_backends)
            }
        return matrix

    def get_statistics(self) -> Dict[str, any]:
        """Get registry statistics."""
        total_ops = len(self._registry)
        by_category = {}
        by_importance = {}
        manual_kernels = 0

        for spec in self._registry.values():
            cat = spec.category.value
            by_category[cat] = by_category.get(cat, 0) + 1
            by_importance[spec.importance] = by_importance.get(spec.importance, 0) + 1
            if spec.manual_kernel_available:
                manual_kernels += 1

        return {
            "total_operators": total_ops,
            "by_category": by_category,
            "by_importance": by_importance,
            "manual_kernels_available": manual_kernels,
        }


# Global singleton registry
_global_registry = None


def get_registry() -> OperatorRegistry:
    """Get the global operator registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = OperatorRegistry()
    return _global_registry
