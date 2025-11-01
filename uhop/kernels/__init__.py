"""
Kernel Registry for UHOP.

Maps operator names to their corresponding kernel files across different backends.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional


class BackendType(Enum):
    """Supported backend types."""

    CUDA = "cuda"
    HIP = "hip"
    METAL = "metal"
    OPENCL = "opencl"


@dataclass
class KernelFileInfo:
    """Information about a kernel file."""

    operator: str
    backend: BackendType
    file_path: Path
    kernel_names: List[str]  # Function names in the kernel file


class KernelRegistry:
    """
    Registry mapping operators to their kernel implementations.
    """

    def __init__(self):
        self.kernels_dir = Path(__file__).parent
        self._registry: Dict[str, Dict[BackendType, KernelFileInfo]] = {}
        self._initialize_registry()

    def _initialize_registry(self):
        """Initialize the kernel registry with all available kernels."""

        # CUDA kernels
        self._register_cuda_kernels()

        # HIP kernels
        self._register_hip_kernels()

        # Metal kernels
        self._register_metal_kernels()

        # OpenCL kernels
        self._register_opencl_kernels()

    def _register_cuda_kernels(self):
        """Register CUDA kernel files."""
        cuda_dir = self.kernels_dir / "cuda"

        kernels = [
            # Elementwise ops
            ("elementwise_add", "elementwise.cu", ["elementwise_add"]),
            ("elementwise_sub", "elementwise.cu", ["elementwise_sub"]),
            ("elementwise_mul", "elementwise.cu", ["elementwise_mul"]),
            ("elementwise_div", "elementwise.cu", ["elementwise_div"]),
            ("elementwise_pow", "elementwise.cu", ["elementwise_pow"]),
            ("elementwise_max", "elementwise.cu", ["elementwise_max"]),
            ("elementwise_min", "elementwise.cu", ["elementwise_min"]),
            ("elementwise_leakyrelu", "elementwise.cu", ["elementwise_leakyrelu"]),
            ("elementwise_exp", "elementwise.cu", ["elementwise_exp"]),
            ("elementwise_log", "elementwise.cu", ["elementwise_log"]),
            ("elementwise_sqrt", "elementwise.cu", ["elementwise_sqrt"]),
            ("sigmoid", "elementwise.cu", ["sigmoid_kernel"]),
            ("tanh", "elementwise.cu", ["tanh_kernel"]),
            ("matmul", "matmul.cu", ["matmul_kernel"]),
            ("bmm", "bmm.cu", ["bmm_kernel"]),
            (
                "einsum",
                "einsum.cu",
                [
                    "einsum_matmul_kernel",
                    "einsum_bmm_kernel",
                    "einsum_tensor_contract_kernel",
                ],
            ),
            ("conv1d", "conv1d.cu", ["conv1d_kernel"]),
            ("conv2d", "conv2d.cu", ["conv2d_kernel"]),
            ("conv3d", "conv3d.cu", ["conv3d_kernel"]),
            # Backward conv2d
            ("conv2d_input_grad", "conv2d_backward.cu", ["conv2d_input_grad"]),
            ("conv2d_weight_grad", "conv2d_backward.cu", ["conv2d_weight_grad"]),
            (
                "scaled_dot_product_attention",
                "attention.cu",
                [
                    "scaled_dot_product_attention_kernel",
                    "attention_softmax_kernel",
                    "attention_output_kernel",
                ],
            ),
            ("layernorm", "layernorm.cu", ["layernorm_kernel"]),
            ("batchnorm", "batchnorm.cu", ["batchnorm_kernel", "batchnorm_2d_kernel"]),
            ("relu", "relu.cu", ["relu_kernel"]),
            ("gelu", "gelu.cu", ["gelu_kernel", "gelu_exact_kernel"]),
            ("silu", "silu.cu", ["silu_kernel", "swish_kernel"]),
            # Reductions
            ("reduce_sum", "reduce.cu", ["reduce_sum_atomic"]),
            ("reduce_max", "reduce.cu", ["reduce_max_atomic"]),
            ("reduce_min", "reduce.cu", ["reduce_min_atomic"]),
            ("reduce_norm", "reduce.cu", ["reduce_norm_atomic"]),
            # Matrix utils
            ("transpose", "transpose_dot.cu", ["transpose2d"]),
            ("dot", "transpose_dot.cu", ["dot_product"]),
            ("maxpool2d", "pooling.cu", ["maxpool2d_kernel"]),
            ("avgpool2d", "pooling.cu", ["avgpool2d_kernel"]),
            ("softmax", "softmax.cu", ["softmax_kernel"]),
            ("logsoftmax", "softmax.cu", ["logsoftmax_kernel"]),
            ("rnn", "rnn.cu", ["rnn_cell_kernel"]),
            ("lstm", "lstm.cu", ["lstm_cell_kernel"]),
            (
                "fused_add_layernorm_gelu",
                "fused_add_norm_act.cu",
                ["fused_add_layernorm_gelu_kernel"],
            ),
            (
                "fused_add_layernorm_relu",
                "fused_add_norm_act.cu",
                ["fused_add_layernorm_relu_kernel"],
            ),
            (
                "fused_add_layernorm_silu",
                "fused_add_norm_act.cu",
                ["fused_add_layernorm_silu_kernel"],
            ),
        ]

        for op_name, filename, kernel_names in kernels:
            self._register(op_name, BackendType.CUDA, cuda_dir / filename, kernel_names)

    def _register_hip_kernels(self):
        """Register HIP kernel files."""
        hip_dir = self.kernels_dir / "hip"

        kernels = [
            ("elementwise_add", "elementwise.hip", ["elementwise_add"]),
            ("elementwise_sub", "elementwise.hip", ["elementwise_sub"]),
            ("elementwise_mul", "elementwise.hip", ["elementwise_mul"]),
            ("sigmoid", "elementwise.hip", ["sigmoid_kernel"]),
            ("matmul", "matmul.hip", ["matmul_kernel"]),
            ("bmm", "bmm.hip", ["bmm_kernel"]),
            ("conv2d", "conv2d.hip", ["conv2d_kernel"]),
            # Backward conv2d
            ("conv2d_input_grad", "conv2d_backward.hip", ["conv2d_input_grad"]),
            ("conv2d_weight_grad", "conv2d_backward.hip", ["conv2d_weight_grad"]),
            ("relu", "activations.hip", ["relu_kernel"]),
            ("gelu", "activations.hip", ["gelu_kernel"]),
            ("silu", "activations.hip", ["silu_kernel"]),
            ("layernorm", "layernorm.hip", ["layernorm_kernel"]),
            ("maxpool2d", "pooling.hip", ["maxpool2d_kernel"]),
            ("avgpool2d", "pooling.hip", ["avgpool2d_kernel"]),
            ("reduce_sum", "reduce.hip", ["reduce_sum_atomic"]),
        ]

        for op_name, filename, kernel_names in kernels:
            self._register(op_name, BackendType.HIP, hip_dir / filename, kernel_names)

    def _register_metal_kernels(self):
        """Register Metal kernel files."""
        metal_dir = self.kernels_dir / "metal"

        kernels = [
            ("elementwise_add", "elementwise.metal", ["elementwise_add"]),
            ("elementwise_mul", "elementwise.metal", ["elementwise_mul"]),
            ("sigmoid", "elementwise.metal", ["sigmoid_kernel"]),
            ("matmul", "matmul.metal", ["matmul_kernel"]),
            ("bmm", "bmm.metal", ["bmm_kernel"]),
            ("conv2d", "conv2d.metal", ["conv2d_kernel"]),
            # Backward conv2d
            ("conv2d_input_grad", "conv2d_backward.metal", ["conv2d_input_grad"]),
            ("conv2d_weight_grad", "conv2d_backward.metal", ["conv2d_weight_grad"]),
            ("relu", "activations.metal", ["relu_kernel"]),
            ("gelu", "activations.metal", ["gelu_kernel"]),
            ("silu", "activations.metal", ["silu_kernel"]),
            ("layernorm", "layernorm.metal", ["layernorm_kernel"]),
            ("maxpool2d", "pooling.metal", ["maxpool2d_kernel"]),
            ("avgpool2d", "pooling.metal", ["avgpool2d_kernel"]),
            ("softmax", "softmax.metal", ["softmax_kernel"]),
            ("logsoftmax", "softmax.metal", ["logsoftmax_kernel"]),
        ]

        for op_name, filename, kernel_names in kernels:
            self._register(
                op_name, BackendType.METAL, metal_dir / filename, kernel_names
            )

    def _register_opencl_kernels(self):
        """Register OpenCL kernel files."""
        opencl_dir = self.kernels_dir / "opencl"

        kernels = [
            # Elementwise
            ("elementwise_add", "elementwise.cl", ["elementwise_add"]),
            ("elementwise_sub", "elementwise.cl", ["elementwise_sub"]),
            ("elementwise_mul", "elementwise.cl", ["elementwise_mul"]),
            ("elementwise_div", "elementwise.cl", ["elementwise_div"]),
            ("elementwise_pow", "elementwise.cl", ["elementwise_pow"]),
            ("elementwise_max", "elementwise.cl", ["elementwise_max"]),
            ("elementwise_min", "elementwise.cl", ["elementwise_min"]),
            ("elementwise_leakyrelu", "elementwise.cl", ["elementwise_leakyrelu"]),
            ("elementwise_exp", "elementwise.cl", ["elementwise_exp"]),
            ("elementwise_log", "elementwise.cl", ["elementwise_log"]),
            ("elementwise_sqrt", "elementwise.cl", ["elementwise_sqrt"]),
            ("sigmoid", "elementwise.cl", ["sigmoid_kernel"]),
            ("tanh", "elementwise.cl", ["tanh_kernel"]),
            ("matmul", "matmul.cl", ["matmul_kernel"]),
            ("bmm", "bmm.cl", ["bmm_kernel"]),
            ("conv1d", "conv1d.cl", ["conv1d_kernel"]),
            ("conv2d", "conv2d.cl", ["conv2d_kernel"]),
            # Backward conv2d
            ("conv2d_input_grad", "conv2d_backward.cl", ["conv2d_input_grad"]),
            ("conv2d_weight_grad", "conv2d_backward.cl", ["conv2d_weight_grad"]),
            ("depthwise_conv2d", "depthwise_conv2d.cl", ["depthwise_conv2d"]),
            ("relu", "activations.cl", ["relu_kernel"]),
            ("gelu", "activations.cl", ["gelu_kernel"]),
            ("silu", "activations.cl", ["silu_kernel"]),
            ("layernorm", "layernorm.cl", ["layernorm_kernel"]),
            ("group_norm", "groupnorm.cl", ["group_norm"]),
            ("maxpool2d", "pooling.cl", ["maxpool2d_kernel"]),
            ("avgpool2d", "pooling.cl", ["avgpool2d_kernel"]),
            ("adaptive_avgpool2d", "adaptive_pool2d.cl", ["adaptive_avgpool2d"]),
            ("adaptive_maxpool2d", "adaptive_pool2d.cl", ["adaptive_maxpool2d"]),
            ("softmax", "softmax.cl", ["softmax_kernel"]),
            ("logsoftmax", "softmax.cl", ["logsoftmax_kernel"]),
            # Tensor utilities
            ("reshape", "tensor_utils.cl", ["reshape_noop"]),
            ("slice", "tensor_utils.cl", ["slice2d"]),
            ("concat", "tensor_utils.cl", ["concat1d"]),
            ("pad", "tensor_utils.cl", ["pad2d_constant"]),
            ("broadcast_to", "tensor_utils.cl", ["broadcast_row"]),
            # Matrix utils
            ("transpose", "transpose_dot.cl", ["transpose2d"]),
            ("dot", "transpose_dot.cl", ["dot_product"]),
            # Reductions (two-stage)
            ("reduce_sum", "reduce.cl", ["reduce_sum_partials", "reduce_sum_finalize"]),
            (
                "reduce_mean",
                "reduce.cl",
                ["reduce_sum_partials", "reduce_mean_finalize"],
            ),
            ("reduce_max", "reduce.cl", ["reduce_max_partials", "reduce_max_finalize"]),
            ("reduce_min", "reduce.cl", ["reduce_min_partials", "reduce_min_finalize"]),
            (
                "reduce_norm",
                "reduce.cl",
                ["reduce_norm_partials", "reduce_norm_finalize"],
            ),
        ]

        for op_name, filename, kernel_names in kernels:
            self._register(
                op_name, BackendType.OPENCL, opencl_dir / filename, kernel_names
            )

    def _register(
        self,
        operator: str,
        backend: BackendType,
        file_path: Path,
        kernel_names: List[str],
    ):
        """Register a kernel file."""
        if operator not in self._registry:
            self._registry[operator] = {}

        self._registry[operator][backend] = KernelFileInfo(
            operator=operator,
            backend=backend,
            file_path=file_path,
            kernel_names=kernel_names,
        )

    def get_kernel_file(
        self, operator: str, backend: BackendType
    ) -> Optional[KernelFileInfo]:
        """Get kernel file information for an operator and backend."""
        return self._registry.get(operator, {}).get(backend)

    def get_all_kernels(self, operator: str) -> Dict[BackendType, KernelFileInfo]:
        """Get all kernel implementations for an operator."""
        return self._registry.get(operator, {})

    def get_available_operators(self, backend: BackendType) -> List[str]:
        """Get list of operators supported by a backend."""
        operators = []
        for op_name, backends in self._registry.items():
            if backend in backends:
                operators.append(op_name)
        return sorted(operators)

    def get_kernel_source(self, operator: str, backend: BackendType) -> Optional[str]:
        """Load and return the kernel source code."""
        kernel_info = self.get_kernel_file(operator, backend)
        if kernel_info and kernel_info.file_path.exists():
            return kernel_info.file_path.read_text()
        return None


# Global registry instance
_registry = None


def get_kernel_registry() -> KernelRegistry:
    """Get the global kernel registry instance."""
    global _registry
    if _registry is None:
        _registry = KernelRegistry()
    return _registry
