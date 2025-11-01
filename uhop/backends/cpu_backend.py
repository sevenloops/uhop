"""
CPU Backend for UHOP.

Universal fallback backend using NumPy and PyTorch CPU operations.
Ensures all operators work even without GPU acceleration.
"""

import logging
from typing import Dict, List

import numpy as np

from uhop.backends.base import Backend

logger = logging.getLogger(__name__)


class CPUBackend(Backend):
    """
    CPU fallback backend using NumPy and PyTorch CPU.

    Provides reliable fallback for all operators when GPU is not available.
    """

    def __init__(self):
        super().__init__("cpu")
        self._torch = None
        self._numpy_available = False

    def initialize(self) -> bool:
        """Initialize CPU backend (always available)."""
        if self._initialized:
            return self.capabilities.available

        try:
            import torch

            self._torch = torch
            self._numpy_available = True

            self.capabilities.available = True
            self.capabilities.device_count = 1
            self.capabilities.device_names = ["CPU"]

            # Get CPU info
            import psutil

            self.capabilities.memory_gb = psutil.virtual_memory().total / (1024**3)

            logger.info("[CPU] Initialized CPU backend")

            # Set up CPU kernels
            self._setup_cpu_kernels()

        except Exception as e:
            # CPU backend should almost never fail
            self.capabilities.available = False
            self.capabilities.error_msg = str(e)
            logger.error(f"[CPU] Initialization failed: {e}")

        self._initialized = True
        return self.capabilities.available

    def check_vendor_libs(self) -> Dict[str, bool]:
        """CPU uses NumPy/PyTorch - no special vendor libs."""
        return {
            "numpy": self._numpy_available,
        }

    def get_supported_ops(self) -> List[str]:
        """CPU supports all operators (as fallback)."""
        return [
            # Linear algebra
            "matmul",
            "bmm",
            "einsum",
            # Convolutions
            "conv1d",
            "conv2d",
            "conv3d",
            # Activations
            "relu",
            "gelu",
            "silu",
            "sigmoid",
            "tanh",
            # Normalization
            "layernorm",
            "batchnorm",
            "groupnorm",
            "rmsnorm",
            # Pooling
            "maxpool2d",
            "avgpool2d",
            "adaptiveavgpool2d",
            # Softmax
            "softmax",
            "logsoftmax",
            # RNN
            "rnn",
            "lstm",
            "gru",
            # Elementwise
            "add",
            "mul",
            "div",
            # Reductions
            "sum",
            "mean",
            "max",
            # Attention
            "scaled_dot_product_attention",
            "multi_head_attention",
            # Fused ops
            "fused_bias_gelu",
            "fused_add_layernorm",
            "fused_matmul_bias_silu",
            "fused_conv2d_relu",
        ]

    def _setup_cpu_kernels(self):
        """Set up CPU-based kernels using PyTorch/NumPy."""
        if self._torch is None:
            return

        # Matrix multiplication
        def cpu_matmul(A, B):
            if not isinstance(A, self._torch.Tensor):
                if isinstance(A, np.ndarray):
                    A = self._torch.from_numpy(A).float()
                else:
                    A = self._torch.tensor(A, dtype=self._torch.float32)
            if not isinstance(B, self._torch.Tensor):
                if isinstance(B, np.ndarray):
                    B = self._torch.from_numpy(B).float()
                else:
                    B = self._torch.tensor(B, dtype=self._torch.float32)

            return self._torch.matmul(A.cpu(), B.cpu())

        self.register_vendor_kernel("matmul", cpu_matmul, "cpu")

        # Batched matrix multiplication
        def cpu_bmm(A, B):
            if not isinstance(A, self._torch.Tensor):
                A = self._torch.tensor(A, dtype=self._torch.float32)
            if not isinstance(B, self._torch.Tensor):
                B = self._torch.tensor(B, dtype=self._torch.float32)

            return self._torch.bmm(A.cpu(), B.cpu())

        self.register_vendor_kernel("bmm", cpu_bmm, "cpu")

        # Conv2D
        def cpu_conv2d(
            input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1
        ):
            if not isinstance(input, self._torch.Tensor):
                input = self._torch.tensor(input, dtype=self._torch.float32)
            if not isinstance(weight, self._torch.Tensor):
                weight = self._torch.tensor(weight, dtype=self._torch.float32)
            if bias is not None and not isinstance(bias, self._torch.Tensor):
                bias = self._torch.tensor(bias, dtype=self._torch.float32)

            return self._torch.nn.functional.conv2d(
                input.cpu(),
                weight.cpu(),
                bias.cpu() if bias is not None else None,
                stride,
                padding,
                dilation,
                groups,
            )

        self.register_vendor_kernel("conv2d", cpu_conv2d, "cpu")

        # Activations
        def cpu_relu(x):
            if not isinstance(x, self._torch.Tensor):
                x = self._torch.tensor(x, dtype=self._torch.float32)
            return self._torch.relu(x.cpu())

        self.register_vendor_kernel("relu", cpu_relu, "cpu")

        def cpu_gelu(x):
            if not isinstance(x, self._torch.Tensor):
                x = self._torch.tensor(x, dtype=self._torch.float32)
            return self._torch.nn.functional.gelu(x.cpu())

        self.register_vendor_kernel("gelu", cpu_gelu, "cpu")

        def cpu_silu(x):
            if not isinstance(x, self._torch.Tensor):
                x = self._torch.tensor(x, dtype=self._torch.float32)
            return self._torch.nn.functional.silu(x.cpu())

        self.register_vendor_kernel("silu", cpu_silu, "cpu")

        # Layer Normalization
        def cpu_layernorm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
            if not isinstance(input, self._torch.Tensor):
                input = self._torch.tensor(input, dtype=self._torch.float32)

            input = input.cpu()
            if weight is not None:
                weight = weight.cpu()
            if bias is not None:
                bias = bias.cpu()

            return self._torch.nn.functional.layer_norm(
                input, normalized_shape, weight, bias, eps
            )

        self.register_vendor_kernel("layernorm", cpu_layernorm, "cpu")

        # Batch Normalization
        def cpu_batchnorm(
            input,
            running_mean,
            running_var,
            weight=None,
            bias=None,
            training=False,
            momentum=0.1,
            eps=1e-5,
        ):
            if not isinstance(input, self._torch.Tensor):
                input = self._torch.tensor(input, dtype=self._torch.float32)

            return self._torch.nn.functional.batch_norm(
                input.cpu(),
                running_mean.cpu() if running_mean is not None else None,
                running_var.cpu() if running_var is not None else None,
                weight.cpu() if weight is not None else None,
                bias.cpu() if bias is not None else None,
                training,
                momentum,
                eps,
            )

        self.register_vendor_kernel("batchnorm", cpu_batchnorm, "cpu")

        # Softmax
        def cpu_softmax(x, dim=-1):
            if not isinstance(x, self._torch.Tensor):
                x = self._torch.tensor(x, dtype=self._torch.float32)
            return self._torch.nn.functional.softmax(x.cpu(), dim=dim)

        self.register_vendor_kernel("softmax", cpu_softmax, "cpu")

        # Max Pooling
        def cpu_maxpool2d(input, kernel_size, stride=None, padding=0):
            if not isinstance(input, self._torch.Tensor):
                input = self._torch.tensor(input, dtype=self._torch.float32)

            return self._torch.nn.functional.max_pool2d(
                input.cpu(), kernel_size, stride, padding
            )

        self.register_vendor_kernel("maxpool2d", cpu_maxpool2d, "cpu")

        # Elementwise operations
        def cpu_add(x, y):
            if not isinstance(x, self._torch.Tensor):
                x = self._torch.tensor(x, dtype=self._torch.float32)
            if not isinstance(y, self._torch.Tensor):
                y = self._torch.tensor(y, dtype=self._torch.float32)
            return x.cpu() + y.cpu()

        self.register_vendor_kernel("add", cpu_add, "cpu")

        def cpu_mul(x, y):
            if not isinstance(x, self._torch.Tensor):
                x = self._torch.tensor(x, dtype=self._torch.float32)
            if not isinstance(y, self._torch.Tensor):
                y = self._torch.tensor(y, dtype=self._torch.float32)
            return x.cpu() * y.cpu()

        self.register_vendor_kernel("mul", cpu_mul, "cpu")

        # Reductions
        def cpu_sum(x, dim=None, keepdim=False):
            if not isinstance(x, self._torch.Tensor):
                x = self._torch.tensor(x, dtype=self._torch.float32)
            return self._torch.sum(x.cpu(), dim=dim, keepdim=keepdim)

        self.register_vendor_kernel("sum", cpu_sum, "cpu")

        def cpu_mean(x, dim=None, keepdim=False):
            if not isinstance(x, self._torch.Tensor):
                x = self._torch.tensor(x, dtype=self._torch.float32)
            return self._torch.mean(x.cpu(), dim=dim, keepdim=keepdim)

        self.register_vendor_kernel("mean", cpu_mean, "cpu")

        # Fused operations
        def cpu_fused_bias_gelu(x, bias):
            if not isinstance(x, self._torch.Tensor):
                x = self._torch.tensor(x, dtype=self._torch.float32)
            if not isinstance(bias, self._torch.Tensor):
                bias = self._torch.tensor(bias, dtype=self._torch.float32)

            x = x.cpu() + bias.cpu()
            return self._torch.nn.functional.gelu(x)

        self.register_vendor_kernel("fused_bias_gelu", cpu_fused_bias_gelu, "cpu")

        def cpu_fused_add_layernorm(
            x, residual, normalized_shape, weight=None, bias=None, eps=1e-5
        ):
            if not isinstance(x, self._torch.Tensor):
                x = self._torch.tensor(x, dtype=self._torch.float32)
            if not isinstance(residual, self._torch.Tensor):
                residual = self._torch.tensor(residual, dtype=self._torch.float32)

            x = x.cpu() + residual.cpu()
            return self._torch.nn.functional.layer_norm(
                x,
                normalized_shape,
                weight.cpu() if weight is not None else None,
                bias.cpu() if bias is not None else None,
                eps,
            )

        self.register_vendor_kernel(
            "fused_add_layernorm", cpu_fused_add_layernorm, "cpu"
        )

        def cpu_fused_conv2d_relu(input, weight, bias=None, stride=1, padding=0):
            output = cpu_conv2d(input, weight, bias, stride, padding)
            return self._torch.relu(output)

        self.register_vendor_kernel("fused_conv2d_relu", cpu_fused_conv2d_relu, "cpu")

        logger.debug(f"[CPU] Registered {len(self._vendor_kernels)} CPU kernels")
