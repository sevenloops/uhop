"""
cuDNN Backend for UHOP.

Provides access to NVIDIA's cuDNN library for optimized deep learning primitives.
"""

import logging
from typing import Dict, List

from uhop.backends.base import Backend

logger = logging.getLogger(__name__)


class CuDNNBackend(Backend):
    """
    cuDNN backend for high-performance deep learning operations.

    Supports convolutions, normalization, activations, pooling, RNNs, and more.
    """

    def __init__(self):
        super().__init__("cudnn")
        self._torch = None
        self._cudnn_available = False
        self._cudnn_version = None

    def initialize(self) -> bool:
        """Initialize cuDNN backend."""
        if self._initialized:
            return self.capabilities.available

        try:
            import torch

            self._torch = torch

            if not torch.cuda.is_available():
                self.capabilities.available = False
                self.capabilities.error_msg = "CUDA not available"
                self._initialized = True
                return False

            # Check cuDNN availability
            if not torch.backends.cudnn.is_available():
                self.capabilities.available = False
                self.capabilities.error_msg = "cuDNN not available"
                self._initialized = True
                return False

            self._cudnn_available = True
            self._cudnn_version = torch.backends.cudnn.version()

            # Get device info (same as CUDA)
            self.capabilities.available = True
            self.capabilities.device_count = torch.cuda.device_count()

            for i in range(self.capabilities.device_count):
                device_name = torch.cuda.get_device_name(i)
                self.capabilities.device_names.append(device_name)

                props = torch.cuda.get_device_properties(i)
                self.capabilities.compute_capability = f"{props.major}.{props.minor}"
                self.capabilities.memory_gb = props.total_memory / (1024**3)

            self.capabilities.vendor_libs["cudnn"] = True

            logger.info(
                f"[cuDNN] Initialized v{self._cudnn_version} with "
                f"{self.capabilities.device_count} device(s)"
            )

            # Set up cuDNN-optimized kernels
            self._setup_cudnn_kernels()

        except ImportError:
            self.capabilities.available = False
            self.capabilities.error_msg = "torch not installed"
            logger.warning("[cuDNN] torch not available")
        except Exception as e:
            self.capabilities.available = False
            self.capabilities.error_msg = str(e)
            logger.error(f"[cuDNN] Initialization failed: {e}")

        self._initialized = True
        return self.capabilities.available

    def check_vendor_libs(self) -> Dict[str, bool]:
        """Check cuDNN availability."""
        return {
            "cudnn": self._cudnn_available,
        }

    def get_supported_ops(self) -> List[str]:
        """Get list of cuDNN-supported operators."""
        ops = [
            # Convolutions (cuDNN's specialty)
            "conv1d",
            "conv2d",
            "conv3d",
            # Normalization
            "batchnorm",
            "layernorm",
            # Activations
            "relu",
            "sigmoid",
            "tanh",
            "gelu",
            # Pooling
            "maxpool2d",
            "avgpool2d",
            # Softmax
            "softmax",
            "logsoftmax",
            # RNN
            "rnn",
            "lstm",
            "gru",
            # Fused ops
            "fused_conv2d_relu",
            # Linear algebra (via cuBLAS)
            "matmul",
            "bmm",
            # Attention (cuDNN 8.9+)
            "scaled_dot_product_attention",
        ]
        return ops

    def _synchronize(self):
        """Synchronize CUDA device."""
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.synchronize()

    def _setup_cudnn_kernels(self):
        """Set up cuDNN-optimized kernels."""
        if not self._cudnn_available or self._torch is None:
            return

        # Enable cuDNN benchmarking for optimal performance
        self._torch.backends.cudnn.benchmark = True

        # Conv2D (cuDNN's bread and butter)
        def cudnn_conv2d(
            input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1
        ):
            if not isinstance(input, self._torch.Tensor):
                input = self._torch.tensor(input, dtype=self._torch.float32)
            if not isinstance(weight, self._torch.Tensor):
                weight = self._torch.tensor(weight, dtype=self._torch.float32)
            if bias is not None and not isinstance(bias, self._torch.Tensor):
                bias = self._torch.tensor(bias, dtype=self._torch.float32)

            input = input.cuda()
            weight = weight.cuda()
            if bias is not None:
                bias = bias.cuda()

            return self._torch.nn.functional.conv2d(
                input, weight, bias, stride, padding, dilation, groups
            )

        self.register_vendor_kernel("conv2d", cudnn_conv2d, "cudnn")

        # Batch Normalization
        def cudnn_batchnorm(
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

            input = input.cuda()
            if running_mean is not None:
                running_mean = running_mean.cuda()
            if running_var is not None:
                running_var = running_var.cuda()
            if weight is not None:
                weight = weight.cuda()
            if bias is not None:
                bias = bias.cuda()

            return self._torch.nn.functional.batch_norm(
                input, running_mean, running_var, weight, bias, training, momentum, eps
            )

        self.register_vendor_kernel("batchnorm", cudnn_batchnorm, "cudnn")

        # Layer Normalization
        def cudnn_layernorm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
            if not isinstance(input, self._torch.Tensor):
                input = self._torch.tensor(input, dtype=self._torch.float32)

            input = input.cuda()
            if weight is not None:
                weight = weight.cuda()
            if bias is not None:
                bias = bias.cuda()

            return self._torch.nn.functional.layer_norm(
                input, normalized_shape, weight, bias, eps
            )

        self.register_vendor_kernel("layernorm", cudnn_layernorm, "cudnn")

        # ReLU (cuDNN-accelerated)
        def cudnn_relu(x):
            if not isinstance(x, self._torch.Tensor):
                x = self._torch.tensor(x, dtype=self._torch.float32)
            x = x.cuda()
            # cuDNN uses optimized implementation
            return self._torch.nn.functional.relu(x)

        self.register_vendor_kernel("relu", cudnn_relu, "cudnn")

        # Softmax
        def cudnn_softmax(x, dim=-1):
            if not isinstance(x, self._torch.Tensor):
                x = self._torch.tensor(x, dtype=self._torch.float32)
            x = x.cuda()
            return self._torch.nn.functional.softmax(x, dim=dim)

        self.register_vendor_kernel("softmax", cudnn_softmax, "cudnn")

        # Max Pooling 2D
        def cudnn_maxpool2d(input, kernel_size, stride=None, padding=0, dilation=1):
            if not isinstance(input, self._torch.Tensor):
                input = self._torch.tensor(input, dtype=self._torch.float32)
            input = input.cuda()

            return self._torch.nn.functional.max_pool2d(
                input, kernel_size, stride, padding, dilation
            )

        self.register_vendor_kernel("maxpool2d", cudnn_maxpool2d, "cudnn")

        # LSTM (cuDNN has highly optimized RNN kernels)
        def cudnn_lstm(input, hidden, weight_ih, weight_hh, bias_ih=None, bias_hh=None):
            # Note: This is a simplified version; full LSTM requires proper state management
            if not isinstance(input, self._torch.Tensor):
                input = self._torch.tensor(input, dtype=self._torch.float32)

            input = input.cuda()
            # cuDNN LSTM implementation via torch.nn.LSTM
            # For production, would need to properly instantiate LSTM module
            return input  # Placeholder

        # Register but note it needs proper implementation
        # self.register_vendor_kernel("lstm", cudnn_lstm, "cudnn")

        # Fused Conv2D + ReLU
        def cudnn_fused_conv2d_relu(input, weight, bias=None, stride=1, padding=0):
            output = cudnn_conv2d(input, weight, bias, stride, padding)
            return self._torch.nn.functional.relu(output)

        self.register_vendor_kernel(
            "fused_conv2d_relu", cudnn_fused_conv2d_relu, "cudnn"
        )

        # Scaled Dot-Product Attention (cuDNN 8.9+, torch 2.0+)
        if hasattr(self._torch.nn.functional, "scaled_dot_product_attention"):

            def cudnn_sdpa(
                query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False
            ):
                if not isinstance(query, self._torch.Tensor):
                    query = self._torch.tensor(query, dtype=self._torch.float32)
                if not isinstance(key, self._torch.Tensor):
                    key = self._torch.tensor(key, dtype=self._torch.float32)
                if not isinstance(value, self._torch.Tensor):
                    value = self._torch.tensor(value, dtype=self._torch.float32)

                query = query.cuda()
                key = key.cuda()
                value = value.cuda()

                return self._torch.nn.functional.scaled_dot_product_attention(
                    query, key, value, attn_mask, dropout_p, is_causal
                )

            self.register_vendor_kernel(
                "scaled_dot_product_attention", cudnn_sdpa, "cudnn"
            )

        logger.debug(f"[cuDNN] Registered {len(self._vendor_kernels)} vendor kernels")
