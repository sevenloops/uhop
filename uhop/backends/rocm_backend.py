"""
ROCm Backend for UHOP.

Provides support for AMD GPUs using ROCm, MIOpen, and rocBLAS.
"""
from typing import Dict, List, Optional
import logging

from uhop.backends.base import Backend, KernelSource

logger = logging.getLogger(__name__)


class ROCmBackend(Backend):
    """
    ROCm backend for AMD GPU acceleration.
    
    Uses MIOpen for deep learning primitives and rocBLAS for BLAS operations.
    """
    
    def __init__(self):
        super().__init__("rocm")
        self._torch = None
        self._rocm_available = False
        self._miopen_available = False
        self._rocblas_available = False
    
    def initialize(self) -> bool:
        """Initialize ROCm backend."""
        if self._initialized:
            return self.capabilities.available
        
        try:
            import torch
            self._torch = torch
            
            # Check if ROCm build of PyTorch is available
            if not hasattr(torch.version, 'hip') or torch.version.hip is None:
                self.capabilities.available = False
                self.capabilities.error_msg = "ROCm PyTorch not installed"
                self._initialized = True
                return False
            
            if not torch.cuda.is_available():  # PyTorch ROCm uses cuda namespace
                self.capabilities.available = False
                self.capabilities.error_msg = "No AMD GPU detected"
                self._initialized = True
                return False
            
            self._rocm_available = True
            
            # Get device info
            self.capabilities.available = True
            self.capabilities.device_count = torch.cuda.device_count()
            
            for i in range(self.capabilities.device_count):
                device_name = torch.cuda.get_device_name(i)
                self.capabilities.device_names.append(device_name)
                
                props = torch.cuda.get_device_properties(i)
                # ROCm uses gcnArchName
                self.capabilities.compute_capability = getattr(props, 'gcnArchName', 'unknown')
                self.capabilities.memory_gb = props.total_memory / (1024**3)
            
            # Check for MIOpen (ROCm's equivalent to cuDNN)
            self._miopen_available = hasattr(torch.backends, 'miopen') and torch.backends.miopen.is_available()
            if self._miopen_available:
                self.capabilities.vendor_libs["miopen"] = True
            
            # rocBLAS is always available with ROCm
            self._rocblas_available = True
            self.capabilities.vendor_libs["rocblas"] = True
            
            logger.info(
                f"[ROCm] Initialized with {self.capabilities.device_count} AMD GPU(s), "
                f"MIOpen: {self._miopen_available}"
            )
            
            # Set up vendor kernels
            self._setup_rocm_kernels()
            
        except ImportError:
            self.capabilities.available = False
            self.capabilities.error_msg = "ROCm PyTorch not installed"
            logger.warning("[ROCm] ROCm PyTorch not available")
        except Exception as e:
            self.capabilities.available = False
            self.capabilities.error_msg = str(e)
            logger.error(f"[ROCm] Initialization failed: {e}")
        
        self._initialized = True
        return self.capabilities.available
    
    def check_vendor_libs(self) -> Dict[str, bool]:
        """Check ROCm library availability."""
        return {
            "rocblas": self._rocblas_available,
            "miopen": self._miopen_available,
        }
    
    def get_supported_ops(self) -> List[str]:
        """Get list of ROCm-supported operators."""
        ops = [
            # Linear algebra (rocBLAS)
            "matmul", "bmm",
            # Convolutions (MIOpen)
            "conv1d", "conv2d", "conv3d",
            # Normalization (MIOpen)
            "batchnorm", "layernorm",
            # Activations
            "relu", "gelu", "silu", "sigmoid", "tanh",
            # Pooling (MIOpen)
            "maxpool2d", "avgpool2d",
            # Softmax
            "softmax", "logsoftmax",
            # RNN (MIOpen)
            "rnn", "lstm", "gru",
            # Elementwise
            "add", "mul", "div",
            # Reductions
            "sum", "mean", "max",
        ]
        return ops
    
    def _synchronize(self):
        """Synchronize ROCm device."""
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.synchronize()
    
    def _setup_rocm_kernels(self):
        """Set up ROCm/MIOpen vendor kernels."""
        if not self._rocm_available or self._torch is None:
            return
        
        # Matrix multiplication via rocBLAS
        def rocm_matmul(A, B):
            if not isinstance(A, self._torch.Tensor):
                A = self._torch.tensor(A, dtype=self._torch.float32)
            if not isinstance(B, self._torch.Tensor):
                B = self._torch.tensor(B, dtype=self._torch.float32)
            
            A = A.cuda()  # ROCm uses cuda namespace
            B = B.cuda()
            return self._torch.matmul(A, B)
        
        self.register_vendor_kernel("matmul", rocm_matmul, "rocblas")
        
        # Batched matrix multiplication
        def rocm_bmm(A, B):
            if not isinstance(A, self._torch.Tensor):
                A = self._torch.tensor(A, dtype=self._torch.float32)
            if not isinstance(B, self._torch.Tensor):
                B = self._torch.tensor(B, dtype=self._torch.float32)
            
            A = A.cuda()
            B = B.cuda()
            return self._torch.bmm(A, B)
        
        self.register_vendor_kernel("bmm", rocm_bmm, "rocblas")
        
        if self._miopen_available:
            # Conv2D via MIOpen
            def miopen_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
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
            
            self.register_vendor_kernel("conv2d", miopen_conv2d, "miopen")
            
            # Batch Normalization
            def miopen_batchnorm(input, running_mean, running_var, weight=None, bias=None,
                                training=False, momentum=0.1, eps=1e-5):
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
            
            self.register_vendor_kernel("batchnorm", miopen_batchnorm, "miopen")
            
            # Max Pooling
            def miopen_maxpool2d(input, kernel_size, stride=None, padding=0):
                if not isinstance(input, self._torch.Tensor):
                    input = self._torch.tensor(input, dtype=self._torch.float32)
                input = input.cuda()
                
                return self._torch.nn.functional.max_pool2d(
                    input, kernel_size, stride, padding
                )
            
            self.register_vendor_kernel("maxpool2d", miopen_maxpool2d, "miopen")
        
        # Basic activations
        def rocm_relu(x):
            if not isinstance(x, self._torch.Tensor):
                x = self._torch.tensor(x, dtype=self._torch.float32)
            x = x.cuda()
            return self._torch.relu(x)
        
        self.register_vendor_kernel("relu", rocm_relu, "rocm")
        
        def rocm_gelu(x):
            if not isinstance(x, self._torch.Tensor):
                x = self._torch.tensor(x, dtype=self._torch.float32)
            x = x.cuda()
            return self._torch.nn.functional.gelu(x)
        
        self.register_vendor_kernel("gelu", rocm_gelu, "rocm")
        
        logger.debug(f"[ROCm] Registered {len(self._vendor_kernels)} vendor kernels")
