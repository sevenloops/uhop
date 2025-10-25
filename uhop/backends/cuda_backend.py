"""
CUDA Backend for UHOP.

Provides base CUDA support using cuBLAS and raw CUDA kernels.
"""
from typing import Dict, List, Optional
import logging

from uhop.backends.base import Backend, KernelSource

logger = logging.getLogger(__name__)


class CUDABackend(Backend):
    """
    CUDA backend using cuBLAS for BLAS operations and raw CUDA for custom kernels.
    """
    
    def __init__(self):
        super().__init__("cuda")
        self._torch = None
        self._cublas_available = False
    
    def initialize(self) -> bool:
        """Initialize CUDA backend and check availability."""
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
            
            # Get device info
            self.capabilities.available = True
            self.capabilities.device_count = torch.cuda.device_count()
            
            for i in range(self.capabilities.device_count):
                device_name = torch.cuda.get_device_name(i)
                self.capabilities.device_names.append(device_name)
                
                # Get compute capability
                props = torch.cuda.get_device_properties(i)
                self.capabilities.compute_capability = f"{props.major}.{props.minor}"
                self.capabilities.memory_gb = props.total_memory / (1024**3)
            
            # Check for cuBLAS (always available with CUDA)
            self._cublas_available = True
            self.capabilities.vendor_libs["cublas"] = True
            
            logger.info(
                f"[CUDA] Initialized with {self.capabilities.device_count} device(s), "
                f"compute capability {self.capabilities.compute_capability}"
            )
            
        except ImportError:
            self.capabilities.available = False
            self.capabilities.error_msg = "torch not installed"
            logger.warning("[CUDA] torch not available")
        except Exception as e:
            self.capabilities.available = False
            self.capabilities.error_msg = str(e)
            logger.error(f"[CUDA] Initialization failed: {e}")
        
        self._initialized = True
        return self.capabilities.available
    
    def check_vendor_libs(self) -> Dict[str, bool]:
        """Check cuBLAS availability."""
        return {
            "cublas": self._cublas_available,
        }
    
    def get_supported_ops(self) -> List[str]:
        """Get list of CUDA-supported operators."""
        ops = [
            # Linear algebra
            "matmul", "bmm", "einsum",
            # Convolutions
            "conv1d", "conv2d", "conv3d",
            # Activations
            "relu", "gelu", "silu", "sigmoid", "tanh",
            # Normalization
            "layernorm", "batchnorm", "groupnorm", "rmsnorm",
            # Pooling
            "maxpool2d", "avgpool2d", "adaptiveavgpool2d",
            # Softmax
            "softmax", "logsoftmax",
            # Elementwise
            "add", "mul", "div",
            # Reductions
            "sum", "mean", "max",
            # Attention
            "scaled_dot_product_attention", "multi_head_attention",
            # RNN
            "rnn", "lstm", "gru",
            # Fused ops
            "fused_bias_gelu", "fused_add_layernorm", "fused_matmul_bias_silu", "fused_conv2d_relu",
        ]
        return ops
    
    def _synchronize(self):
        """Synchronize CUDA device."""
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.synchronize()
    
    def _setup_vendor_kernels(self):
        """Set up cuBLAS-based kernels."""
        if not self._cublas_available or self._torch is None:
            return
        
        # Matrix multiplication using cuBLAS via PyTorch
        def cuda_matmul(A, B):
            if not isinstance(A, self._torch.Tensor):
                A = self._torch.tensor(A, dtype=self._torch.float32)
            if not isinstance(B, self._torch.Tensor):
                B = self._torch.tensor(B, dtype=self._torch.float32)
            
            A = A.cuda()
            B = B.cuda()
            return self._torch.matmul(A, B)
        
        self.register_vendor_kernel("matmul", cuda_matmul, "cublas")
        
        # Batched matrix multiplication
        def cuda_bmm(A, B):
            if not isinstance(A, self._torch.Tensor):
                A = self._torch.tensor(A, dtype=self._torch.float32)
            if not isinstance(B, self._torch.Tensor):
                B = self._torch.tensor(B, dtype=self._torch.float32)
            
            A = A.cuda()
            B = B.cuda()
            return self._torch.bmm(A, B)
        
        self.register_vendor_kernel("bmm", cuda_bmm, "cublas")
        
        # Basic activations
        def cuda_relu(x):
            if not isinstance(x, self._torch.Tensor):
                x = self._torch.tensor(x, dtype=self._torch.float32)
            x = x.cuda()
            return self._torch.relu(x)
        
        self.register_vendor_kernel("relu", cuda_relu, "cuda")
        
        def cuda_gelu(x):
            if not isinstance(x, self._torch.Tensor):
                x = self._torch.tensor(x, dtype=self._torch.float32)
            x = x.cuda()
            return self._torch.nn.functional.gelu(x)
        
        self.register_vendor_kernel("gelu", cuda_gelu, "cuda")
        
        logger.debug("[CUDA] Registered cuBLAS vendor kernels")
