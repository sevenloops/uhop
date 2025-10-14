# uhop/backends/__init__.py
from .torch_backend import (
    is_torch_available,
    torch_matmul,
    torch_conv2d,
    torch_relu,
)
from .triton_backend import (
    is_triton_available,
    triton_matmul,
    triton_conv2d,
    triton_relu,
)
from .opencl_backend import (
    is_opencl_available,
    opencl_matmul,
    opencl_conv2d,
    opencl_conv2d_relu,
    opencl_relu,
)

__all__ = [
    "is_torch_available", "torch_matmul", "torch_conv2d", "torch_relu",
    "is_triton_available", "triton_matmul", "triton_conv2d", "triton_relu",
    "is_opencl_available", "opencl_matmul", "opencl_conv2d", "opencl_conv2d_relu", "opencl_relu",
]
