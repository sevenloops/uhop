# UHOP Kernel Library

This directory contains low-level GPU kernel implementations for UHOP across multiple backends.

## Directory Structure

```
kernels/
├── __init__.py          # Kernel registry system
├── cuda/                # NVIDIA CUDA kernels (.cu)
├── hip/                 # AMD ROCm/HIP kernels (.hip)
├── metal/               # Apple Metal shaders (.metal)
└── opencl/              # OpenCL kernels (.cl)
```

## Supported Operators

### Linear Algebra

- **matmul** - Matrix multiplication
- **bmm** - Batched matrix multiplication
- **einsum** - Einstein summation (various contraction patterns)

### Convolution

- **conv1d** - 1D convolution
- **conv2d** - 2D convolution
- **conv3d** - 3D convolution

### Attention

- **scaled_dot_product_attention** - Multi-head attention with optional causal masking

### Normalization

- **layernorm** - Layer normalization
- **batchnorm** - Batch normalization (1D and 2D variants)

### Activations

- **relu** - Rectified Linear Unit
- **gelu** - Gaussian Error Linear Unit (approximation and exact)
- **silu** - Sigmoid Linear Unit (Swish)

### Pooling

- **maxpool2d** - 2D max pooling
- **avgpool2d** - 2D average pooling

### Softmax

- **softmax** - Standard softmax with numerical stability
- **logsoftmax** - Log-softmax

### Recurrent

- **rnn** - Vanilla RNN cell
- **lstm** - LSTM cell (4-gate implementation)

### Fused Operations

- **fused_add_layernorm_gelu** - Residual + LayerNorm + GELU (Transformer MLPs)
- **fused_add_layernorm_relu** - Residual + LayerNorm + ReLU
- **fused_add_layernorm_silu** - Residual + LayerNorm + SiLU

## Backend Coverage

| Operator   | CUDA | HIP | Metal | OpenCL |
| ---------- | ---- | --- | ----- | ------ |
| matmul     | ✓    | ✓   | ✓     | ✓      |
| bmm        | ✓    | ✓   | ✓     | ✓      |
| einsum     | ✓    | -   | -     | -      |
| conv1d     | ✓    | -   | -     | ✓      |
| conv2d     | ✓    | ✓   | ✓     | ✓      |
| conv3d     | ✓    | -   | -     | -      |
| attention  | ✓    | -   | -     | -      |
| layernorm  | ✓    | ✓   | ✓     | ✓      |
| batchnorm  | ✓    | -   | -     | -      |
| relu       | ✓    | ✓   | ✓     | ✓      |
| gelu       | ✓    | ✓   | ✓     | ✓      |
| silu       | ✓    | ✓   | ✓     | ✓      |
| maxpool2d  | ✓    | ✓   | ✓     | ✓      |
| avgpool2d  | ✓    | ✓   | ✓     | ✓      |
| softmax    | ✓    | -   | ✓     | ✓      |
| logsoftmax | ✓    | -   | ✓     | ✓      |
| rnn        | ✓    | -   | -     | -      |
| lstm       | ✓    | -   | -     | -      |
| fused_ops  | ✓    | -   | -     | -      |

## Usage

### Using the Kernel Registry

```python
from uhop.kernels import get_kernel_registry, BackendType

registry = get_kernel_registry()

# Get kernel file info
kernel_info = registry.get_kernel_file("matmul", BackendType.CUDA)
print(f"Kernel file: {kernel_info.file_path}")
print(f"Functions: {kernel_info.kernel_names}")

# Load kernel source
source = registry.get_kernel_source("matmul", BackendType.CUDA)

# List available operators for a backend
ops = registry.get_available_operators(BackendType.METAL)
print(f"Metal operators: {ops}")
```

### Compiling Kernels

#### CUDA (.cu)

```bash
nvcc -ptx -o matmul.ptx kernels/cuda/matmul.cu
```

#### HIP (.hip)

```bash
hipcc --genco -o matmul.co kernels/hip/matmul.hip
```

#### Metal (.metal)

```bash
xcrun -sdk macosx metal -c kernels/metal/matmul.metal -o matmul.air
xcrun -sdk macosx metallib matmul.air -o matmul.metallib
```

#### OpenCL (.cl)

```python
import pyopencl as cl
source = open("kernels/opencl/matmul.cl").read()
program = cl.Program(context, source).build()
```

## Kernel Features

### Optimization Techniques Used

1. **Memory Coalescing** - Aligned memory access patterns
2. **Shared Memory** - Tiling for cache reuse (where applicable)
3. **Register Blocking** - Minimize memory traffic
4. **Numerical Stability** - Shifted softmax, Welford's algorithm
5. **Fused Operations** - Reduce kernel launch overhead and memory bandwidth

### Numerical Considerations

- **GELU**: Uses tanh approximation by default (`gelu_kernel`), exact version available (`gelu_exact_kernel`)
- **Softmax**: Max-subtraction for numerical stability
- **LayerNorm**: Online variance computation, epsilon for stability
- **Attention**: Supports causal masking for autoregressive models

## Performance Notes

- CUDA kernels are optimized for NVIDIA Tensor Cores where applicable
- HIP kernels target AMD CDNA/RDNA architectures
- Metal kernels leverage Apple Silicon unified memory
- OpenCL kernels are portable but may need tuning per device

## Adding New Kernels

1. Create kernel file in appropriate backend directory
2. Register in `__init__.py` registry
3. Add tests in `tests/`
4. Update this README

Example:

```python
# In __init__.py
kernels = [
    ("my_op", "my_op.cu", ["my_op_kernel"]),
]
```

## License

Part of the UHOP project. See repository root for license information.
