# Enhanced IR (v0.1.0)

This document describes the enhanced Intermediate Representation (IR) that includes versioning, memory spaces, layouts, and stable hashing.

## Overview

The enhanced IR builds upon the MVP IR with additional features needed for multi-backend support, AI kernel generation, and reproducible dataset records.

## New Features

### 1. Versioning

All IR operations now include a version field for forward compatibility:

```python
from uhop.ir.ir_enhanced import IR_VERSION

print(f"IR Version: {IR_VERSION}")  # "0.1.0"
```

### 2. Memory Spaces

Support for different memory hierarchies:

```python
from uhop.ir.ir_enhanced import (
    MEMORY_SPACE_GLOBAL, MEMORY_SPACE_LOCAL,
    MEMORY_SPACE_CONSTANT, MEMORY_SPACE_PRIVATE
)

# Usage in tensors
tensor = Tensor(
    name="weights",
    shape=(64, 64),
    memory_space=MEMORY_SPACE_CONSTANT
)
```

### 3. Layout Support

Multiple data layout formats:

```python
from uhop.ir.ir import (
    LAYOUT_ROW_MAJOR, LAYOUT_COL_MAJOR,
    LAYOUT_NCHW, LAYOUT_NHWC
)

# Image data with NCHW layout
image_tensor = Tensor(
    name="input",
    shape=(1, 3, 224, 224),
    layout=LAYOUT_NCHW
)
```

### 4. Strides Support

Explicit stride information for advanced memory access patterns:

```python
tensor = Tensor(
    name="strided",
    shape=(32, 32),
    strides=(1024, 32)  # Custom memory layout
)
```

### 5. Enhanced Schedule

Additional scheduling parameters for optimization:

```python
schedule = Schedule(
    tile_m=16,
    tile_n=16,
    tile_k=8,
    vectorize=4,
    unroll=2  # New parameter
)
```

### 6. Stable Hashing

SHA256-based stable hashing for reproducible kernel identification:

```python
from uhop.ir.ir_enhanced import compute_stable_hash

mm = MatMul(A=Tensor("A", (8, 16)), B=Tensor("B", (16, 4)))
mm_dict = mm.to_dict()
ir_key = compute_stable_hash(mm_dict)
print(f"IR Key: {ir_key}")  # 64-character SHA256 hash
```

## Usage Examples

### Enhanced MatMul with Memory Spaces

```python
from uhop.ir.ir import Tensor, MatMul, Schedule

# Create tensors with specific memory spaces
A = Tensor("A", (64, 128), memory_space=MEMORY_SPACE_GLOBAL)
B = Tensor("B", (128, 32), memory_space=MEMORY_SPACE_LOCAL)

# MatMul operation with enhanced schedule
mm = MatMul(
    A=A,
    B=B,
    schedule=Schedule(tile_m=8, vectorize=4, unroll=2)
)

# Serialize to JSON-compatible dict
mm_dict = mm.to_dict()
print(f"Version: {mm_dict['version']}")
print(f"A memory space: {mm_dict['A']['memory_space']}")

# Compute stable hash
ir_key = compute_stable_hash(mm_dict)
print(f"IR Key: {ir_key}")
```

### Image Processing with Layouts

```python
from uhop.ir.ir_enhanced import Tensor, Relu

# NHWC layout for image processing
input_tensor = Tensor(
    name="input",
    shape=(1, 224, 224, 3),  # Batch, Height, Width, Channels
    layout=LAYOUT_NHWC,
    memory_space=MEMORY_SPACE_GLOBAL
)

# ReLU operation
relu = Relu(X=input_tensor)
```

## Integration with OpenCL Lowering

The OpenCL lowering now respects memory spaces:

```python
from uhop.ir.opencl_lowering import lower_to_opencl

# Lowering will use the correct memory qualifiers
mm = MatMul(
    A=Tensor("A", (64, 128), memory_space=MEMORY_SPACE_LOCAL),
    B=Tensor("B", (128, 32), memory_space=MEMORY_SPACE_GLOBAL)
)

opencl_kernel = lower_to_opencl(mm)
# Generated kernel will use __local for A and __global for B
```

## Backward Compatibility

The enhanced IR maintains full backward compatibility with existing code:

- Existing `Tensor` constructors work without changes
- All existing operations (`MatMul`, `Relu`, `FusedMatMulRelu`) continue to work
- Serialization format is extended but compatible
- Default values ensure existing code continues to work

## Registry Integration

The IR registry now uses SHA256 hashing:

```python
from uhop.ir.registry import compute_ir_key, IRKernelIndex

# Compute stable key
ir_key = compute_ir_key(ir_dict)

# Store in registry
index = IRKernelIndex()
index.set(ir_key, "AMD Radeon RX 6800", source_hash="...")
```

## Testing

Comprehensive tests verify all enhanced features:

- Memory space serialization roundtrip
- Layout preservation
- Stable hash consistency
- Version field inclusion
- Backward compatibility

## Next Steps

This enhanced IR enables:

1. **Multi-backend support** - Memory spaces inform backend-specific optimizations
2. **AI kernel generation** - Rich metadata for better prompt generation
3. **Dataset tracking** - Stable hashing enables reproducible experiments
4. **Performance optimization** - Layout and stride information for memory access patterns

## Migration Guide

For existing code, no changes are required. The enhanced features are opt-in:

```python
# Existing code continues to work
from uhop.ir import Tensor, MatMul  # Still works

# New features available
from uhop.ir.ir import Tensor, compute_stable_hash  # Enhanced version
```

---

**Version**: 0.1.0
**Status**: Complete
**Compatibility**: Full backward compatibility
