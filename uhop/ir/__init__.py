"""

UHOP Enhanced IR (v0.1.0) for kernel lowering.







Enhanced IR with versioning, memory spaces, layouts, and stable hashing.
"""

from .ir import (
    IR_VERSION,
    LAYOUT_COL_MAJOR,
    LAYOUT_NCHW,
    LAYOUT_NHWC,
    LAYOUT_ROW_MAJOR,
    MEMORY_SPACE_CONSTANT,
    MEMORY_SPACE_GLOBAL,
    MEMORY_SPACE_LOCAL,
    MEMORY_SPACE_PRIVATE,
    DType,
    FusedMatMulRelu,
    MatMul,
    Op,
    Relu,
    Schedule,
    Tensor,
    compute_stable_hash,
    ir_from_dict,
)

__all__ = [
    "DType",
    "Tensor",
    "Schedule",
    "Op",
    "MatMul",
    "Relu",
    "FusedMatMulRelu",
    "ir_from_dict",
    "compute_stable_hash",
    "IR_VERSION",
    "MEMORY_SPACE_GLOBAL",
    "MEMORY_SPACE_LOCAL",
    "MEMORY_SPACE_PRIVATE",
    "MEMORY_SPACE_CONSTANT",
    "LAYOUT_ROW_MAJOR",
    "LAYOUT_COL_MAJOR",
    "LAYOUT_NCHW",
    "LAYOUT_NHWC",
]
