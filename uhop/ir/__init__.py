"""
UHOP Minimal IR (v0) for kernel lowering.

This IR intentionally models a tiny subset needed for the MVP:
 - Tensors with shape and dtype
 - Ops: MatMul, Relu, and FusedMatMulRelu
 - Optional Schedule hints (tile sizes / vectorization)

The IR can be serialized to and from JSON-compatible dicts.
"""
from .ir import (
    DType,
    Tensor,
    Schedule,
    Op,
    MatMul,
    Relu,
    FusedMatMulRelu,
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
]
