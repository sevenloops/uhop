from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


# Simple dtype alias for MVP
DType = str  # e.g., "f32"


@dataclass
class Tensor:
    name: str
    shape: Tuple[int, ...]
    dtype: DType = "f32"

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "shape": list(self.shape), "dtype": self.dtype}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Tensor":
        return Tensor(
            name=str(d["name"]),
            shape=tuple(int(x) for x in d.get("shape", [])),
            dtype=str(d.get("dtype", "f32")),
        )


@dataclass
class Schedule:
    tile_m: Optional[int] = None
    tile_n: Optional[int] = None
    tile_k: Optional[int] = None
    vectorize: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        if self.tile_m is not None:
            d["tile_m"] = int(self.tile_m)
        if self.tile_n is not None:
            d["tile_n"] = int(self.tile_n)
        if self.tile_k is not None:
            d["tile_k"] = int(self.tile_k)
        if self.vectorize is not None:
            d["vectorize"] = int(self.vectorize)
        return d

    @staticmethod
    def from_dict(d: Optional[Dict[str, Any]]) -> "Schedule":
        if not d:
            return Schedule()
        return Schedule(
            tile_m=(int(d["tile_m"]) if d.get("tile_m") is not None else None),
            tile_n=(int(d["tile_n"]) if d.get("tile_n") is not None else None),
            tile_k=(int(d["tile_k"]) if d.get("tile_k") is not None else None),
            vectorize=(
                int(d["vectorize"]) if d.get("vectorize") is not None else None
            ),
        )


class Op:
    op_type: str = "op"

    def to_dict(self) -> Dict[str, Any]:
        raise NotImplementedError


@dataclass
class MatMul(Op):
    A: Tensor
    B: Tensor
    C: Optional[Tensor] = None
    schedule: Optional[Schedule] = None
    op_type: str = "matmul"

    def infer_output(self) -> Tensor:
        m, k = self.A.shape
        kb, n = self.B.shape
        if k != kb:
            raise ValueError("MatMul shape mismatch: A.cols != B.rows")
        name = self.C.name if self.C else "C"
        return Tensor(name=name, shape=(m, n), dtype=self.A.dtype)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.op_type,
            "A": self.A.to_dict(),
            "B": self.B.to_dict(),
            "C": self.C.to_dict() if self.C else None,
            "schedule": self.schedule.to_dict() if self.schedule else None,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "MatMul":
        A = Tensor.from_dict(d["A"]) if isinstance(d.get("A"), dict) else d["A"]
        B = Tensor.from_dict(d["B"]) if isinstance(d.get("B"), dict) else d["B"]
        C = (
            Tensor.from_dict(d["C"]) if isinstance(d.get("C"), dict) else None
        )
        sched = Schedule.from_dict(d.get("schedule"))
        return MatMul(A=A, B=B, C=C, schedule=sched)


@dataclass
class Relu(Op):
    X: Tensor
    Y: Optional[Tensor] = None
    op_type: str = "relu"

    def infer_output(self) -> Tensor:
        name = self.Y.name if self.Y else "Y"
        return Tensor(name=name, shape=self.X.shape, dtype=self.X.dtype)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.op_type,
            "X": self.X.to_dict(),
            "Y": self.Y.to_dict() if self.Y else None,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Relu":
        X = Tensor.from_dict(d["X"]) if isinstance(d.get("X"), dict) else d["X"]
        Y = (
            Tensor.from_dict(d["Y"]) if isinstance(d.get("Y"), dict) else None
        )
        return Relu(X=X, Y=Y)


@dataclass
class FusedMatMulRelu(Op):
    A: Tensor
    B: Tensor
    Y: Optional[Tensor] = None
    schedule: Optional[Schedule] = None
    op_type: str = "fused_matmul_relu"

    def infer_output(self) -> Tensor:
        m, k = self.A.shape
        kb, n = self.B.shape
        if k != kb:
            raise ValueError("MatMul shape mismatch: A.cols != B.rows")
        name = self.Y.name if self.Y else "Y"
        return Tensor(name=name, shape=(m, n), dtype=self.A.dtype)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.op_type,
            "A": self.A.to_dict(),
            "B": self.B.to_dict(),
            "Y": self.Y.to_dict() if self.Y else None,
            "schedule": self.schedule.to_dict() if self.schedule else None,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "FusedMatMulRelu":
        A = Tensor.from_dict(d["A"]) if isinstance(d.get("A"), dict) else d["A"]
        B = Tensor.from_dict(d["B"]) if isinstance(d.get("B"), dict) else d["B"]
        Y = (
            Tensor.from_dict(d["Y"]) if isinstance(d.get("Y"), dict) else None
        )
        sched = Schedule.from_dict(d.get("schedule"))
        return FusedMatMulRelu(A=A, B=B, Y=Y, schedule=sched)


def ir_from_dict(d: Dict[str, Any]) -> Op:
    """Create an IR Op from a JSON-compatible dict."""
    typ = str(d.get("type", "")).lower()
    if typ == "matmul":
        return MatMul.from_dict(d)
    if typ == "relu":
        return Relu.from_dict(d)
    if typ in ("fused_matmul_relu", "matmul_relu", "fused_matmul+relu"):
        return FusedMatMulRelu.from_dict(d)
    raise ValueError(f"Unknown IR op type: {typ}")
