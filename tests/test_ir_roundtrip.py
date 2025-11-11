
from uhop.ir import Tensor, MatMul, Relu, FusedMatMulRelu, Schedule, ir_from_dict


def test_matmul_roundtrip():
    mm = MatMul(
        A=Tensor("A", (8, 16)),
        B=Tensor("B", (16, 4)),
        schedule=Schedule(tile_m=4, tile_n=4, tile_k=4),
    )
    d = mm.to_dict()
    op2 = ir_from_dict(d)
    assert isinstance(op2, MatMul)
    assert op2.A.shape == (8, 16)
    assert op2.B.shape == (16, 4)
    assert op2.schedule.tile_m == 4


def test_fused_roundtrip():
    fused = FusedMatMulRelu(
        A=Tensor("A", (4, 8)), B=Tensor("B", (8, 2)), schedule=Schedule(vectorize=4)
    )
    d = fused.to_dict()
    op2 = ir_from_dict(d)
    assert isinstance(op2, FusedMatMulRelu)
    assert op2.A.shape == (4, 8)
    assert op2.B.shape == (8, 2)
    assert op2.schedule.vectorize == 4


def test_relu_roundtrip():
    relu = Relu(X=Tensor("X", (10,)))
    d = relu.to_dict()
    op2 = ir_from_dict(d)
    assert isinstance(op2, Relu)
    assert op2.X.shape == (10,)
