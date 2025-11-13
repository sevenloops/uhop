from uhop.ir import Tensor, MatMul, Relu, FusedMatMulRelu, Schedule, ir_from_dict
from uhop.ir import MEMORY_SPACE_GLOBAL, MEMORY_SPACE_LOCAL, LAYOUT_ROW_MAJOR, LAYOUT_NCHW


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


def test_tensor_memory_spaces():
    """Test that memory spaces are preserved in roundtrip."""
    tensor = Tensor("test", (32, 32), memory_space=MEMORY_SPACE_LOCAL, layout=LAYOUT_NCHW)
    d = tensor.to_dict()
    tensor2 = Tensor.from_dict(d)
    assert tensor2.memory_space == MEMORY_SPACE_LOCAL
    assert tensor2.layout == LAYOUT_NCHW


def test_op_versioning():
    """Test that version field is included in serialized ops."""
    mm = MatMul(
        A=Tensor("A", (8, 16)),
        B=Tensor("B", (16, 4)),
    )
    d = mm.to_dict()
    assert "version" in d
    assert d["version"] == "0.1.0"


def test_tensor_strides():
    """Test that strides are preserved in roundtrip."""
    tensor = Tensor("test", (32, 32), strides=(1024, 32))
    d = tensor.to_dict()
    tensor2 = Tensor.from_dict(d)
    assert tensor2.strides == (1024, 32)


def test_schedule_unroll():
    """Test that unroll parameter is preserved."""
    schedule = Schedule(tile_m=8, unroll=4)
    d = schedule.to_dict()
    schedule2 = Schedule.from_dict(d)
    assert schedule2.tile_m == 8
    assert schedule2.unroll == 4

