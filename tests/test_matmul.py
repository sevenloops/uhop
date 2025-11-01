# tests/test_matmul.py
import numpy as np

from uhop import UHopOptimizer

hop = UHopOptimizer()


@hop.optimize("matmul")
def matmul_np(A, B):
    return np.array(A) @ np.array(B)


def test_small_matmul():
    A = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    B = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    R = matmul_np(A, B)
    expected = np.array([[19.0, 22.0], [43.0, 50.0]], dtype=np.float32)
    assert np.allclose(R, expected), f"expected {expected}, got {R}"
