# tests/test_matmul.py
import numpy as np
from uhop import UHopOptimizer

hop = UHopOptimizer()

@hop.optimize("matmul")
def matmul_np(A, B):
    return np.array(A) @ np.array(B)

def test_small_matmul():
    A = np.array([[1.,2.],[3.,4.]], dtype=np.float32)
    B = np.array([[5.,6.],[7.,8.]], dtype=np.float32)
    R = matmul_np(A, B)
    expected = np.array([[19.,22.],[43.,50.]], dtype=np.float32)
    assert np.allclose(R, expected), f"expected {expected}, got {R}"
