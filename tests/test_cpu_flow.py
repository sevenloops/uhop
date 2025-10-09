# tests/test_cpu_flow.py
import numpy as np
from uhop.kernels.numpy_kernels import numpy_matmul

def test_numpy_matmul_small():
    a = np.array([[1., 2.], [3., 4.]])
    b = np.array([[5., 6.], [7., 8.]])
    res = numpy_matmul(a, b)
    expected = np.array([[19., 22.], [43., 50.]])
    assert np.allclose(res, expected), f"Expected {expected}, got {res}"
