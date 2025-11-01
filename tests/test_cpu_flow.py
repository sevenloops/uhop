# tests/test_cpu_flow.py
import numpy as np

from uhop.kernels.numpy_kernels import numpy_matmul


def test_numpy_matmul_small():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[5.0, 6.0], [7.0, 8.0]])
    res = numpy_matmul(a, b)
    expected = np.array([[19.0, 22.0], [43.0, 50.0]])
    assert np.allclose(res, expected), f"Expected {expected}, got {res}"
