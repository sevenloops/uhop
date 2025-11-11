import os
import numpy as np
import pytest

from uhop import UHopOptimizer
from uhop.backends.opencl_backend import is_opencl_available

pytestmark = pytest.mark.skipif(not is_opencl_available(), reason="OpenCL not available")

OPT = UHopOptimizer()

@OPT.optimize("matmul")
def mm(A, B):
    return np.array(A) @ np.array(B)

@pytest.mark.parametrize("shapes", [
    ((32, 48), (48, 16)),
    ((64, 32), (32, 96)),
    ((7, 5), (5, 11)),
    ((128, 64), (64, 1)),
    ((1, 33), (33, 17)),
])
def test_opencl_matmul_various_shapes(shapes):
    os.environ["UHOP_OPENCL_MATMUL_IMPL"] = "tiled"
    A_shape, B_shape = shapes
    A = np.random.randn(*A_shape).astype(np.float32)
    B = np.random.randn(*B_shape).astype(np.float32)
    expected = A @ B
    got = mm(A, B)
    assert got.shape == expected.shape
    assert np.allclose(got, expected, atol=1e-4)

