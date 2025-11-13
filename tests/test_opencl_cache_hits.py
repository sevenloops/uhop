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


def test_opencl_matmul_cache_hit_roundtrip():
    # Force OpenCL preference first so optimizer selects it (and caches backend=opencl)
    os.environ["UHOP_BACKEND_PREFERENCE"] = "opencl,torch,triton,cpu,numpy"
    os.environ["UHOP_CACHE_PER_SHAPE"] = "1"
    os.environ["UHOP_FORCE_BASELINE"] = "0"
    os.environ["UHOP_OPENCL_MATMUL_IMPL"] = "tiled"  # ensure tiled path is enabled
    A = np.random.randn(128, 96).astype(np.float32)
    B = np.random.randn(96, 64).astype(np.float32)
    # First run (should tune & cache)
    C1 = mm(A, B)
    # Second run: expect same backend chosen and identical result
    C2 = mm(A, B)
    assert np.allclose(C1, C2, atol=1e-5)
    from uhop.cache import CACHE

    rec = CACHE.get("matmul|('numpy', (128, 96), 'float32');('numpy', (96, 64), 'float32')")
    assert rec and rec.get("backend") in ("opencl", "triton", "torch"), "Backend cache not set or unexpected backend"
