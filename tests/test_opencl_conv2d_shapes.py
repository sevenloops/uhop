import numpy as np
import pytest

from uhop.backends.opencl_backend import is_opencl_available, opencl_conv2d

pytestmark = pytest.mark.skipif(not is_opencl_available(), reason="OpenCL not available")


def conv2d_numpy_valid(x, w, stride=1, padding=0):
    # x: (N,C,H,W), w: (Cout,Cin,KH,KW); stride/padding limited to 1/0 for this test
    assert stride == 1 and padding == 0
    N, C, H, W = x.shape
    Cout, Cin, KH, KW = w.shape
    assert Cin == C
    outH = H - KH + 1
    outW = W - KW + 1
    out = np.zeros((N, Cout, outH, outW), dtype=np.float32)
    for n in range(N):
        for co in range(Cout):
            for y in range(outH):
                for x0 in range(outW):
                    s = 0.0
                    for ci in range(C):
                        for ky in range(KH):
                            for kx in range(KW):
                                s += x[n, ci, y + ky, x0 + kx] * w[co, ci, ky, kx]
                    out[n, co, y, x0] = s
    return out


@pytest.mark.parametrize(
    "NCHW_Cout_K",
    [
        ((1, 1, 8, 8), 1, (3, 3)),
        ((1, 3, 16, 16), 4, (3, 3)),
        ((2, 2, 9, 9), 3, (5, 5)),
    ],
)
def test_opencl_conv2d_various_shapes(NCHW_Cout_K):
    (N, C, H, W), Cout, (KH, KW) = NCHW_Cout_K
    x = np.random.randn(N, C, H, W).astype(np.float32)
    w = np.random.randn(Cout, C, KH, KW).astype(np.float32)

    expected = conv2d_numpy_valid(x, w, stride=1, padding=0)
    got = opencl_conv2d(x, w, stride=1, padding=0)

    assert got.shape == expected.shape
    assert np.allclose(got, expected, atol=1e-3)
