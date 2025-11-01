import numpy as np
import pytest

try:
    import torch
    import torch.nn.functional as F

    TORCH = True
except Exception:
    TORCH = False

from uhop.backends import is_opencl_available, opencl_conv2d


@pytest.mark.skipif(not is_opencl_available(), reason="OpenCL not available")
@pytest.mark.parametrize(
    "N,Cin,H,W,Cout,KH,KW,stride,pad",
    [
        (1, 3, 32, 32, 4, 3, 3, 1, 1),
        (2, 8, 64, 64, 16, 3, 3, 2, 1),
        (1, 1, 28, 28, 1, 5, 5, 1, 2),
    ],
)
def test_conv2d_correctness(N, Cin, H, W, Cout, KH, KW, stride, pad):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((N, Cin, H, W), dtype=np.float32)
    w = rng.standard_normal((Cout, Cin, KH, KW), dtype=np.float32)
    out = opencl_conv2d(x, w, stride=stride, padding=pad)
    outH = (H + 2 * pad - KH) // stride + 1
    outW = (W + 2 * pad - KW) // stride + 1
    if TORCH:
        ref = F.conv2d(
            torch.from_numpy(x), torch.from_numpy(w), stride=stride, padding=pad
        ).numpy()
    else:
        ref = np.zeros((N, Cout, outH, outW), dtype=np.float32)
        for n in range(N):
            for co in range(Cout):
                for y in range(outH):
                    for z in range(outW):
                        s = 0.0
                        for ci in range(Cin):
                            for r in range(KH):
                                for s2 in range(KW):
                                    iy = y * stride - pad + r
                                    ix = z * stride - pad + s2
                                    if 0 <= iy < H and 0 <= ix < W:
                                        s += x[n, ci, iy, ix] * w[co, ci, r, s2]
                        ref[n, co, y, z] = s
    err = float(np.max(np.abs(out - ref)))
    assert err < 1e-3
