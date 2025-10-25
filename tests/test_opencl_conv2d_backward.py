import os
import numpy as np
import pytest

from uhop.backends.opencl_backend import (
    is_opencl_available,
    opencl_conv2d_backward_input,
    opencl_conv2d_backward_weight,
)


pytestmark = pytest.mark.skipif(
    not is_opencl_available(), reason="pyopencl not available"
)


def _conv2d_out_dims(H, W, KH, KW, stride, pad):
    outH = (H + 2 * pad - KH) // stride + 1
    outW = (W + 2 * pad - KW) // stride + 1
    return int(outH), int(outW)


def _ref_conv2d_input_grad(N, Cin, H, W, Cout, KH, KW, outH, outW, stride, pad, grad_out, weight):
    gi = np.zeros((N, Cin, H, W), dtype=np.float32)
    for n in range(N):
        for c in range(Cin):
            for h in range(H):
                for w in range(W):
                    s = 0.0
                    for co in range(Cout):
                        for kh in range(KH):
                            for kw in range(KW):
                                oh_nom = h + pad - kh
                                ow_nom = w + pad - kw
                                if (oh_nom % stride == 0) and (ow_nom % stride == 0):
                                    oh = oh_nom // stride
                                    ow = ow_nom // stride
                                    if 0 <= oh < outH and 0 <= ow < outW:
                                        s += grad_out[n, co, oh, ow] * weight[co, c, kh, kw]
                    gi[n, c, h, w] = s
    return gi


def _ref_conv2d_weight_grad(N, Cin, H, W, Cout, KH, KW, outH, outW, stride, pad, inp, grad_out):
    gw = np.zeros((Cout, Cin, KH, KW), dtype=np.float32)
    for co in range(Cout):
        for c in range(Cin):
            for kh in range(KH):
                for kw in range(KW):
                    s = 0.0
                    for n in range(N):
                        for oh in range(outH):
                            for ow in range(outW):
                                ih = oh * stride + kh - pad
                                iw = ow * stride + kw - pad
                                if 0 <= ih < H and 0 <= iw < W:
                                    s += inp[n, c, ih, iw] * grad_out[n, co, oh, ow]
                    gw[co, c, kh, kw] = s
    return gw


def test_opencl_conv2d_backward_small_random():
    # Small but non-trivial case
    N, Cin, H, W = 1, 2, 6, 6
    Cout, KH, KW = 3, 3, 3
    stride, pad = 1, 1

    rng = np.random.default_rng(0)
    inp = rng.standard_normal((N, Cin, H, W), dtype=np.float32)
    w = rng.standard_normal((Cout, Cin, KH, KW), dtype=np.float32)
    outH, outW = _conv2d_out_dims(H, W, KH, KW, stride, pad)
    go = rng.standard_normal((N, Cout, outH, outW), dtype=np.float32)

    gi_ref = _ref_conv2d_input_grad(N, Cin, H, W, Cout, KH, KW, outH, outW, stride, pad, go, w)
    gw_ref = _ref_conv2d_weight_grad(N, Cin, H, W, Cout, KH, KW, outH, outW, stride, pad, inp, go)

    gi_dev = opencl_conv2d_backward_input((N, Cin, H, W), w, go, stride=stride, padding=pad)
    gw_dev = opencl_conv2d_backward_weight(inp, go, (Cout, Cin, KH, KW), stride=stride, padding=pad)

    assert np.allclose(gi_dev, gi_ref, rtol=1e-4, atol=1e-4)
    assert np.allclose(gw_dev, gw_ref, rtol=1e-4, atol=1e-4)
