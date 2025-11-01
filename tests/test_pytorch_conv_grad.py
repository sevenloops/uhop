# tests/test_pytorch_conv_grad.py
import pytest
import torch
import torch.nn.functional as F

from uhop.pytorch_wrappers import UHOPConv2DFunction


@pytest.mark.parametrize(
    "N,Cin,H,W,Cout,KH,KW",
    [
        (2, 3, 16, 16, 4, 3, 3),
        (1, 1, 8, 8, 1, 3, 3),
    ],
)
def test_uhoP_conv2d_backward_matches_torch(N, Cin, H, W, Cout, KH, KW):
    torch.manual_seed(0)
    x = torch.randn(N, Cin, H, W, dtype=torch.float32, requires_grad=True)
    w = torch.randn(Cout, Cin, KH, KW, dtype=torch.float32, requires_grad=True)

    # UHOP forward/backward
    y_uh = UHOPConv2DFunction.apply(x, w, 1, 0)
    loss_uh = (y_uh**2).mean()
    gx_uh, gw_uh = torch.autograd.grad(loss_uh, (x, w))

    # Torch direct
    y_t = F.conv2d(x, w, stride=1, padding=0)
    loss_t = (y_t**2).mean()
    gx_t, gw_t = torch.autograd.grad(loss_t, (x, w))

    assert torch.allclose(gx_uh, gx_t, atol=1e-4, rtol=1e-4)
    assert torch.allclose(gw_uh, gw_t, atol=1e-4, rtol=1e-4)
