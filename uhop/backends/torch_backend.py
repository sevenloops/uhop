# uhop/backends/torch_backend.py
"""
Torch backend wrapper that will run operations on the best torch device available:
- CUDA if available (works for CUDA and ROCm-built PyTorch)
- MPS (Apple) if available
- CPU fallback if neither
This wrapper returns NumPy arrays for UHOP consumers when input is numpy,
but accepts torch.Tensor and returns torch.Tensor if input was tensor to avoid copies.
"""
import numpy as np


def _torch_device_preference():
    try:
        import torch
    except Exception:
        return None
    try:
        if torch.cuda.is_available():
            return torch.device("cuda")
    except Exception:
        pass
    try:
        if (
            getattr(torch.backends, "mps", None) is not None
            and torch.backends.mps.is_available()
        ):
            return torch.device("mps")
    except Exception:
        pass
    return torch.device("cpu")


def is_torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


def _to_torch(x):
    import torch

    if isinstance(x, torch.Tensor):
        return x
    return torch.from_numpy(np.array(x))


def torch_matmul(a, b):
    import torch

    dev = _torch_device_preference()
    ta = _to_torch(a).to(dev)
    tb = _to_torch(b).to(dev)
    tr = torch.matmul(ta, tb)
    # if inputs were torch tensors, return torch Tensor to caller to avoid copies
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return tr
    return tr.cpu().numpy()


def torch_conv2d(input_np, weight_np, stride=1, padding=0):
    """
    input_np: (N, C_in, H, W) numpy or torch tensor
    weight_np: (C_out, C_in, KH, KW) numpy or torch tensor
    """
    import torch
    import torch.nn.functional as F

    dev = _torch_device_preference()
    inp = _to_torch(input_np).to(dev)
    w = _to_torch(weight_np).to(dev)
    out = F.conv2d(inp, w, stride=stride, padding=padding)
    # return torch tensor if inputs were torch; otherwise numpy
    if isinstance(input_np, torch.Tensor) or isinstance(weight_np, torch.Tensor):
        return out
    return out.cpu().numpy()


def torch_relu(x):
    import torch

    dev = _torch_device_preference()
    tx = _to_torch(x).to(dev)
    out = torch.relu(tx)
    if isinstance(x, torch.Tensor):
        return out
    return out.cpu().numpy()
