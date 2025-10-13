# uhop/backends/torch_backend.py
"""
Torch backend wrapper that will run operations on the best torch device available:
- CUDA if available (works for CUDA and ROCm-built PyTorch)
- MPS (Apple) if available
- CPU fallback if neither
This wrapper returns NumPy arrays for UHOP consumers.
"""
import numpy as np

def _torch_device_preference():
    try:
        import torch
    except Exception:
        return None
    # prefer CUDA (works for ROCm-built torch too)
    try:
        if torch.cuda.is_available():
            return torch.device("cuda")
    except Exception:
        pass
    # prefer MPS on macOS
    try:
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
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

def torch_has_accelerator() -> bool:
    try:
        import torch
    except Exception:
        return False
    try:
        if torch.cuda.is_available():
            return True
    except Exception:
        pass
    try:
        return getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    except Exception:
        return False

def torch_matmul(a, b):
    import torch
    dev = _torch_device_preference()
    ta = torch.from_numpy(np.array(a)).to(dev)
    tb = torch.from_numpy(np.array(b)).to(dev)
    tr = torch.matmul(ta, tb)
    return tr.cpu().numpy()

def torch_conv2d(input_np, weight_np, stride=1, padding=0):
    """
    input_np: (N, C_in, H, W)
    weight_np: (C_out, C_in, KH, KW)
    """
    import torch
    import torch.nn.functional as F
    dev = _torch_device_preference()
    inp = torch.from_numpy(np.array(input_np)).to(dev)
    w = torch.from_numpy(np.array(weight_np)).to(dev)
    out = F.conv2d(inp, w, stride=stride, padding=padding)
    return out.cpu().numpy()

def torch_relu(x):
    import torch
    dev = _torch_device_preference()
    tx = torch.from_numpy(np.array(x)).to(dev)
    return torch.relu(tx).cpu().numpy()
