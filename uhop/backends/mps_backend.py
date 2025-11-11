"""
uhop/backends/mps_backend.py

Thin wrappers over torch to explicitly target Apple's Metal Performance
Shaders (MPS) device when available. These helpers are optional convenience
facades; the main torch backend already prefers CUDA > MPS > CPU.
"""

from __future__ import annotations

from typing import Any

try:
    import torch  # type: ignore

    _TORCH_OK = True
except Exception:  # pragma: no cover - torch not available in minimal envs
    torch = None
    _TORCH_OK = False


def is_mps_available() -> bool:
    if not _TORCH_OK:
        return False
    try:  # pragma: no cover - depends on host
        return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except Exception:
        return False


def _to_tensor(x: Any, device: str):
    if _TORCH_OK and isinstance(x, torch.Tensor):
        return x.to(device)
    # Prefer as_tensor to avoid unnecessary copy when possible
    return torch.as_tensor(x, device=device) if _TORCH_OK else x


def mps_matmul(A: Any, B: Any):
    """Matrix multiply on MPS explicitly (falls back to CPU if unavailable)."""
    if not _TORCH_OK:
        raise RuntimeError("torch not available for MPS backend")
    dev = "mps" if is_mps_available() else "cpu"
    a = _to_tensor(A, dev)
    b = _to_tensor(B, dev)
    out = a @ b
    return out


def mps_relu(X: Any):
    if not _TORCH_OK:
        raise RuntimeError("torch not available for MPS backend")
    dev = "mps" if is_mps_available() else "cpu"
    x = _to_tensor(X, dev)
    return torch.relu(x)


def mps_conv2d(x: Any, w: Any, stride: int = 1, padding: int = 0):
    if not _TORCH_OK:
        raise RuntimeError("torch not available for MPS backend")
    import torch.nn.functional as F  # type: ignore

    dev = "mps" if is_mps_available() else "cpu"
    xt = _to_tensor(x, dev)
    wt = _to_tensor(w, dev)
    return F.conv2d(xt, wt, stride=stride, padding=padding)
