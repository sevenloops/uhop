"""
UHOP Torch Shim (MVP)
---------------------

Convenience wrappers that route selected ops through UHOP's multi-backend
optimizer and policy, returning PyTorch tensors by default when inputs are
torch tensors.

Usage:
    import torch
    from uhop.torch_shim import matmul, relu

    a = torch.randn(64, 128, device='cuda' if torch.cuda.is_available() else 'cpu')
    b = torch.randn(128, 32, device=a.device)
    y = matmul(a, b)  # UHOP will select the best available backend
    z = relu(y)
"""
from __future__ import annotations

from typing import Any


def _is_torch_tensor(x: Any) -> bool:
    try:
        import torch  # type: ignore

        return isinstance(x, torch.Tensor)
    except Exception:
        return False


def _to_numpy(x: Any):
    try:
        import torch  # type: ignore

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    try:
        import numpy as np
    except Exception:  # pragma: no cover
        raise RuntimeError("numpy not available")
    if isinstance(x, np.ndarray):
        return x
    return np.array(x)


def _maybe_to_torch_like(ref: Any, x):
    try:
        import torch  # type: ignore

        if isinstance(ref, torch.Tensor):
            return torch.as_tensor(x, device=ref.device, dtype=ref.dtype)
    except Exception:
        pass
    return x


def matmul(a, b):
    """UHOP-optimized matrix multiplication.

    Falls back to torch.matmul if UHOP policy is unavailable.
    """
    try:
        from .backends.base import get_backend_manager  # type: ignore
        from .policy import BackendPolicy  # type: ignore
        from .cache import UhopCache  # type: ignore

        mgr = get_backend_manager()
        # Ensure backends are registered lazily
        try:
            from .backends.registry import ensure_default_backends_registered

            ensure_default_backends_registered()
        except Exception:
            pass
        pol = BackendPolicy(mgr, UhopCache())
        sel = pol.select("matmul", (a, b), {})
        if sel is not None:
            out = sel.run(a, b)
            return out
    except Exception:
        pass
    # Fallback: torch or numpy
    if _is_torch_tensor(a) or _is_torch_tensor(b):
        import torch  # type: ignore

        return torch.matmul(a, b)

    return _to_numpy(a) @ _to_numpy(b)


def relu(x):
    """UHOP-optimized ReLU."""
    try:
        from .backends.base import get_backend_manager  # type: ignore
        from .policy import BackendPolicy  # type: ignore
        from .cache import UhopCache  # type: ignore

        mgr = get_backend_manager()
        try:
            from .backends.registry import ensure_default_backends_registered

            ensure_default_backends_registered()
        except Exception:
            pass
        pol = BackendPolicy(mgr, UhopCache())
        sel = pol.select("relu", (x,), {})
        if sel is not None:
            return sel.run(x)
    except Exception:
        pass
    # Fallback
    if _is_torch_tensor(x):
        import torch  # type: ignore

        return torch.relu(x)
    import numpy as np

    return np.maximum(_to_numpy(x), 0)
