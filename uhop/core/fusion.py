"""
Simple kernel fusion utilities.

These are functional compositions that allow us to fuse operations at the call level
when a single fused kernel isn't available yet on a backend.
"""
from typing import Callable, Any


def fuse_kernels(conv_fn: Callable[..., Any], relu_fn: Callable[..., Any]) -> Callable[..., Any]:
    """
    Return a callable that runs relu(conv(*args, **kwargs)).
    This enables easy fused execution paths for backends that don't yet have fused kernels.
    """
    def _fused(*args, **kwargs):
        return relu_fn(conv_fn(*args, **kwargs))

    return _fused
