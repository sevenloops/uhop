"""
Simple operation registry for UHOP elementwise/autotune demo.
Holds canonical op names and metadata used by the autotuner/dispatch.
"""
from typing import Callable, Dict, Any
import json
from pathlib import Path

_OPS: Dict[str, Dict[str, Any]] = {}

def register_op(name: str, fallback: Callable = None, *, supports_broadcast: bool = True):
    """Register an op with optional fallback implementation and metadata."""
    _OPS[name] = {
        "name": name,
        "fallback": fallback,
        "supports_broadcast": supports_broadcast,
    }
    return _OPS[name]

def get_op(name: str):
    return _OPS.get(name)

def list_ops():
    return list(_OPS.keys())

def save_registry(path: str):
    p = Path(path)
    p.write_text(json.dumps({k: {"supports_broadcast": v["supports_broadcast"]} for k,v in _OPS.items()}, indent=2))

# Register default simple ops
def _default_add(a, b):
    return a + b

register_op("add", fallback=_default_add, supports_broadcast=True)
register_op("mul", fallback=lambda a,b: a * b, supports_broadcast=True)
