"""Central environment configuration utilities for UHOP.

Provides typed accessors, a registry of known UHOP_* variables, and helper
functions to introspect current effective configuration. This consolidates
scattered os.environ lookups and makes validation/test stubbing easier.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


@dataclass(frozen=True)
class EnvVarMeta:
    name: str
    description: str
    default: Any
    parser: Callable[[str], Any]
    choices: Optional[List[str]] = None
    category: str = "general"


def _parse_bool(val: str) -> bool:
    return str(val).lower() in ("1", "true", "yes", "on")


def _parse_int(val: str) -> int:
    try:
        return int(val)
    except Exception:
        return 0


def _parse_csv(val: str) -> List[str]:
    return [v.strip() for v in str(val).split(",") if v.strip()]


def _identity(val: str) -> str:
    return val


_REGISTRY: Dict[str, EnvVarMeta] = {
    # Process behavior
    "UHOP_LOAD_DOTENV": EnvVarMeta(
        name="UHOP_LOAD_DOTENV",
        description="Load .env files on startup (set 0 to disable)",
        default="1",
        parser=_parse_bool,
        category="general",
    ),
    "UHOP_DISABLE_AUTOTUNE": EnvVarMeta(
        name="UHOP_DISABLE_AUTOTUNE",
        description="Disable autotune reading/writing (use defaults only)",
        default="0",
        parser=_parse_bool,
        category="optimizer",
    ),
    # Backend selection & validation
    "UHOP_BACKEND_PREFERENCE": EnvVarMeta(
        name="UHOP_BACKEND_PREFERENCE",
        description="Comma-separated backend order preference (e.g. opencl,torch,triton,cpu,numpy)",
        default="",
        parser=_identity,
        category="optimizer",
    ),
    "UHOP_FORCE_BASELINE": EnvVarMeta(
        name="UHOP_FORCE_BASELINE",
        description="Force using baseline (NumPy) implementations; bypass optimizer",
        default="0",
        parser=_parse_bool,
        category="optimizer",
    ),
    "UHOP_CACHE_PER_SHAPE": EnvVarMeta(
        name="UHOP_CACHE_PER_SHAPE",
        description="Cache optimizer decisions per shape/dtype instead of per op only",
        default="1",
        parser=_parse_bool,
        category="optimizer",
    ),
    "UHOP_STRICT_VALIDATE": EnvVarMeta(
        name="UHOP_STRICT_VALIDATE",
        description="Stricter AI kernel validation / correctness checks",
        default="0",
        parser=_parse_bool,
        category="validation",
    ),
    # OpenCL matmul / conv controls
    "UHOP_OPENCL_MATMUL_IMPL": EnvVarMeta(
        name="UHOP_OPENCL_MATMUL_IMPL",
        description="Matmul implementation override: naive|tiled|clblast",
        default="naive",
        parser=_identity,
        choices=["naive", "tiled", "clblast"],
        category="opencl",
    ),
    "UHOP_OPENCL_CONV_IMPL": EnvVarMeta(
        name="UHOP_OPENCL_CONV_IMPL",
        description="Conv2D implementation: auto|tiled|im2col_gemm",
        default="auto",
        parser=_identity,
        choices=["auto", "tiled", "im2col_gemm", "im2col"],
        category="opencl",
    ),
    "UHOP_OPENCL_ENABLE_TILED": EnvVarMeta(
        name="UHOP_OPENCL_ENABLE_TILED",
        description="Permit using tiled OpenCL matmul path when selected",
        default="0",
        parser=_parse_bool,
        category="opencl",
    ),
    "UHOP_OPENCL_FORCE_NAIVE": EnvVarMeta(
        name="UHOP_OPENCL_FORCE_NAIVE",
        description="Force naive matmul even if tiled requested",
        default="0",
        parser=_parse_bool,
        category="opencl",
    ),
    "UHOP_OPENCL_DEVICE_INDEX": EnvVarMeta(
        name="UHOP_OPENCL_DEVICE_INDEX",
        description="Select OpenCL device by flattened index",
        default="0",
        parser=_parse_int,
        category="opencl",
    ),
    "UHOP_OPENCL_VEC_CANDIDATES": EnvVarMeta(
        name="UHOP_OPENCL_VEC_CANDIDATES",
        description="Candidate vector widths for OpenCL kernels (e.g. 1,2,4)",
        default="1",
        parser=_parse_csv,
        category="opencl",
    ),
    "UHOP_OPENCL_VALIDATE": EnvVarMeta(
        name="UHOP_OPENCL_VALIDATE",
        description="Enable output validation for tiled kernels",
        default="1",
        parser=_parse_bool,
        category="opencl",
    ),
    "UHOP_OPENCL_FLIP_GWS": EnvVarMeta(
        name="UHOP_OPENCL_FLIP_GWS",
        description="Experimental flip of global work size dimension mapping",
        default="0",
        parser=_parse_bool,
        category="opencl",
    ),
    "UHOP_OPENCL_TILED_DIAG": EnvVarMeta(
        name="UHOP_OPENCL_TILED_DIAG",
        description="Emit diagnostic warnings for tiled matmul tuning/validation",
        default="0",
        parser=_parse_bool,
        category="opencl",
    ),
    "UHOP_OPENCL_TILED_DIAG_STRICT": EnvVarMeta(
        name="UHOP_OPENCL_TILED_DIAG_STRICT",
        description="Assert raw vs backend tiled parity during diagnostics",
        default="0",
        parser=_parse_bool,
        category="opencl",
    ),
    "UHOP_OPENCL_NAIVE_VEC": EnvVarMeta(
        name="UHOP_OPENCL_NAIVE_VEC",
        description="Override naive matmul vector width at fallback",
        default="1",
        parser=_parse_int,
        category="opencl",
    ),
    # AI generation
    "UHOP_OPENAI_MODEL": EnvVarMeta(
        name="UHOP_OPENAI_MODEL",
        description="Default OpenAI model for codegen",
        default="gpt-4o-mini",
        parser=_identity,
        category="ai",
    ),
    "UHOP_SB_BLOCK_NET": EnvVarMeta(
        name="UHOP_SB_BLOCK_NET",
        description="Block network inside AI sandbox runner by default",
        default="1",
        parser=_parse_bool,
        category="ai",
    ),
    "UHOP_AI_DEBUG": EnvVarMeta(
        name="UHOP_AI_DEBUG",
        description="Enable verbose AI codegen debug logging",
        default="0",
        parser=_parse_bool,
        category="ai",
    ),
    # Logging / agent
    "UHOP_LOG_LEVEL": EnvVarMeta(
        name="UHOP_LOG_LEVEL",
        description="Override log verbosity (DEBUG,INFO,WARNING,ERROR)",
        default="INFO",
        parser=_identity,
        category="logging",
    ),
    "UHOP_AGENT_SERVER": EnvVarMeta(
        name="UHOP_AGENT_SERVER",
        description="Local agent websocket server endpoint",
        default="ws://localhost:8787/agent",
        parser=_identity,
        category="agent",
    ),
}


def get(name: str) -> Any:
    meta = _REGISTRY.get(name)
    if not meta:
        return os.environ.get(name)
    raw = os.environ.get(name, str(meta.default))
    try:
        return meta.parser(raw)
    except Exception:
        return meta.default


def as_dict(include_unset: bool = False) -> Dict[str, Any]:
    data = {}
    for k, meta in _REGISTRY.items():
        raw = os.environ.get(k)
        if raw is None and not include_unset:
            continue
        data[k] = get(k)
    return data


def describe() -> List[Dict[str, Any]]:
    info = []
    for meta in _REGISTRY.values():
        info.append(
            {
                "name": meta.name,
                "category": meta.category,
                "default": meta.default,
                "current": get(meta.name),
                "description": meta.description,
                "choices": meta.choices or [],
            }
        )
    return sorted(info, key=lambda x: (x["category"], x["name"]))


__all__ = ["get", "as_dict", "describe", "EnvVarMeta"]

# Runtime overrides registry (set via set()) for introspection.
_OVERRIDES: Dict[str, Any] = {}
_SET_LOCK = threading.Lock()


def set(name: str, value: Any) -> None:
    """Set an environment variable (stringifying value) and record override.

    This centralizes mutation points so fallback logic (e.g., forcing naive
    matmul) can be tracked and surfaced to users via a future `uhop config list`.
    """
    with _SET_LOCK:
        os.environ[name] = str(value)
        _OVERRIDES[name] = value


def overrides() -> Dict[str, Any]:
    return dict(_OVERRIDES)


__all__.extend(["set", "overrides"])
