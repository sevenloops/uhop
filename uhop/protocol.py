"""Protocol schema validators for UHOP v0.1.

Provides lightweight dataclass representations and validation helpers for
incoming/outgoing messages. These are intentionally minimal (not full JSON
schema) to keep runtime overhead low and avoid large deps.

Validation strategy:
  * Ensure required top-level fields exist (`v`, `type`).
  * Enforce protocol version match (currently "0.1").
  * For `request` envelopes: require `id` (str), `action` (known set), optional `params` (dict).
  * For `response` envelopes: require `id`, bool `ok`, and either `data` (dict) or `error` (str).
  * For `hello`: require `agent` (str) and `version` (str).
  * For `log`: require `level` (str) and `line` (str).
  * Reject unexpected types or structural mismatches.

Extension: New actions can be appended to `KNOWN_ACTIONS`; additive fields are
ignored unless they conflict with mandatory ones.

Usage: backend websocket server can call `validate_incoming(msg)` before
processing. If invalid, it returns (False, reason). For responses generated
server-side, `build_error_response(id, reason)` ensures schema compliance.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

PROTOCOL_VERSION = "0.1"
KNOWN_ACTIONS = {"info", "benchmark", "compile_kernel", "validate", "cache_list", "cache_set", "cache_delete", "run_demo", "generate_kernel"}


@dataclass
class ValidationResult:
    ok: bool
    reason: str | None = None


def _require_type(obj: Dict[str, Any], field: str, t) -> Tuple[bool, str | None]:
    v = obj.get(field)
    if isinstance(v, t):
        return True, None
    return False, f"Field '{field}' must be {t.__name__}"


def validate_incoming(obj: Dict[str, Any]) -> ValidationResult:
    if not isinstance(obj, dict):
        return ValidationResult(False, "Message must be JSON object")
    v = obj.get("v")
    if v != PROTOCOL_VERSION:
        return ValidationResult(False, f"Unsupported protocol version '{v}'")
    mtype = obj.get("type")
    if not isinstance(mtype, str):
        return ValidationResult(False, "Missing or invalid 'type'")
    if mtype == "hello":
        ok, reason = _require_type(obj, "agent", str)
        if not ok:
            return ValidationResult(False, reason)
        ok, reason = _require_type(obj, "version", str)
        if not ok:
            return ValidationResult(False, reason)
        return ValidationResult(True)
    if mtype == "log":
        for f, t in ("level", str), ("line", str):
            ok, reason = _require_type(obj, f, t)
            if not ok:
                return ValidationResult(False, reason)
        return ValidationResult(True)
    if mtype == "request":
        ok, reason = _require_type(obj, "id", str)
        if not ok:
            return ValidationResult(False, reason)
        ok, reason = _require_type(obj, "action", str)
        if not ok:
            return ValidationResult(False, reason)
        action = obj.get("action")
        if action not in KNOWN_ACTIONS:
            return ValidationResult(False, f"Unknown action '{action}'")
        params = obj.get("params")
        if params is not None and not isinstance(params, dict):
            return ValidationResult(False, "'params' must be object if present")
        return ValidationResult(True)
    if mtype == "response":
        ok, reason = _require_type(obj, "id", str)
        if not ok:
            return ValidationResult(False, reason)
        if not isinstance(obj.get("ok"), bool):
            return ValidationResult(False, "Field 'ok' must be bool")
        if obj["ok"]:
            if "data" not in obj:
                return ValidationResult(False, "Successful response missing 'data'")
            if not isinstance(obj.get("data"), dict):
                return ValidationResult(False, "'data' must be object when ok=true")
        else:
            if "error" not in obj:
                return ValidationResult(False, "Error response missing 'error'")
            if not isinstance(obj.get("error"), str):
                return ValidationResult(False, "'error' must be string when ok=false")
        return ValidationResult(True)
    return ValidationResult(False, f"Unknown message type '{mtype}'")


def build_error_response(rid: str, reason: str) -> Dict[str, Any]:
    return {"v": PROTOCOL_VERSION, "type": "response", "id": rid, "ok": False, "error": reason}


def build_ok_response(rid: str, data: Dict[str, Any]) -> Dict[str, Any]:
    return {"v": PROTOCOL_VERSION, "type": "response", "id": rid, "ok": True, "data": data}


__all__ = [
    "ValidationResult",
    "validate_incoming",
    "build_error_response",
    "build_ok_response",
    "PROTOCOL_VERSION",
    "KNOWN_ACTIONS",
]
