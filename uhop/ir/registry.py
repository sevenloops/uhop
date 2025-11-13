from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from ..cache import CACHE_DIR
from .ir import IR_VERSION, compute_stable_hash


def compute_ir_key(ir_obj: Dict[str, Any]) -> str:
    """Compute a stable key for an IR descriptor dict.

    Uses JSON dumps with sorted keys and SHA256 hash; returns hex digest.
    """
    return compute_stable_hash(ir_obj)


class IRKernelIndex:
    """Persist mapping from (ir_key, device) -> source_hash and optional binary path.

    Stored at ~/.uhop_mvp_cache/ir_kernels.json
    """

    def __init__(self, path: Optional[Path] = None):
        self.path = path or (Path(os.path.expanduser(str(CACHE_DIR))) / "ir_kernels.json")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write({})

    def _read(self) -> Dict[str, Any]:
        try:
            data = json.loads(self.path.read_text())
            # Add version info if not present
            if "version" not in data:
                data["version"] = IR_VERSION
                self._write(data)
            return data
        except Exception:
            return {"version": IR_VERSION}

    def _write(self, data: Dict[str, Any]):
        data["version"] = IR_VERSION
        self.path.write_text(json.dumps(data, indent=2))

    def _key(self, ir_key: str, device: str) -> str:
        return f"{ir_key}|{device}"

    def set(
        self,
        ir_key: str,
        device: str,
        source_hash: str,
        kernel_name: Optional[str] = None,
        binary_path: Optional[str] = None,
    ):
        idx = self._read()
        idx[self._key(ir_key, device)] = {
            "source_hash": source_hash,
            "kernel_name": kernel_name,
            "binary_path": binary_path,
        }
        self._write(idx)

    def get(self, ir_key: str, device: str) -> Optional[Dict[str, Any]]:
        idx = self._read()
        v = idx.get(self._key(ir_key, device))
        return v if isinstance(v, dict) else None

    def get_all_for_device(self, device: str) -> Dict[str, Dict[str, Any]]:
        """Get all entries for a specific device."""
        idx = self._read()
        result = {}
        for key, value in idx.items():
            if key.endswith(f"|{device}") and isinstance(value, dict):
                ir_key = key.split("|")[0]
                result[ir_key] = value
        return result

    def clear_device(self, device: str):
        """Clear all entries for a specific device."""
        idx = self._read()
        keys_to_remove = [k for k in idx.keys() if k.endswith(f"|{device}")]
        for key in keys_to_remove:
            del idx[key]
        self._write(idx)
