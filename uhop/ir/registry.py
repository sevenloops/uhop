from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from ..cache import CACHE_DIR


def compute_ir_key(ir_obj: Dict[str, Any]) -> str:
    """Compute a stable key for an IR descriptor dict.

    Uses JSON dumps with sorted keys and SHA1 hash; returns hex digest.
    """
    norm = json.dumps(ir_obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()


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
            return json.loads(self.path.read_text())
        except Exception:
            return {}

    def _write(self, data: Dict[str, Any]):
        self.path.write_text(json.dumps(data, indent=2))

    def _key(self, ir_key: str, device: str) -> str:
        return f"{ir_key}|{device}"

    def set(self, ir_key: str, device: str, source_hash: str, kernel_name: Optional[str] = None, binary_path: Optional[str] = None):
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
