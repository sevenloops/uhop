# uhop/cache.py
import os
import json
from pathlib import Path
from typing import Optional, Dict, Any

CACHE_DIR = Path(os.path.expanduser("~")) / ".uhop_mvp_cache"

class UhopCache:
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.cache_dir / "index.json"
        if not self.index_file.exists():
            self._write_index({})

    def _read_index(self) -> Dict[str, Any]:
        try:
            return json.loads(self.index_file.read_text())
        except Exception:
            return {}

    def _write_index(self, data: Dict[str, Any]):
        self.index_file.write_text(json.dumps(data, indent=2))

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        idx = self._read_index()
        return idx.get(key)

    def set(self, key: str, metadata: Dict[str, Any]):
        idx = self._read_index()
        idx[key] = metadata
        self._write_index(idx)

    def path_for(self, name: str) -> str:
        return str(self.cache_dir / name)
