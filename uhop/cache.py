# uhop/cache.py
import os
import json
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

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


class UhopAutotune:
    """
    Persist simple autotune parameters (like OpenCL local sizes) keyed by
    (backend, op, kernel_name, device, shape_key).
    """
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.file = self.cache_dir / "autotune.json"
        if not self.file.exists():
            self._write({})

    def _read(self) -> Dict[str, Any]:
        try:
            return json.loads(self.file.read_text())
        except Exception:
            return {}

    def _write(self, data: Dict[str, Any]):
        self.file.write_text(json.dumps(data, indent=2))

    def _key(self, backend: str, op: str, kernel_name: str, device: str, shape_key: str) -> str:
        return f"{backend}|{op}|{kernel_name}|{device}|{shape_key}"

    def get_lsz(self, backend: str, op: str, kernel_name: str, device: str, shape_key: str) -> Optional[Tuple[int, ...]]:
        idx = self._read()
        k = self._key(backend, op, kernel_name, device, shape_key)
        v = idx.get(k)
        if isinstance(v, dict) and "lsz" in v and isinstance(v["lsz"], list):
            try:
                return tuple(int(x) for x in v["lsz"])  # type: ignore
            except Exception:
                return None
        return None

    def set_lsz(self, backend: str, op: str, kernel_name: str, device: str, shape_key: str, lsz: List[int]):
        idx = self._read()
        k = self._key(backend, op, kernel_name, device, shape_key)
        idx[k] = {"lsz": list(lsz)}
        self._write(idx)


class KernelRegistry:
    """
    Persist compiled kernel artifacts and provide a process-wide registry.
    For OpenCL, we optionally store program binaries keyed by (device_name, source_hash).
    Note: OpenCL binaries may be vendor/driver specific; fall back gracefully.
    """
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.meta_file = self.cache_dir / "kernels.json"
        if not self.meta_file.exists():
            self._write({})

    def _read(self) -> Dict[str, Any]:
        try:
            return json.loads(self.meta_file.read_text())
        except Exception:
            return {}

    def _write(self, data: Dict[str, Any]):
        self.meta_file.write_text(json.dumps(data, indent=2))

    def _key(self, backend: str, device: str, source_hash: str) -> str:
        return f"{backend}|{device}|{source_hash}"

    def save_opencl_binary(self, device: str, source_hash: str, binaries: List[bytes]) -> str:
        idx = self._read()
        key = self._key("opencl", device, source_hash)
        out_path = self.cache_dir / f"ocl_{source_hash}.bin"
        # Store the first binary blob
        try:
            out_path.write_bytes(binaries[0])
            idx[key] = {"path": str(out_path)}
            self._write(idx)
            return str(out_path)
        except Exception:
            return ""

    def load_opencl_binary(self, device: str, source_hash: str) -> Optional[str]:
        idx = self._read()
        key = self._key("opencl", device, source_hash)
        rec = idx.get(key)
        if rec and isinstance(rec, dict):
            p = rec.get("path")
            if p and Path(p).exists():
                return p
        return None


class OpenCLBufferPool:
    """
    Very simple OpenCL buffer pool keyed by (ctx_id, flags, size). Exact-size match only.
    """
    def __init__(self):
        self._pool: Dict[Tuple[int, int, int], Any] = {}

    def get(self, ctx, size: int, flags: int):
        key = (id(ctx), int(flags), int(size))
        buf = self._pool.get(key)
        if buf is not None:
            return buf
        # Create a new buffer; caller is responsible for writing data
        import pyopencl as cl  # type: ignore
        buf = cl.Buffer(ctx, flags, size)
        self._pool[key] = buf
        return buf

# singleton buffer pool
OPENCL_BUFFER_POOL = OpenCLBufferPool()
