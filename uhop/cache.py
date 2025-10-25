# uhop/cache.py
import os
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

CACHE_DIR = Path(os.path.expanduser("~")) / ".uhop_mvp_cache"
CACHE_VERSION = 1


class UhopCache:
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.cache_dir / "index.json"
        if not self.index_file.exists():
            self._write_index({"_meta": {"cache_version": CACHE_VERSION}})

    def _read_index(self) -> Dict[str, Any]:
        try:
            return json.loads(self.index_file.read_text())
        except Exception:
            return {"_meta": {"cache_version": CACHE_VERSION}}

    def _write_index(self, data: Dict[str, Any]):
        self.index_file.write_text(json.dumps(data, indent=2))

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        idx = self._read_index()
        return idx.get(key)

    def all(self) -> Dict[str, Any]:
        """Return all cache records (excluding meta)."""
        idx = self._read_index()
        return {k: v for k, v in idx.items() if k != "_meta"}

    def meta(self) -> Dict[str, Any]:
        """Return cache metadata (version, etc.)."""
        idx = self._read_index()
        m = idx.get("_meta")
        return m if isinstance(m, dict) else {"cache_version": CACHE_VERSION}

    def set(self, key: str, metadata: Dict[str, Any]):
        idx = self._read_index()
        now = datetime.utcnow().isoformat() + "Z"
        enriched: Dict[str, Any] = {**metadata}
        # Attempt to enrich with device/hardware hints
        hw_meta = enriched.get("hardware")
        hw = hw_meta if isinstance(hw_meta, dict) else None
        if hw:
            # compose a simple device hint string for filtering
            vendor = hw.get("vendor") or hw.get("details", {}).get("vendor")
            kind = hw.get("kind")
            name = hw.get("name")
            dev_hint = "|".join(str(x) for x in (kind, vendor, name) if x)
            if dev_hint:
                enriched.setdefault("device_hint", dev_hint)
            # driver info if present in details
            details = hw.get("details") or {}
            if isinstance(details, dict):
                # torch/opencl versions if present
                for k in ("torch", "platform", "device_version"):
                    if k in details:
                        enriched.setdefault("driver_info", details)
                        break
        # Compute source hash if path to file exists
        p = enriched.get("path")
        if isinstance(p, str) and p:
            try:
                fp = Path(p)
                if fp.exists() and fp.is_file():
                    data = fp.read_bytes()
                    enriched.setdefault(
                        "source_hash", hashlib.sha256(data).hexdigest()
                    )
            except Exception:
                pass
        # Timestamp and version
        enriched["_cached_at"] = now
        enriched["_cache_version"] = CACHE_VERSION
        idx[key] = enriched
        self._write_index(idx)

    # Invalidation helpers
    def invalidate_all(self) -> int:
        """Remove all cache entries and return number of removed items."""
        idx = self._read_index()
        n = len([k for k in idx.keys() if k != "_meta"])
        self._write_index({"_meta": {"cache_version": CACHE_VERSION}})
        return n

    def invalidate_device(self, query: str) -> int:
        """Remove entries with device match (substring, case-insensitive).

        Returns number of removed entries.
        """
        idx = self._read_index()
        if not query:
            return 0
        q = str(query).lower()
        removed = 0
        keys = [k for k in idx.keys() if k != "_meta"]
        for k in keys:
            v = idx.get(k)
            if not isinstance(v, dict):
                continue
            hit = False
            hint = str(v.get("device_hint", "")).lower()
            if q in hint:
                hit = True
            else:
                hw = v.get("hardware")
                if isinstance(hw, dict):
                    for field in ("vendor", "kind", "name"):
                        val = str(hw.get(field, "")).lower()
                        if q in val:
                            hit = True
                            break
            if hit:
                del idx[k]
                removed += 1
        self._write_index(idx)
        return removed

    def invalidate_backend(self, backend: str) -> int:
        """Remove entries for a specific backend (exact match)."""
        idx = self._read_index()
        if not backend:
            return 0
        removed = 0
        keys = [k for k in idx.keys() if k != "_meta"]
        for k in keys:
            v = idx.get(k)
            if isinstance(v, dict) and v.get("backend") == backend:
                del idx[k]
                removed += 1
        self._write_index(idx)
        return removed

    def path_for(self, name: str) -> str:
        return str(self.cache_dir / name)

    def clear(self):
        """Remove all cache entries (keeps meta)."""
        self._write_index({"_meta": {"cache_version": CACHE_VERSION}})

    def delete(self, key: str):
        idx = self._read_index()
        if key in idx:
            del idx[key]
            self._write_index(idx)


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

    def _key(
        self,
        backend: str,
        op: str,
        kernel_name: str,
        device: str,
        shape_key: str,
    ) -> str:
        return f"{backend}|{op}|{kernel_name}|{device}|{shape_key}"

    def get_lsz(
        self,
        backend: str,
        op: str,
        kernel_name: str,
        device: str,
        shape_key: str,
    ) -> Optional[Tuple[int, ...]]:
        idx = self._read()
        k = self._key(backend, op, kernel_name, device, shape_key)
        v = idx.get(k)
        if isinstance(v, dict) and "lsz" in v and isinstance(v["lsz"], list):
            try:
                return tuple(int(x) for x in v["lsz"])  # type: ignore
            except Exception:
                return None
        return None

    def set_lsz(
        self,
        backend: str,
        op: str,
        kernel_name: str,
        device: str,
        shape_key: str,
        lsz: List[int],
    ):
        idx = self._read()
        k = self._key(backend, op, kernel_name, device, shape_key)
        idx[k] = {"lsz": list(lsz)}
        self._write(idx)

    # Generic parameter helpers for storing additional autotune metadata
    def get_params(
        self,
        backend: str,
        op: str,
        kernel_name: str,
        device: str,
        shape_key: str,
    ) -> Optional[Dict[str, Any]]:
        idx = self._read()
        k = self._key(backend, op, kernel_name, device, shape_key)
        v = idx.get(k)
        if isinstance(v, dict):
            return v
        return None

    def set_params(
        self,
        backend: str,
        op: str,
        kernel_name: str,
        device: str,
        shape_key: str,
        params: Dict[str, Any],
    ):
        idx = self._read()
        k = self._key(backend, op, kernel_name, device, shape_key)
        # merge with existing if present
        cur = idx.get(k)
        if isinstance(cur, dict):
            merged = {**cur, **params}
        else:
            merged = {**params}
        idx[k] = merged
        self._write(idx)


class KernelRegistry:
    """
    Persist compiled kernel artifacts and provide a process-wide registry.
    For OpenCL, we optionally store program binaries keyed by
    (device_name, source_hash).
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

    def save_opencl_binary(
        self, device: str, source_hash: str, binaries: List[bytes]
    ) -> str:
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

    def load_opencl_binary(
        self, device: str, source_hash: str
    ) -> Optional[str]:
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
    Very simple OpenCL buffer pool keyed by (ctx_id, flags, size).
    Exact-size match only.
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

