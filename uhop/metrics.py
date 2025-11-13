"""Metrics snapshot utilities.

Collects a compact KPI snapshot from cache and autotune stores under
~/.uhop_mvp_cache by default. Safe to run on any machine.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_CACHE_DIR = Path(os.path.expanduser("~")) / ".uhop_mvp_cache"


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text()) if path.exists() else {}
    except Exception:
        return {}


def _summarize_backend_selection(index: Dict[str, Any]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for k, v in index.items():
        if k == "_meta" or not isinstance(v, dict):
            continue
        b = v.get("backend")
        if not b:
            continue
        counts[b] = counts.get(b, 0) + 1
    return counts


def _extract_opencl_matmul_perf(auto: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for k, v in auto.items():
        if not isinstance(v, dict):
            continue
        # key format: backend|op|kernel|device|shape
        try:
            backend, op, kernel, device, shape = k.split("|", 4)
        except Exception:
            continue
        if backend != "opencl" or op != "matmul":
            continue
        row = {
            "kernel": kernel,
            "device": device,
            "shape": shape,
            "last_ms": v.get("last_ms"),
            "last_gflops": v.get("last_gflops"),
        }
        rows.append(row)
    return rows


def _extract_opencl_conv2d_perf(auto: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for k, v in auto.items():
        if not isinstance(v, dict):
            continue
        try:
            backend, op, kernel, device, shape = k.split("|", 4)
        except Exception:
            continue
        if backend != "opencl" or op != "conv2d":
            continue
        row = {
            "kernel": kernel,
            "device": device,
            "shape": shape,
            "last_ms": v.get("last_ms"),
            "last_gflops": v.get("last_gflops"),
            # Stage timings recorded in conv2d implementation (best-effort)
            "im2col_ms": v.get("im2col_ms"),
            "gemm_ms": v.get("gemm_ms"),
            "copy_ms": v.get("copy_ms"),
            "chunked": v.get("chunked"),
            "chunk_count": v.get("chunk_count"),
            "var_ms": v.get("var_ms"),
            "mean_ms": v.get("mean_ms"),
            "retune_suggested": v.get("retune_suggested"),
        }
        rows.append(row)
    return rows


def write_kpi_snapshot(output: Path | str, cache_dir: Optional[Path | str] = None) -> Path:
    """Write a KPI snapshot JSON with selection distribution and perf rows.

    Args:
        output: Destination file path.
        cache_dir: Override cache directory; defaults to ~/.uhop_mvp_cache
    Returns:
        Path to written file.
    """
    cdir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    index = _read_json(cdir / "index.json")
    auto = _read_json(cdir / "autotune.json")
    # Build simple per-backend latency histogram from cache index entries
    latencies: Dict[str, List[float]] = {}
    for k, v in (index or {}).items():
        if k == "_meta" or not isinstance(v, dict):
            continue
        backend_name = v.get("backend")
        latency_ms = v.get("latency_ms")
        if backend_name and isinstance(latency_ms, (int, float)):
            latencies.setdefault(backend_name, []).append(float(latency_ms))

    def _quantiles(vals: List[float]) -> Dict[str, float]:
        if not vals:
            return {"p50": None, "p90": None, "p99": None, "mean": None, "count": 0}
        s = sorted(vals)
        n = len(s)

        def _pick(p: float) -> float:
            if n == 1:
                return s[0]
            idx = int(p * (n - 1))
            return float(s[idx])

        import statistics as _stats

        return {
            "p50": _pick(0.50),
            "p90": _pick(0.90),
            "p99": _pick(0.99),
            "mean": float(_stats.mean(s)) if n else None,
            "count": n,
        }

    latency_quantiles = {b: _quantiles(v) for b, v in latencies.items()}
    snapshot = {
        "ts": int(time.time()),
        "cache_dir": str(cdir),
        "backend_selection_counts": _summarize_backend_selection(index or {}),
        "backend_latency_quantiles": latency_quantiles,
        "opencl_matmul": _extract_opencl_matmul_perf(auto or {}),
        "opencl_conv2d": _extract_opencl_conv2d_perf(auto or {}),
    }
    outp = Path(output)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(snapshot, indent=2))
    return outp


def load_kpi_snapshot(path: Path | str) -> Dict[str, Any]:
    return _read_json(Path(path)) or {}


__all__ = ["write_kpi_snapshot", "load_kpi_snapshot"]
