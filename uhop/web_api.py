"""
Public-facing minimal HTTP API to support the online demo page.

Endpoints:
  GET  /health              -> { ok: true }
  GET  /info                -> JSON similar to `uhop info --json`
  POST /demo/matmul         -> { stdout, stderr, code, timings }

Security and limits:
  - CORS: Access-Control-Allow-Origin: * (demo only)
  - Binds to 0.0.0.0 by default; use reverse proxy/WAF in production.
  - Strict size/iter limits to avoid CPU abuse.
  - No arbitrary command execution.

Usage (local):
  python -m uhop.web_api --host 0.0.0.0 --port 5824
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse


def _cors_headers(h: BaseHTTPRequestHandler):
    h.send_header("Access-Control-Allow-Origin", "*")
    h.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    h.send_header("Access-Control-Allow-Headers", "Content-Type")


def _info_json() -> dict:
    from .backends import (is_opencl_available, is_torch_available,
                           is_triton_available)
    from .hardware import detect_hardware

    hw = detect_hardware()
    mps_avail = None
    torch_pref = None
    try:
        import torch  # type: ignore

        mps_avail = bool(
            getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        )
        try:
            from .backends.torch_backend import \
                _torch_device_preference as _pref

            dev = _pref()
            torch_pref = getattr(dev, "type", None) if dev is not None else None
        except Exception:
            torch_pref = None
    except Exception:
        mps_avail = None

    return {
        "vendor": hw.vendor,
        "kind": hw.kind,
        "name": hw.name,
        "details": hw.details,
        "torch_available": is_torch_available(),
        "torch_mps_available": mps_avail,
        "torch_preferred_device": torch_pref,
        "triton_available": is_triton_available(),
        "opencl_available": is_opencl_available(),
    }


def _demo_matmul(size: int, iters: int) -> dict:
    import time

    import numpy as np

    from . import optimize

    # Enforce strict limits for online demo
    size = max(16, min(int(size), 256))
    iters = max(1, min(int(iters), 3))

    # Baseline
    def matmul_naive(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        N, M = A.shape
        M2, K = B.shape
        assert M == M2
        C = np.zeros((N, K), dtype=np.float32)
        for i in range(N):
            for k in range(K):
                s = 0.0
                for j in range(M):
                    s += float(A[i, j]) * float(B[j, k])
                C[i, k] = s
        return C

    @optimize("matmul")
    def matmul_np(A, B):
        return np.array(A) @ np.array(B)

    rng = np.random.default_rng(0)
    A = rng.random((size, size), dtype=np.float32)
    B = rng.random((size, size), dtype=np.float32)

    # Warmup
    _ = matmul_np(A, B)

    def _med(run, iters=iters):
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            run()
            times.append(time.perf_counter() - t0)
        return float(np.median(times))

    t_uhop = _med(lambda: matmul_np(A, B))
    t_naive = _med(lambda: matmul_naive(A, B), iters=1)
    won = t_uhop < t_naive

    stdout = (
        "UHOP (optimized over naive): {:.6f} s\n".format(t_uhop)
        + "Naive Python baseline     : {:.6f} s\n".format(t_naive)
        + ("UHOP wins \u2705\n" if won else "Baseline was faster in this config.\n")
    )
    return {
        "code": 0,
        "stderr": "",
        "stdout": stdout,
        "timings": {"uhop": t_uhop, "naive": t_naive, "uhop_won": won},
        "size": size,
        "iters": iters,
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quiet
        return

    def _read_json(self) -> dict:
        try:
            length = int(self.headers.get("Content-Length", "0") or 0)
            body = self.rfile.read(length).decode("utf-8") if length else "{}"
            return json.loads(body)
        except Exception:
            return {}

    def do_OPTIONS(self):  # noqa: N802
        self.send_response(204)
        _cors_headers(self)
        self.end_headers()

    def do_GET(self):  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self.send_response(200)
            _cors_headers(self)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"ok": True}).encode("utf-8"))
            return
        if parsed.path == "/info":
            self.send_response(200)
            _cors_headers(self)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(_info_json()).encode("utf-8"))
            return
        self.send_response(404)
        _cors_headers(self)
        self.end_headers()

    def do_POST(self):  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/demo/matmul":
            data = self._read_json()
            try:
                size = int(data.get("size", 128))
                iters = int(data.get("iters", 2))
            except Exception:
                size, iters = 128, 2
            resp = _demo_matmul(size, iters)
            self.send_response(200)
            _cors_headers(self)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode("utf-8"))
            return
        self.send_response(404)
        _cors_headers(self)
        self.end_headers()


def run(host: str = "0.0.0.0", port: int = 5824):
    server = HTTPServer((host, int(port)), Handler)
    print(f"[UHOP][WebAPI] Listening on http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--host", type=str, default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5824)
    args = ap.parse_args()
    run(args.host, args.port)
