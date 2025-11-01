"""
Minimal local HTTP bridge to run whitelisted UHOP CLI commands from the
docs site.

Usage:
  python -m uhop.web_bridge --port 5823
or via CLI:
  uhop web-bridge --port 5823

Security:
  - Listens on 127.0.0.1 only
  - CORS: Access-Control-Allow-Origin: * (local use)
  - Whitelisted commands only: must start with 'uhop' and subcommand in ALLOWED
"""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import List

ALLOWED_SUBCOMMANDS: List[str] = [
    "info",
    "demo",
    "demo-conv2d-relu",
    "cache",
    "ai-generate",
    "ai-generate-fused",
]


def _cors_headers(handler: BaseHTTPRequestHandler):
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type")


def _is_allowed(cmd: str) -> bool:
    cmd = cmd.strip()
    if not cmd or not cmd.startswith("uhop"):
        return False
    # Extract subcommand token (2nd token)
    try:
        parts = shlex.split(cmd)
    except Exception:
        return False
    if len(parts) < 2:
        return False
    sub = parts[1]
    return sub in ALLOWED_SUBCOMMANDS


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quiet
        return

    def do_OPTIONS(self):  # noqa: N802
        self.send_response(204)
        _cors_headers(self)
        self.end_headers()

    def do_GET(self):  # noqa: N802
        if self.path.startswith("/health"):
            self.send_response(200)
            _cors_headers(self)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"ok": True}).encode("utf-8"))
            return
        self.send_response(404)
        _cors_headers(self)
        self.end_headers()

    def do_POST(self):  # noqa: N802
        if self.path != "/run":
            self.send_response(404)
            _cors_headers(self)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", "0") or 0)
        try:
            body = self.rfile.read(length).decode("utf-8") if length else "{}"
            data = json.loads(body)
        except Exception:
            self.send_response(400)
            _cors_headers(self)
            self.end_headers()
            return
        cmd = str(data.get("cmd", "")).strip()
        timeout = int(data.get("timeout", 60))
        if not _is_allowed(cmd):
            self.send_response(400)
            _cors_headers(self)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                json.dumps({"error": "command not allowed"}).encode("utf-8")
            )
            return
        try:
            parts = shlex.split(cmd)
            # Launch via current Python to avoid PATH issues for 'uhop'.
            # Example:
            #   'uhop info --json' ->
            #   [sys.executable, '-m', 'uhop.cli', 'info', '--json']
            pycmd = [sys.executable, "-m", "uhop.cli"] + parts[1:]
            proc = subprocess.run(
                pycmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            resp = {
                "code": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            }
        except subprocess.TimeoutExpired:
            resp = {"code": -1, "stdout": "", "stderr": "timeout"}
        except Exception as e:
            resp = {"code": -1, "stdout": "", "stderr": str(e)}
        self.send_response(200)
        _cors_headers(self)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(resp).encode("utf-8"))


def run_web_bridge(port: int = 5823):
    server = HTTPServer(("127.0.0.1", int(port)), _Handler)
    print(f"[UHOP][WebBridge] Listening on http://127.0.0.1:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5823)
    args = ap.parse_args()
    run_web_bridge(args.port)
