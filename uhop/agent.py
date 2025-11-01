"""
UHOP Local Agent
-----------------
Runs on the user's machine and bridges the hosted portal to local hardware.

Protocol (JSON over WebSocket):
  Client -> Server on connect:
    { "type": "hello", "agent": "uhop-agent", "version": "0.1.0", "token": "<optional>" }

  Server -> Client requests:
    { "id": "uuid", "type": "request", "action": "info" }
    { "id": "uuid", "type": "request", "action": "run_demo", "params": { "size": 128, "iters": 2 } }
    { "id": "uuid", "type": "request", "action": "generate_kernel", "params": { "target": "opencl" } }

  Client -> Server responses:
    { "id": "uuid", "type": "response", "ok": true, "data": { ... } }
    { "id": "uuid", "type": "response", "ok": false, "error": "message" }

  Streaming logs (either direction):
    { "type": "log", "level": "info|warn|error", "line": "..." }

Usage:
  uhop-agent --server ws://localhost:8787/agent --token <optional>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, Optional

try:
    import websocket  # type: ignore
except Exception:  # pragma: no cover - handled at runtime
    websocket = None  # type: ignore


@dataclass
class AgentConfig:
    server: str
    token: Optional[str] = None
    reconnect: bool = True
    reconnect_delay: float = 2.0


def _send(ws, obj: Dict[str, Any]):
    try:
        ws.send(json.dumps(obj))
    except Exception:
        pass


def _info_json() -> Dict[str, Any]:
    # Reuse web_api logic for consistent info
    from .web_api import _info_json as _impl

    return _impl()


def _run_demo(size: int, iters: int) -> Dict[str, Any]:
    from .web_api import _demo_matmul as _impl

    return _impl(size, iters)


def _generate_kernel(target: str = "opencl") -> Dict[str, Any]:
    """Call CLI to trigger AI generation, then read latest file.

    Returns { language, code } or raises on error.
    """
    import glob
    import subprocess
    from pathlib import Path

    # Prefer the same Python that runs the agent
    py = sys.executable or "python"
    root = Path(__file__).resolve().parent.parent

    # Invoke CLI; if OPENAI_API_KEY isn't set, backend/portal should handle fallback messaging.
    cmd = [py, "-m", "uhop.cli", "ai-generate", "matmul", "--target", target]
    p = subprocess.run(
        cmd, cwd=str(root), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if p.returncode != 0:
        raise RuntimeError(
            f"ai-generate failed: {p.stderr.strip() or p.stdout.strip()}"
        )

    gen_dir = root / "uhop" / "generated_kernels"
    files = sorted(
        (Path(f) for f in glob.glob(str(gen_dir / "ai_matmul_*.cl"))),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not files:
        raise RuntimeError("no generated kernels found")
    code = files[0].read_text(encoding="utf-8")
    return {"language": "opencl", "code": code}


def run_agent(cfg: AgentConfig, *, debug: bool = False):
    if websocket is None:
        print(
            "websocket-client not installed. Please install with: pip install websocket-client",
            file=sys.stderr,
        )
        sys.exit(2)

    url = cfg.server

    while True:
        ws = None
        try:
            try_url = url
            if debug:
                print(f"[agent] connecting to {try_url}")
                try:
                    websocket.enableTrace(True)
                except Exception:
                    pass
            try:
                ws = websocket.create_connection(try_url, timeout=5)
            except Exception:
                # IPv6/localhost issues: try 127.0.0.1
                if try_url.startswith("ws://localhost"):
                    try_url = try_url.replace("ws://localhost", "ws://127.0.0.1")
                    if debug:
                        print(f"[agent] retry connecting to {try_url}")
                    ws = websocket.create_connection(try_url, timeout=5)
                else:
                    raise
            # After connecting, remove short timeout so idle does not disconnect
            try:
                ws.settimeout(None)
            except Exception:
                pass
            _send(
                ws,
                {
                    "type": "hello",
                    "agent": "uhop-agent",
                    "version": "0.1.0",
                    "token": cfg.token,
                },
            )
            _send(
                ws,
                {
                    "type": "log",
                    "level": "info",
                    "line": f"[agent] connected to {try_url}",
                },
            )

            while True:
                try:
                    raw = ws.recv()
                except Exception as e:
                    # websocket-client raises WebSocketTimeoutException on idle if timeout set; just continue
                    if e.__class__.__name__ == "WebSocketTimeoutException":
                        continue
                    raise
                if not raw:
                    break
                try:
                    msg = json.loads(raw)
                except Exception:
                    continue

                if msg.get("type") == "request":
                    req_id = msg.get("id")
                    action = msg.get("action")
                    params = msg.get("params") or {}
                    try:
                        if action == "info":
                            data = _info_json()
                            _send(
                                ws,
                                {
                                    "type": "response",
                                    "id": req_id,
                                    "ok": True,
                                    "data": data,
                                },
                            )
                        elif action == "run_demo":
                            size = int(params.get("size", 128))
                            iters = int(params.get("iters", 2))
                            _send(
                                ws,
                                {
                                    "type": "log",
                                    "level": "info",
                                    "line": f"[agent] running demo size={size} iters={iters}",
                                },
                            )
                            data = _run_demo(size, iters)
                            _send(
                                ws,
                                {
                                    "type": "response",
                                    "id": req_id,
                                    "ok": True,
                                    "data": data,
                                },
                            )
                        elif action == "generate_kernel":
                            target = str(params.get("target", "opencl"))
                            _send(
                                ws,
                                {
                                    "type": "log",
                                    "level": "info",
                                    "line": f"[agent] generating kernel target={target}",
                                },
                            )
                            data = _generate_kernel(target)
                            _send(
                                ws,
                                {
                                    "type": "response",
                                    "id": req_id,
                                    "ok": True,
                                    "data": data,
                                },
                            )
                        else:
                            _send(
                                ws,
                                {
                                    "type": "response",
                                    "id": req_id,
                                    "ok": False,
                                    "error": f"unknown action: {action}",
                                },
                            )
                    except Exception as e:
                        _send(
                            ws,
                            {
                                "type": "response",
                                "id": req_id,
                                "ok": False,
                                "error": str(e),
                            },
                        )
                        tb = traceback.format_exc()
                        _send(
                            ws,
                            {
                                "type": "log",
                                "level": "error",
                                "line": f"[agent] error: {e}\n{tb}",
                            },
                        )
                # ignore other message types for now

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[agent] connection error: {e!r}", file=sys.stderr)
        finally:
            try:
                if ws is not None:
                    ws.close()
            except Exception:
                pass
        if not cfg.reconnect:
            break
    time.sleep(cfg.reconnect_delay)


def main(argv: Optional[list[str]] = None):
    ap = argparse.ArgumentParser(description="UHOP Local Agent")
    ap.add_argument(
        "--server",
        type=str,
        default=os.environ.get("UHOP_AGENT_SERVER", "ws://localhost:8787/agent"),
    )
    ap.add_argument("--token", type=str, default=os.environ.get("UHOP_AGENT_TOKEN"))
    ap.add_argument("--no-reconnect", action="store_true")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args(argv)
    cfg = AgentConfig(
        server=args.server, token=args.token, reconnect=not args.no_reconnect
    )
    run_agent(cfg, debug=args.debug)


if __name__ == "__main__":
    main()
