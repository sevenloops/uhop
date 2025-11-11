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
    p = subprocess.run(cmd, cwd=str(root), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ai-generate failed: {p.stderr.strip() or p.stdout.strip()}")

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


def _compile_kernel(descriptor: Dict[str, Any]) -> Dict[str, Any]:
    """Minimal compile handler.

    Supports:
      - IR descriptor under key "ir" (lowered to OpenCL for supported ops)
      - Raw source when source.lang == "opencl" with inline source text

    Optionally accepts build options under descriptor.schedule/build_opts.
    """
    # IR path: lower to OpenCL source when provided
    ir_desc = (descriptor or {}).get("ir")
    src = None
    lang = ""
    kernel_name = None
    if isinstance(ir_desc, dict) and ir_desc:
        try:
            from .ir import ir_from_dict  # type: ignore
            from .ir.opencl_lowering import lower_to_opencl  # type: ignore
            from .ir.registry import compute_ir_key  # type: ignore

            op = ir_from_dict(ir_desc)
            lowered = lower_to_opencl(op)
            src = lowered.get("source")
            lang = lowered.get("language", "")
            kernel_name = lowered.get("kernel_name")
            ir_key = compute_ir_key(ir_desc)
        except Exception as e:  # pragma: no cover (covered by tests around failure)
            raise RuntimeError(f"IR lowering failed: {e}")
    else:
        # Raw source path
        src = ((descriptor or {}).get("source") or {}).get("text")
        lang = ((descriptor or {}).get("source") or {}).get("lang", "")
        kernel_name = None
    build_opts = None
    sched = (descriptor or {}).get("schedule") or {}
    if isinstance(sched, dict):
        # accept explicit build_opts or derive a simple -D TILE/VEC if present
        build_opts = sched.get("build_opts")
        tile = sched.get("tile")
        vec = sched.get("vec")
        opts = []
        if tile:
            opts.append(f"-D TILE={int(tile)}")
        if vec:
            opts.append(f"-D VEC={int(vec)}")
        if opts:
            build_opts = (build_opts + " " if build_opts else "") + " ".join(opts)
    if not isinstance(src, str) or lang.lower() != "opencl":
        raise RuntimeError("compile_kernel supports only IR->OpenCL or source.lang=='opencl'")
    # Build using a throwaway context
    try:
        import pyopencl as cl  # type: ignore

        from .cli import _ensure_opencl_context_for_validation as _ctx

        ctx, q = _ctx()
        prg = cl.Program(ctx, src).build(options=build_opts)
        # Persist binary via KernelRegistry (best-effort)
        import hashlib

        from .cache import KernelRegistry as _KR

        dev = ctx.devices[0]
        dev_name = getattr(dev, "name", "unknown")
        h = hashlib.sha1((src + "\n" + (build_opts or "")).encode("utf-8")).hexdigest()
        try:
            bins = prg.get_info(cl.program_info.BINARIES)
        except Exception:
            bins = None
        bin_path = None
        if bins and isinstance(bins, (list, tuple)) and bins[0]:
            try:
                kr = _KR()
                kr.save_opencl_binary(dev_name, h, bins)  # type: ignore
                # Return path for first binary
                bin_path = kr.load_opencl_binary(dev_name, h)
            except Exception:
                bin_path = None
        # Map IR key to source hash/binary for reuse
        try:
            if isinstance(ir_desc, dict):
                from .ir.registry import IRKernelIndex, compute_ir_key  # type: ignore

                IRKernelIndex().set(
                    compute_ir_key(ir_desc),
                    dev_name,
                    h,
                    kernel_name=kernel_name,
                    binary_path=str(bin_path) if bin_path else None,
                )
        except Exception:
            pass
        art = {
            "id": f"opencl:{h}",
            "device": dev_name,
            "binary_path": str(bin_path) if bin_path else None,
            "compiler_opts": build_opts or "",
            "kernel_name": kernel_name,
        }
        if isinstance(ir_desc, dict):  # attach IR metadata
            art["ir_key"] = ir_key  # type: ignore[name-defined]
        return {"artifact": art}
    except Exception as e:
        raise RuntimeError(f"OpenCL build failed: {e}")


def _validate(params: Dict[str, Any]) -> Dict[str, Any]:
    """Minimal validation handler.

    Params:
      { "op": "matmul|relu", "backend": "opencl|triton|torch|numpy",
        "shapes": { ... }, "runs": int, "tolerance": float }
    """
    import time

    import numpy as np

    op = str(params.get("op", "")).lower()
    backend = str(params.get("backend", "numpy")).lower()
    runs = int(params.get("runs", 3))
    tol = float(params.get("tolerance", 1e-4))
    shapes = params.get("shapes") or {}
    ir_desc = params.get("ir") or None

    def _bench(fn):
        ts = []
        for _ in range(max(1, runs)):
            t0 = time.perf_counter()
            fn()
            ts.append(time.perf_counter() - t0)
        return float(np.mean(ts)), float(np.std(ts) if len(ts) > 1 else 0.0)

    # IR path validation (OpenCL execution)
    if ir_desc:
        try:
            import numpy as np
            import pyopencl as cl  # type: ignore

            from .ir import ir_from_dict  # type: ignore
            from .ir.opencl_lowering import lower_to_opencl  # type: ignore

            op_ir = ir_from_dict(ir_desc)
            lowered = lower_to_opencl(op_ir)
            if lowered.get("language") != "opencl":
                raise RuntimeError("Only OpenCL lowering is supported for IR validate")

            # Allow multiple shape sets: params["shape_sets"] = [{"A":[M,K],"B":[K,N]}]
            shape_sets = params.get("shape_sets")
            results = []

            def _run_once(M, K, N):
                rng = np.random.default_rng(0)
                A = rng.random((M, K), dtype=np.float32)
                B = rng.random((K, N), dtype=np.float32)
                ref = A @ B
                if op_ir.op_type == "fused_matmul_relu":
                    ref = np.maximum(ref, 0)

                ctx = cl.create_some_context(interactive=False)
                q = cl.CommandQueue(ctx)
                prg = cl.Program(ctx, lowered["source"]).build()
                kname = lowered.get("kernel_name") or (
                    "uhop_fused_matmul_relu" if op_ir.op_type == "fused_matmul_relu" else "uhop_matmul"
                )
                kern = getattr(prg, kname)

                mf = cl.mem_flags
                bufA = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
                bufB = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
                out = np.empty((M, N), dtype=np.float32)
                bufC = cl.Buffer(ctx, mf.WRITE_ONLY, size=out.nbytes)

                # Compute global/local sizes from tile
                ts = int(lowered.get("tile") or 16)
                gs0 = ((M + ts - 1) // ts) * ts
                gs1 = ((N + ts - 1) // ts) * ts
                local = (ts, ts)

                kern.set_args(np.int32(M), np.int32(N), np.int32(K), bufA, bufB, bufC)
                cl.enqueue_nd_range_kernel(q, kern, (gs0, gs1), local)
                cl.enqueue_copy(q, out, bufC)
                q.finish()
                err = float(np.max(np.abs(out - ref)))
                return {"max_abs_err": err, "passed": bool(err <= tol)}

            if op_ir.op_type in ("matmul", "fused_matmul_relu"):
                if shape_sets and isinstance(shape_sets, list):
                    for s in shape_sets:
                        A_s = tuple(int(x) for x in (s.get("A") or ()))
                        B_s = tuple(int(x) for x in (s.get("B") or ()))
                        if len(A_s) != 2 or len(B_s) != 2 or A_s[1] != B_s[0]:
                            raise RuntimeError("invalid shape_sets entry for matmul")
                        results.append(_run_once(A_s[0], A_s[1], B_s[1]))
                    all_pass = all(r.get("passed") for r in results)
                    return {"ok": True, "validated": {"multi": results, "passed": bool(all_pass)}}
                else:
                    if op_ir.op_type == "matmul":
                        M, K = op_ir.A.shape  # type: ignore[attr-defined]
                        K2, N = op_ir.B.shape  # type: ignore[attr-defined]
                    else:
                        M, K = op_ir.A.shape  # type: ignore[attr-defined]
                        K2, N = op_ir.B.shape  # type: ignore[attr-defined]
                    assert K == K2
                    single = _run_once(M, K, N)
                    return {"ok": True, "validated": single}
            elif op_ir.op_type == "relu":
                N = op_ir.X.shape[0]  # type: ignore[attr-defined]
                rng = np.random.default_rng(0)
                X = rng.standard_normal(N).astype(np.float32)
                ref = np.maximum(X, 0)
                ctx = cl.create_some_context(interactive=False)
                q = cl.CommandQueue(ctx)
                prg = cl.Program(ctx, lowered["source"]).build()
                kern = getattr(prg, lowered.get("kernel_name") or "uhop_relu")
                mf = cl.mem_flags
                bufX = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=X)
                out = np.empty_like(X)
                bufY = cl.Buffer(ctx, mf.WRITE_ONLY, size=out.nbytes)
                kern.set_args(np.int32(N), bufX, bufY)
                # Use a round-up global size for simplicity
                ts = int(lowered.get("tile") or 64)
                gs = ((N + ts - 1) // ts) * ts
                cl.enqueue_nd_range_kernel(q, kern, (gs,), (ts,))
                cl.enqueue_copy(q, out, bufY)
                q.finish()
                err = float(np.max(np.abs(out - ref)))
                return {"ok": True, "validated": {"max_abs_err": err, "passed": bool(err <= tol)}}
            else:
                raise RuntimeError(f"IR validate unsupported op_type={getattr(op_ir, 'op_type', '?')}")
        except Exception as e:
            raise RuntimeError(f"IR validate failed: {e}")

    if op == "matmul":
        A_shape = tuple(int(x) for x in (shapes.get("A") or (32, 32)))
        B_shape = tuple(int(x) for x in (shapes.get("B") or (32, 32)))
        A = np.random.default_rng(0).random(A_shape, dtype=np.float32)
        B = np.random.default_rng(1).random(B_shape, dtype=np.float32)
        if A.shape[1] != B.shape[0]:
            raise RuntimeError("matmul shapes incompatible (A.cols != B.rows)")
        ref = A @ B

        # target run
        def _run_target():
            if backend == "opencl":
                from .backends import opencl_matmul

                return opencl_matmul(A, B)
            if backend == "torch":
                from .backends import torch_matmul

                return torch_matmul(A, B, keep_format=False)
            if backend == "triton":
                from .backends import triton_matmul

                return triton_matmul(A, B)
            return A @ B

        out = _run_target()
        err = float(np.max(np.abs(out - ref)))
        mean_s, std_s = _bench(_run_target)
        return {
            "ok": True,
            "validated": {"max_abs_err": err, "passed": bool(err <= tol)},
            "stats": {"mean_ms": mean_s * 1000.0, "std_ms": std_s * 1000.0},
        }
    if op == "relu":
        N = int((shapes.get("X") or [1024])[0])
        X = np.random.default_rng(0).standard_normal(N).astype(np.float32)
        ref = np.maximum(X, 0)

        def _run_target():
            if backend == "opencl":
                from .backends import opencl_relu

                return opencl_relu(X)
            if backend == "torch":
                from .backends import torch_relu

                return torch_relu(X, keep_format=False)
            return np.maximum(X, 0)

        out = _run_target()
        err = float(np.max(np.abs(out - ref)))
        mean_s, std_s = _bench(_run_target)
        return {
            "ok": True,
            "validated": {"max_abs_err": err, "passed": bool(err <= tol)},
            "stats": {"mean_ms": mean_s * 1000.0, "std_ms": std_s * 1000.0},
        }
    raise RuntimeError(f"Unsupported op for validate: {op}")


def _list_cache() -> Dict[str, Any]:
    from .cache import UhopCache as _UhopCache

    c = _UhopCache()
    return {"selection": c.all()}


def _set_cache(key: str, record: Dict[str, Any]) -> Dict[str, Any]:
    from .cache import UhopCache as _UhopCache

    c = _UhopCache()
    c.set(key, record)
    return {"ok": True}


def _delete_cache(key: str) -> Dict[str, Any]:
    from .cache import UhopCache as _UhopCache

    c = _UhopCache()
    c.delete(key)
    return {"ok": True}


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
                        elif action == "compile_kernel":
                            desc = params.get("descriptor") or {}
                            data = _compile_kernel(desc)
                            _send(ws, {"type": "response", "id": req_id, "ok": True, "data": data})
                        elif action == "validate":
                            data = _validate(params)
                            _send(ws, {"type": "response", "id": req_id, "ok": True, "data": data})
                        elif action == "cache_list":
                            data = _list_cache()
                            _send(ws, {"type": "response", "id": req_id, "ok": True, "data": data})
                        elif action == "cache_set":
                            key = str(params.get("key"))
                            record = params.get("record") or {}
                            if not key:
                                raise ValueError("missing cache key")
                            data = _set_cache(key, record)
                            _send(ws, {"type": "response", "id": req_id, "ok": True, "data": data})
                        elif action == "cache_delete":
                            key = str(params.get("key"))
                            if not key:
                                raise ValueError("missing cache key")
                            data = _delete_cache(key)
                            _send(ws, {"type": "response", "id": req_id, "ok": True, "data": data})
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
    cfg = AgentConfig(server=args.server, token=args.token, reconnect=not args.no_reconnect)
    run_agent(cfg, debug=args.debug)


if __name__ == "__main__":
    main()
