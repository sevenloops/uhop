# uhop/sandbox.py
"""
Sandbox runner that executes a specified function inside a generated Python module
in a separate subprocess with a timeout. Uses .npy files for arguments and results.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

_SANDBOX_RUNNER = r"""
import sys, importlib.util, numpy as np, os
# Optional resource limits (Linux/Unix)
try:
    import resource as _resource  # type: ignore
except Exception:
    _resource = None

mod_path = sys.argv[1]
fn_name = sys.argv[2]
args_paths = sys.argv[3:-1]
out_path = sys.argv[-1]

_cpu_secs = os.environ.get("UHOP_SB_CPU_SECS")
_mem_mb = os.environ.get("UHOP_SB_MEM_MB")
_block_net = os.environ.get("UHOP_SB_BLOCK_NET")
try:
    if _resource is not None:
        # CPU time limit
        if _cpu_secs is not None:
            secs = max(1, int(float(_cpu_secs)))
            _resource.setrlimit(_resource.RLIMIT_CPU, (secs, secs))
        # Address space (virtual memory) limit
        if _mem_mb is not None:
            mb = max(128, int(float(_mem_mb)))
            bytes_lim = mb * 1024 * 1024
            _resource.setrlimit(_resource.RLIMIT_AS, (bytes_lim, bytes_lim))
except Exception:
    pass

# Optional network disable
try:
    if str(_block_net).lower() in ("1","true","yes","on"):
        import socket as _socket  # type: ignore
        class _NetBlocked:
            def __getattr__(self, name):
                raise RuntimeError("Network access is disabled in UHOP sandbox")
        # Replace common constructors/APIs
        _socket.socket = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("Network disabled"))
        _socket.create_connection = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("Network disabled"))
        _socket.getaddrinfo = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("Network disabled"))
except Exception:
    pass

spec = importlib.util.spec_from_file_location("uhop_gen", mod_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
fn = getattr(mod, fn_name)

# load args
args = [np.load(p, allow_pickle=True) for p in args_paths]
res = fn(*args)
np.save(out_path, res, allow_pickle=True)
"""


def run_generated_python(module_path: str, function_name: str, *args, timeout: int = 10):
    # Basic validation of function name to reduce risk surface
    if (
        not function_name
        or "." in function_name
        or function_name.startswith("_")
        or not function_name.startswith("generated_")
    ):
        raise ValueError(
            "Refusing to execute untrusted function name in sandbox; must start with 'generated_' and contain no dots/underscores prefix"
        )
    module_path = str(Path(module_path).resolve())
    with tempfile.TemporaryDirectory() as td:
        arg_paths = []
        for i, a in enumerate(args):
            p = os.path.join(td, f"arg{i}.npy")
            np.save(p, np.array(a, copy=False))
            arg_paths.append(p)
        out_path = os.path.join(td, "out.npy")
        runner = os.path.join(td, "runner.py")
        with open(runner, "w") as f:
            f.write(_SANDBOX_RUNNER)
        cmd = [sys.executable, runner, module_path, function_name, *arg_paths, out_path]
        env = dict(os.environ)
        # Provide soft resource hints to the runner
        env.setdefault("UHOP_SB_CPU_SECS", str(max(1, int(timeout))))
        # Cap memory to a reasonable ceiling unless user overrides
        env.setdefault("UHOP_SB_MEM_MB", str(1024))  # 1 GiB default
        # Allow caller to opt-in to network blocking
        env.setdefault("UHOP_SB_BLOCK_NET", os.environ.get("UHOP_SB_BLOCK_NET", "1"))
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
        if proc.returncode != 0:
            raise RuntimeError(f"Sandbox failed: rc={proc.returncode}\nSTDOUT:{proc.stdout}\nSTDERR:{proc.stderr}")
        res = np.load(out_path, allow_pickle=True)
        return res
