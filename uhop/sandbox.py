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
import sys, importlib.util, numpy as np
mod_path = sys.argv[1]
fn_name = sys.argv[2]
args_paths = sys.argv[3:-1]
out_path = sys.argv[-1]

spec = importlib.util.spec_from_file_location("uhop_gen", mod_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
fn = getattr(mod, fn_name)

# load args
args = [np.load(p, allow_pickle=True) for p in args_paths]
res = fn(*args)
np.save(out_path, res, allow_pickle=True)
"""


def run_generated_python(
    module_path: str, function_name: str, *args, timeout: int = 10
):
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
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0:
            raise RuntimeError(
                f"Sandbox failed: rc={proc.returncode}\nSTDOUT:{proc.stdout}\nSTDERR:{proc.stderr}"
            )
        res = np.load(out_path, allow_pickle=True)
        return res
