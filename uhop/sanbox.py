# uhop/sandbox.py
"""
Sandboxed executor for running generated kernels in a subprocess.
This reduces risk executing AI-generated code in the main process.
Uses temporary files and a small runner script.
"""
import os
import sys
import tempfile
import subprocess
import numpy as np
from pathlib import Path
from typing import Any

_SANDBOX_RUNNER = r'''
import sys, importlib.util, numpy as np
mod_path = sys.argv[1]
fn_name = sys.argv[2]
a_path = sys.argv[3]
b_path = sys.argv[4]
out_path = sys.argv[5]
spec = importlib.util.spec_from_file_location("uhop_gen", mod_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
fn = getattr(mod, fn_name)
a = np.load(a_path)
b = np.load(b_path)
res = fn(a, b)
np.save(out_path, res)
'''

def run_generated_function(module_path: str, function_name: str, a: Any, b: Any, timeout: int = 10):
    module_path = str(Path(module_path).resolve())
    with tempfile.TemporaryDirectory() as td:
        a_path = os.path.join(td, "a.npy")
        b_path = os.path.join(td, "b.npy")
        out_path = os.path.join(td, "out.npy")
        runner = os.path.join(td, "runner.py")
        np = __import__("numpy")
        np.save(a_path, np.array(a))
        np.save(b_path, np.array(b))
        with open(runner, "w") as f:
            f.write(_SANDBOX_RUNNER)
        cmd = [sys.executable, runner, module_path, function_name, a_path, b_path, out_path]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0:
            raise RuntimeError(f"Sandbox failed: rc={proc.returncode}\nSTDOUT:{proc.stdout}\nSTDERR:{proc.stderr}")
        res = np.load(out_path, allow_pickle=True)
        return res
