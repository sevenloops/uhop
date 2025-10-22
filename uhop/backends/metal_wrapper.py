"""
Minimal Metal wrapper that compiles .metal source via Apple's metal/metallib toolchain
and uses subprocess to run a tiny host helper if available. This is a defensive
implementation: if the macOS toolchain is not present, the wrapper raises.
"""
from pathlib import Path
import subprocess
import shutil
import tempfile


def ensure_metal_toolchain():
    if shutil.which("metal") is None or shutil.which("metallib") is None:
        raise RuntimeError("Apple Metal toolchain not found (metal/metallib).")


class MetalKernel:
    def __init__(self, source: str, kernel_name: str):
        ensure_metal_toolchain()
        # write source to temp .metal file and compile to .metallib
        self._tmpdir = tempfile.mkdtemp(prefix="uhop_metal_")
        self.src_path = Path(self._tmpdir) / "kernel.metal"
        self.metal_lib = Path(self._tmpdir) / "kernel.metallib"
        self.src_path.write_text(source)
        # compile
        # compile to air (object) then metallib
        air_path = Path(self._tmpdir) / "kernel.air"
        subprocess.check_call(["metal", "-c", str(self.src_path), "-o", str(air_path)])
        subprocess.check_call(["metallib", str(air_path), "-o", str(self.metal_lib)])
        # At this stage, a proper loader (via PyObjC or Metal Performance Shaders) would be needed
        # This minimal wrapper only ensures the library compiles; runtime launching requires bindings.

    def launch(self, *args, **kwargs):
        raise NotImplementedError("Runtime launch via Metal is not implemented in this minimal wrapper. Use host-side helper or extend with PyObjC bindings.")


def time_kernel_run(*args, **kwargs):
    raise NotImplementedError("Metal runtime timing not implemented in this wrapper.")
