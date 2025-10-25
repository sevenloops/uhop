# Building and using CLBlast with UHOP (Windows/Linux/macOS)

CLBlast provides highly tuned BLAS routines on top of OpenCL. UHOP can optionally use CLBlast for GEMM (matmul) and im2col+GEMM Conv2D when available.

This guide shows how to build and install CLBlast and point UHOP to the built library.

## TL;DR

1) Build or install CLBlast for your platform
2) Set the environment variable `CLBLAST_LIBRARY` to the absolute path of the shared library (DLL/SO/Dylib)
3) Run benchmarks with:

```bash
# Prefer CLBlast for matmul
UHOP_OPENCL_MATMUL_IMPL=clblast python -m uhop.benchmarks.opencl_ops_bench --ops matmul --validate
# Prefer im2col+GEMM for conv2d
UHOP_OPENCL_CONV_IMPL=im2col_gemm python -m uhop.benchmarks.opencl_ops_bench --ops conv2d --validate
```

## Windows (recommended: vcpkg)

Prereqs: Git, CMake, a compiler (MSVC or clang/LLVM), OpenCL SDK/ICD runtime.

```bash
# Install vcpkg (if not already)
git clone https://github.com/microsoft/vcpkg.git %USERPROFILE%/vcpkg
%USERPROFILE%/vcpkg/bootstrap-vcpkg.bat

# Install CLBlast x64
%USERPROFILE%/vcpkg/vcpkg install clblast:x64-windows

# Find the built DLL
# Usually under: %USERPROFILE%/vcpkg/installed/x64-windows/bin/CLBlast.dll
# Point UHOP to it:
setx CLBLAST_LIBRARY "%USERPROFILE%\vcpkg\installed\x64-windows\bin\CLBlast.dll"
```

Alternative (CMake from source):

```bash
# From a working directory
git clone https://github.com/CNugteren/CLBlast.git
cd CLBlast
cmake -S . -B build -D CMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel
# The DLL will be in build/Release/ (CLBlast.dll)
# Set CLBLAST_LIBRARY to the absolute path of the DLL.
```

## Linux (APT/Yum/Pacman or conda-forge)

- Debian/Ubuntu (if available): `sudo apt install -y libclblast-dev`
- Fedora: `sudo dnf install -y clblast` (package names vary)
- Arch: `sudo pacman -S clblast`
- Conda-forge (cross-platform): `conda install -c conda-forge clblast`

Set `CLBLAST_LIBRARY` if UHOP does not find it automatically (e.g., `/usr/lib/x86_64-linux-gnu/libclblast.so`).

## macOS (Homebrew or conda-forge)

- Homebrew: `brew install clblast`
- Conda-forge: `conda install -c conda-forge clblast`

Set `CLBLAST_LIBRARY` to something like `/opt/homebrew/lib/libclblast.dylib`.

## Verification

```bash
python - <<'PY'
from uhop.backends.clblast_integration import load_clblast
lib = load_clblast()
print("CLBlast loaded:", bool(lib))
PY
```

If `False`, double-check your `CLBLAST_LIBRARY` path and OpenCL runtime installation.

## Notes

- CLBlast should match your OpenCL runtime bitness (x64).
- On Windows, ensure the DLL is reachable or `CLBLAST_LIBRARY` is set to its absolute path.
- UHOP falls back to tiled kernels if CLBlast is missing or a call fails; a warning is printed.

## Troubleshooting (Windows)

- Access violation in `CLBlastSgemm` (e.g., "access violation reading 0x60"):
  - Ensure you are using a 64-bit Python and 64-bit CLBlast build
  - Update your GPU's OpenCL driver/ICD (AMD/NVIDIA/Intel)
  - Confirm the queue/buffers are from the same OpenCL context (UHOP handles this internally)
  - UHOP vends a robust ctypes wrapper (WinDLL on Windows, non-NULL cl_event*) but some driver stacks still fault; if it persists, set `UHOP_OPENCL_CONV_IMPL=tiled` and/or `UHOP_OPENCL_MATMUL_IMPL=tiled` to use stable tiled kernels
  - Optionally try CLBlast installed via a different channel (vcpkg vs conda-forge) to vary the build toolchain
