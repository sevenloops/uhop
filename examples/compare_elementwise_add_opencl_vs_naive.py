"""
Compare elementwise add using UHOP's OpenCL kernel vs a naive Python baseline.

This uses the elementwise add kernel template and the autotuner to select a good
block/grid, then compiles and runs it via PyOpenCL. The baseline is a plain
Python loop adding two arrays element-wise.
"""

import argparse
import statistics
import time
from pathlib import Path

import numpy as np
from jinja2 import Template

from uhop.autotuner import get_cached_or_tune


def render_opencl_add_kernel(context: dict) -> str:
    tpl_path = Path("uhop/kernels/opencl/elementwise_add.cl.jinja")
    src = tpl_path.read_text()
    return Template(src).render(**context)


def naive_python_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    assert a.shape == b.shape
    out = np.empty_like(a)
    for i in range(a.size):
        out[i] = a[i] + b[i]
    return out


def bench(fn, *args, warmup=1, iters=3):
    for _ in range(warmup):
        fn(*args)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def main():
    parser = argparse.ArgumentParser(description="UHOP OpenCL add vs naive Python")
    parser.add_argument("--size", type=int, default=2_000_000, help="Number of elements")
    parser.add_argument("--dtype", type=str, default="float32", help="Data type (float32/int32)")
    args = parser.parse_args()

    size = int(args.size)
    dtype = str(args.dtype)
    np_dtype = np.float32 if "float" in dtype else np.int32

    # Prepare inputs
    rng = np.random.default_rng(0)
    a = rng.random(size, dtype=np_dtype)
    b = rng.random(size, dtype=np_dtype)

    # Tune and compile UHOP OpenCL kernel
    best = get_cached_or_tune("add", size=size, dtype=dtype, device="opencl")
    context = best["kernel_source_context"]
    source = render_opencl_add_kernel(context)

    from uhop.backends import opencl_wrapper

    cl = opencl_wrapper.cl
    kernel = opencl_wrapper.OpenCLKernel(source, "elem_op")

    # Device buffers
    mf = cl.mem_flags
    da = cl.Buffer(kernel.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    db = cl.Buffer(kernel.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
    dout = cl.Buffer(kernel.ctx, mf.WRITE_ONLY, a.nbytes)

    # Launch configuration from autotuner
    threads = int(best["block"])  # local_size
    grid = int(best["grid"])  # number of workgroups along X
    global_size = (grid * threads,)
    local_size = (threads,)

    # Define a runner that launches the kernel
    def run_ocl():
        # Set args and run once
        args_tuple = (da, db, dout, np.uint64(size))
        kernel.kernel.set_args(*args_tuple)
        evt = cl.enqueue_nd_range_kernel(kernel.queue, kernel.kernel, global_size, local_size)
        evt.wait()

    # Warmups + timed median (kernel time only via profiling)
    # For a fairer wall time, we use perf_counter as the baseline does.
    t_ocl = bench(run_ocl, warmup=2, iters=6)

    # Read back result for one correctness check
    out = np.empty_like(a)
    cl.enqueue_copy(kernel.queue, out, dout).wait()
    expect = a + b
    assert np.allclose(out, expect, atol=1e-5, rtol=1e-4)

    # Baseline naive Python
    t_naive = bench(naive_python_add, a, b, warmup=1, iters=3)

    print(f"Elements: {size}")
    print(f"UHOP OpenCL median time: {t_ocl:.6f} s")
    print(f"Naive Python median time: {t_naive:.6f} s")
    if t_ocl > 0:
        print(f"Speedup (naive / UHOP): {t_naive / t_ocl:.2f}x")


if __name__ == "__main__":
    main()
