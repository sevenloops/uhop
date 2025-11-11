"""
Quick smoke test for OpenCL kernels added in uhop/kernels/opencl.
This script compiles a few kernels and runs them to validate they execute.
Requires: pyopencl, numpy, an OpenCL device.
"""

import numpy as np

try:
    import pyopencl as cl
except Exception as e:
    raise SystemExit("pyopencl not available: install pyopencl to run this smoke test")

from pathlib import Path

KDIR = Path(__file__).resolve().parents[1] / "uhop" / "kernels" / "opencl"


def build_program(ctx, fname: str) -> cl.Program:
    src = (KDIR / fname).read_text()
    return cl.Program(ctx, src).build()


def run_elementwise_add(ctx, q):
    prg = build_program(ctx, "elementwise.cl")
    a = np.arange(1024, dtype=np.float32)
    b = np.ones_like(a)
    out = np.empty_like(a)
    mf = cl.mem_flags
    a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
    out_buf = cl.Buffer(ctx, mf.WRITE_ONLY, out.nbytes)
    prg.elementwise_add(q, (a.size,), None, a_buf, b_buf, out_buf, np.int32(a.size))
    cl.enqueue_copy(q, out, out_buf).wait()
    assert np.allclose(out, a + b)
    print("elementwise_add: OK")


def run_reduce_sum(ctx, q):
    prg = build_program(ctx, "reduce.cl")
    x = np.random.randn(4096).astype(np.float32)
    partials = np.empty(64, dtype=np.float32)
    out = np.zeros(1, dtype=np.float32)
    mf = cl.mem_flags
    x_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x)
    parts_buf = cl.Buffer(ctx, mf.WRITE_ONLY, partials.nbytes)
    out_buf = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=out)
    # stage 1
    lsz = 256
    gsz = 64 * lsz
    prg.reduce_sum_partials(q, (gsz,), (lsz,), x_buf, parts_buf, np.int32(x.size))
    cl.enqueue_copy(q, partials, parts_buf).wait()
    # stage 2
    prg.reduce_sum_finalize(q, (1,), None, parts_buf, out_buf, np.int32(partials.size))
    cl.enqueue_copy(q, out, out_buf).wait()
    assert np.allclose(out[0], x.sum(), atol=1e-3)
    print("reduce_sum: OK")


def run_transpose(ctx, q):
    prg = build_program(ctx, "transpose_dot.cl")
    x = np.arange(3 * 4, dtype=np.float32).reshape(3, 4)
    y = np.zeros((4, 3), dtype=np.float32)
    mf = cl.mem_flags
    x_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x)
    y_buf = cl.Buffer(ctx, mf.WRITE_ONLY, y.nbytes)
    prg.transpose2d(q, (x.shape[1], x.shape[0]), None, x_buf, y_buf, np.int32(x.shape[0]), np.int32(x.shape[1]))
    cl.enqueue_copy(q, y, y_buf).wait()
    assert np.allclose(y, x.T)
    print("transpose2d: OK")


def main():
    ctx = cl.create_some_context(interactive=False)
    q = cl.CommandQueue(ctx)
    run_elementwise_add(ctx, q)
    run_reduce_sum(ctx, q)
    run_transpose(ctx, q)
    print("OpenCL smoke tests passed.")


if __name__ == "__main__":
    main()
