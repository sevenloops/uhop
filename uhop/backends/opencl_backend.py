# uhop/backends/opencl_backend.py
"""
OpenCL backend (pyopencl) for matmul / relu and a correctness-first conv2d fallback.
Useful for AMD & Intel GPUs where ROCm or native PyTorch support is missing.
"""
try:
    import pyopencl as cl
    import numpy as np  # type: ignore
    _OPENCL_AVAILABLE = True
except Exception:
    cl = None
    np = None
    _OPENCL_AVAILABLE = False

def is_opencl_available():
    return _OPENCL_AVAILABLE

if _OPENCL_AVAILABLE:
    def _build_ctx_queue():
        ctx = cl.create_some_context()
        q = cl.CommandQueue(ctx)
        return ctx, q

    def opencl_matmul(a, b):
        import numpy as _np
        ctx, q = _build_ctx_queue()
        a = _np.array(a, dtype=_np.float32)
        b = _np.array(b, dtype=_np.float32)
        m, k = a.shape
        k2, n = b.shape
        assert k == k2
        c = _np.zeros((m, n), dtype=_np.float32)
        mf = cl.mem_flags
        a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
        b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
        c_buf = cl.Buffer(ctx, mf.WRITE_ONLY, c.nbytes)
        prg = cl.Program(ctx, """
        __kernel void matmul(const int M, const int N, const int K,
                             __global const float* A,
                             __global const float* B,
                             __global float* C) {
            int row = get_global_id(0);
            int col = get_global_id(1);
            if (row < M && col < N) {
                float s = 0.0f;
                for (int k = 0; k < K; ++k) {
                    s += A[row*K + k] * B[k*N + col];
                }
                C[row*N + col] = s;
            }
        }
        """).build()
        prg.matmul(q, (m, n), None, _np.int32(m), _np.int32(n), _np.int32(k), a_buf, b_buf, c_buf)
        cl.enqueue_copy(q, c, c_buf)
        q.finish()
        return c

    def opencl_relu(x):
        import numpy as _np
        ctx, q = _build_ctx_queue()
        x = _np.array(x, dtype=_np.float32).ravel()
        out = _np.empty_like(x)
        mf = cl.mem_flags
        a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x)
        out_buf = cl.Buffer(ctx, mf.WRITE_ONLY, out.nbytes)
        prg = cl.Program(ctx, """
        __kernel void relu(__global const float* A, __global float* Out, const int N) {
            int i = get_global_id(0);
            if (i < N) {
                float v = A[i];
                Out[i] = v > 0.0f ? v : 0.0f;
            }
        }
        """).build()
        prg.relu(q, (x.size,), None, a_buf, out_buf, _np.int32(x.size))
        cl.enqueue_copy(q, out, out_buf)
        q.finish()
        return out.reshape(-1)  # caller should reshape

    def opencl_conv2d(input_np, weight_np, stride=1, padding=0):
        # Correctness-first CPU fallback to ensure functionality on all OpenCL-capable machines.
        import numpy as _np
        N, C, H, W = input_np.shape
        Cout, Cin, KH, KW = weight_np.shape
        outH = H - KH + 1
        outW = W - KW + 1
        out = _np.zeros((N, Cout, outH, outW), dtype=_np.float32)
        for n in range(N):
            for co in range(Cout):
                for y in range(outH):
                    for x in range(outW):
                        s = 0.0
                        for ci in range(Cin):
                            for ky in range(KH):
                                for kx in range(KW):
                                    s += input_np[n, ci, y+ky, x+kx] * weight_np[co, ci, ky, kx]
                        out[n, co, y, x] = s
        return out
else:
    def opencl_matmul(*args, **kwargs):
        raise RuntimeError("pyopencl not available")
    def opencl_relu(*args, **kwargs):
        raise RuntimeError("pyopencl not available")
    def opencl_conv2d(*args, **kwargs):
        raise RuntimeError("pyopencl not available")
