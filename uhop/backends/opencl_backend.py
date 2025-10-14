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
    # Global context/queue and cached programs/kernels (per device)
    _CTX = None
    _Q = None
    _PRG_CACHE = {}  # key: (device_name, BLOCK) -> (program, k_matmul, k_relu, k_conv2d_relu)
    _TUNED = {}      # key: device_name -> chosen BLOCK
    _DEVICE_INDEX = None  # Optional: global GPU device ordinal across all platforms

    _KERNEL_SRC_TEMPLATE = """
    #define BLOCK {BLOCK}
    __kernel void matmul_tiled(const int M, const int N, const int K,
                               __global const float* A,
                               __global const float* B,
                               __global float* C) {
        const int globalRow = get_global_id(0);
        const int globalCol = get_global_id(1);
        const int localRow = get_local_id(0);
        const int localCol = get_local_id(1);
        __local float As[BLOCK][BLOCK];
        __local float Bs[BLOCK][BLOCK];
        float acc = 0.0f;
        const int numTiles = (K + BLOCK - 1) / BLOCK;
        for (int t = 0; t < numTiles; ++t) {
            int aRow = globalRow;
            int aCol = t * BLOCK + localCol;
            int bRow = t * BLOCK + localRow;
            int bCol = globalCol;
            As[localRow][localCol] = (aRow < M && aCol < K) ? A[aRow*K + aCol] : 0.0f;
            Bs[localRow][localCol] = (bRow < K && bCol < N) ? B[bRow*N + bCol] : 0.0f;
            barrier(CLK_LOCAL_MEM_FENCE);
            for (int k = 0; k < BLOCK; ++k) {
                acc += As[localRow][k] * Bs[k][localCol];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (globalRow < M && globalCol < N) {
            C[globalRow*N + globalCol] = acc;
        }
    }
    __kernel void relu(__global const float* A, __global float* Out, const int N) {
        int i = get_global_id(0);
        if (i < N) {
            float v = A[i];
            Out[i] = v > 0.0f ? v : 0.0f;
        }
    }

    // Fused Conv2D + ReLU
    // NCHW input, OIHW weights (Cout, Cin, KH, KW), stride=1/2, padding>=0
    __kernel void conv2d_relu(
        const int N, const int Cin, const int H, const int W,
        const int Cout, const int KH, const int KW,
        const int outH, const int outW,
        const int stride, const int padding,
        __global const float* Input,    // shape: N*C*H*W
        __global const float* Weight,   // shape: Cout*Cin*KH*KW
        __global float* Output          // shape: N*Cout*outH*outW
    ) {
        // Use 3D NDRange: (x: outW, y: outH, z: N*Cout)
        int x = get_global_id(0);
        int y = get_global_id(1);
        int z = get_global_id(2);
        if (x >= outW || y >= outH) return;
        int n = z / Cout;
        int co = z % Cout;
        if (n >= N) return;

        float acc = 0.0f;
        for (int ci = 0; ci < Cin; ++ci) {
            for (int ky = 0; ky < KH; ++ky) {
                for (int kx = 0; kx < KW; ++kx) {
                    int iy = y * stride - padding + ky;
                    int ix = x * stride - padding + kx;
                    if (iy >= 0 && iy < H && ix >= 0 && ix < W) {
                        int in_idx = ((n * Cin + ci) * H + iy) * W + ix;
                        int w_idx  = ((co * Cin + ci) * KH + ky) * KW + kx;
                        acc += Input[in_idx] * Weight[w_idx];
                    }
                }
            }
        }
        // ReLU
        if (acc < 0.0f) acc = 0.0f;
        int out_idx = ((n * Cout + co) * outH + y) * outW + x;
        Output[out_idx] = acc;
    }
    """

    def _flatten_gpu_devices(platforms):
        devs = []
        for p in platforms:
            for d in p.get_devices():
                if d.type & cl.device_type.GPU:
                    devs.append((p, d))
        return devs

    def set_opencl_device(index: int):
        """
        Select GPU device by global index across all OpenCL platforms.
        Resets cached context/queue and kernel caches so the next call reinitializes on the chosen device.
        """
        global _DEVICE_INDEX, _CTX, _Q, _PRG_CACHE, _TUNED
        _DEVICE_INDEX = int(index)
        # Clear caches so next ensure will rebuild for the new device
        _CTX = None
        _Q = None
        _PRG_CACHE.clear()
        _TUNED.clear()

    def _ensure_ctx_queue():
        global _CTX, _Q
        if _CTX is not None and _Q is not None:
            return _CTX, _Q
        # Prefer a GPU device explicitly to avoid interactive selection.
        try:
            platforms = cl.get_platforms()
            # Allow override via env or setter
            import os as _os
            env_idx = _os.environ.get("UHOP_OPENCL_DEVICE_INDEX")
            global _DEVICE_INDEX
            chosen_idx = _DEVICE_INDEX
            if env_idx is not None:
                try:
                    chosen_idx = int(env_idx)
                except Exception:
                    pass
            if chosen_idx is not None:
                gpu_devs = _flatten_gpu_devices(platforms)
                if 0 <= chosen_idx < len(gpu_devs):
                    p, d = gpu_devs[chosen_idx]
                    _ctx = cl.Context(devices=[d])
                    _q = cl.CommandQueue(_ctx)
                    _CTX, _Q = _ctx, _q
                    return _CTX, _Q
            # Default: first GPU found
            for p in platforms:
                gpus = [d for d in p.get_devices() if d.type & cl.device_type.GPU]
                if gpus:
                    _ctx = cl.Context(devices=[gpus[0]])
                    _q = cl.CommandQueue(_ctx)
                    _CTX, _Q = _ctx, _q
                    return _CTX, _Q
        except Exception:
            pass
        # Fallback: let pyopencl choose
        _CTX = cl.create_some_context(interactive=False)
        _Q = cl.CommandQueue(_CTX)
        return _CTX, _Q

    def _get_device_name(ctx):
        try:
            return ctx.devices[0].name
        except Exception:
            return "unknown_device"

    def _get_or_build_program(block: int):
        ctx, _ = _ensure_ctx_queue()
        dev_name = _get_device_name(ctx)
        key = (dev_name, block)
        if key in _PRG_CACHE:
            return _PRG_CACHE[key]
        src = _KERNEL_SRC_TEMPLATE.format(BLOCK=block)
        prg = cl.Program(ctx, src).build()
        k_mm = cl.Kernel(prg, "matmul_tiled")
        k_relu = cl.Kernel(prg, "relu")
        # conv2d_relu may fail to build on very old drivers; guard creation
        cl_k_conv = None
        try:
            cl_k_conv = cl.Kernel(prg, "conv2d_relu")
        except Exception:
            cl_k_conv = None
        _PRG_CACHE[key] = (prg, k_mm, k_relu, cl_k_conv)
        return _PRG_CACHE[key]

    def _ensure_tuned_program():
        ctx, _ = _ensure_ctx_queue()
        dev_name = _get_device_name(ctx)
        if dev_name in _TUNED:
            block = _TUNED[dev_name]
            return (*_get_or_build_program(block), block)
        # Autotune BLOCK over safe candidates
        import numpy as _np, time as _time
        candidates = [8, 16]  # conservative; 32 may exceed work-group limits on some AMD GPUs
        # small test size scaled to BLOCK
        best_t = float("inf")
        best_b = 16
        for b in candidates:
            try:
                _, k_mm, _, _ = _get_or_build_program(b)
                m = n = k = b * 32  # e.g., 256 or 512
                A = _np.random.rand(m, k).astype(_np.float32)
                B = _np.random.rand(k, n).astype(_np.float32)
                mf = cl.mem_flags
                a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
                b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
                c_buf = cl.Buffer(ctx, mf.WRITE_ONLY, A.dtype.itemsize * m * n)
                q = _Q
                global_m = ((m + b - 1) // b) * b
                global_n = ((n + b - 1) // b) * b
                # Warmup
                k_mm.set_args(_np.int32(m), _np.int32(n), _np.int32(k), a_buf, b_buf, c_buf)
                cl.enqueue_nd_range_kernel(q, k_mm, (global_m, global_n), (b, b))
                q.finish()
                t0 = _time.perf_counter()
                cl.enqueue_nd_range_kernel(q, k_mm, (global_m, global_n), (b, b))
                q.finish()
                t = _time.perf_counter() - t0
                if t < best_t:
                    best_t = t
                    best_b = b
            except Exception:
                continue
        _TUNED[dev_name] = best_b
        return (*_get_or_build_program(best_b), best_b)

    def opencl_matmul(a, b):
        import numpy as _np
        ctx, q = _ensure_ctx_queue()
        _, _K_MATMUL, _, _, BLOCK = _ensure_tuned_program()
        a = _np.array(a, dtype=_np.float32)
        b = _np.array(b, dtype=_np.float32)
        m, k = a.shape
        k2, n = b.shape
        assert k == k2
        c = _np.empty((m, n), dtype=_np.float32)
        mf = cl.mem_flags
        a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
        b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
        c_buf = cl.Buffer(ctx, mf.WRITE_ONLY, c.nbytes)
        # Launch with tuned tiled workgroup
        global_m = ((m + BLOCK - 1) // BLOCK) * BLOCK
        global_n = ((n + BLOCK - 1) // BLOCK) * BLOCK
        _K_MATMUL.set_args(_np.int32(m), _np.int32(n), _np.int32(k), a_buf, b_buf, c_buf)
        cl.enqueue_nd_range_kernel(q, _K_MATMUL, (global_m, global_n), (BLOCK, BLOCK))
        cl.enqueue_copy(q, c, c_buf)
        q.finish()
        return c

    def opencl_relu(x):
        import numpy as _np
        ctx, q = _ensure_ctx_queue()
        _, _, _K_RELU, _, _ = _ensure_tuned_program()
        x = _np.array(x, dtype=_np.float32).ravel()
        out = _np.empty_like(x)
        mf = cl.mem_flags
        a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x)
        out_buf = cl.Buffer(ctx, mf.WRITE_ONLY, out.nbytes)
        _K_RELU.set_args(a_buf, out_buf, _np.int32(x.size))
        cl.enqueue_nd_range_kernel(q, _K_RELU, (x.size,), None)
        cl.enqueue_copy(q, out, out_buf)
        q.finish()
        return out.reshape(-1)  # caller should reshape

    def opencl_conv2d(input_np, weight_np, stride=1, padding=0):
        # Correctness-first CPU fallback retained; prefer GPU path via fused conv2d+relu when feasible.
        import numpy as _np
        N, C, H, W = input_np.shape
        Cout, Cin, KH, KW = weight_np.shape
        assert C == Cin
        if stride != 1 and stride != 2:
            raise ValueError("opencl_conv2d currently supports stride 1 or 2")
        outH = (H + 2*padding - KH) // stride + 1
        outW = (W + 2*padding - KW) // stride + 1
        out = _np.zeros((N, Cout, outH, outW), dtype=_np.float32)
        for n in range(N):
            for co in range(Cout):
                for y in range(outH):
                    for x in range(outW):
                        float_acc = 0.0
                        for ci in range(Cin):
                            for ky in range(KH):
                                for kx in range(KW):
                                    int_iy = y*stride - padding + ky
                                    int_ix = x*stride - padding + kx
                                    if 0 <= int_iy < H and 0 <= int_ix < W:
                                        float_acc += input_np[n, ci, int_iy, int_ix] * weight_np[co, ci, ky, kx]
                        out[n, co, y, x] = float_acc
        return out

    def opencl_conv2d_relu(input_np, weight_np, stride=1, padding=0):
        """Conv2D+ReLU using im2col + OpenCL matmul + OpenCL relu.
        Falls back to CPU conv if OpenCL matmul is unavailable.
        """
        import numpy as _np
        x = _np.array(input_np, dtype=_np.float32, copy=False)
        w = _np.array(weight_np, dtype=_np.float32, copy=False)
        N, C, H, W = x.shape
        Cout, Cin, KH, KW = w.shape
        assert C == Cin, "Cin mismatch"
        outH = (H + 2*padding - KH) // stride + 1
        outW = (W + 2*padding - KW) // stride + 1
        # im2col with sliding_window_view for efficiency
        try:
            from numpy.lib.stride_tricks import sliding_window_view
            if padding > 0:
                x_pad = _np.pad(x, ((0,0),(0,0),(padding,padding),(padding,padding)), mode='constant')
            else:
                x_pad = x
            windows = sliding_window_view(x_pad, (KH, KW), axis=(2,3))  # (N, C, outH, outW, KH, KW)
            windows = windows[:, :, ::stride, ::stride, :, :]
            X_col = windows.reshape(N*outH*outW, C*KH*KW)
            W_col = w.reshape(Cout, C*KH*KW).T  # (C*KH*KW, Cout)
            # GPU matmul
            Y_col = opencl_matmul(X_col, W_col)  # (N*outH*outW, Cout)
            # ReLU on GPU
            Y_col = opencl_relu(Y_col.reshape(-1)).reshape(N*outH*outW, Cout)
            out = Y_col.reshape(N, outH, outW, Cout).transpose(0,3,1,2)
            return out
        except Exception:
            # Fallback: CPU conv then ReLU
            out = opencl_conv2d(input_np, weight_np, stride=stride, padding=padding)
            out = _np.maximum(out, 0.0, out)
            return out
else:
    def opencl_matmul(*args, **kwargs):
        raise RuntimeError("pyopencl not available")
    def opencl_relu(*args, **kwargs):
        raise RuntimeError("pyopencl not available")
    def opencl_conv2d(*args, **kwargs):
        raise RuntimeError("pyopencl not available")
    def opencl_conv2d_relu(*args, **kwargs):
        raise RuntimeError("pyopencl not available")
