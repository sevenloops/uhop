import os
import numpy as np
import pytest

try:
    import pyopencl as cl  # type: ignore
except Exception:
    cl = None

from uhop.backends.opencl_backend import is_opencl_available, opencl_matmul

pytestmark = pytest.mark.skipif(not is_opencl_available(), reason="OpenCL not available")
# Gate heavy debug diagnostics unless explicitly enabled
_RUN_DEBUG_TILED = os.environ.get("UHOP_RUN_DEBUG_TILED", "0").lower() in ("1", "true", "yes", "on")


def _run_shape(m, k, n):
    os.environ["UHOP_OPENCL_ENABLE_TILED"] = "1"
    os.environ["UHOP_OPENCL_MATMUL_IMPL"] = "tiled"
    A = np.random.randn(m, k).astype(np.float32)
    B = np.random.randn(k, n).astype(np.float32)
    C = opencl_matmul(A, B)
    ref = A @ B
    err = float(np.max(np.abs(ref - C)))
    return err, C


@pytest.mark.parametrize("m,k,n", [
    (64, 64, 64),            # square
    (96, 64, 80),            # rectangular
    (37, 19, 53),            # odd sizes
])
def test_tiled_validation_fallback_correct(m, k, n):
    err, C = _run_shape(m, k, n)
    # Even if tiled fails, fallback must yield correct result (small error)
    assert err < 1e-4, f"Fallback result incorrect for shape ({m},{k},{n}) err={err:.2e}"


@pytest.mark.skipif(not _RUN_DEBUG_TILED, reason="set UHOP_RUN_DEBUG_TILED=1 to run tiled debug diagnostics")
def test_tiled_debug_partial_accums():
    # Only run for a small square shape
    m = k = n = 64
    os.environ["UHOP_OPENCL_ENABLE_TILED"] = "1"
    os.environ["UHOP_OPENCL_MATMUL_IMPL"] = "tiled"

    # Acquire context/queue directly for debug kernel
    from uhop.backends.opencl_backend import _build_ctx_queue
    ctx, q = _build_ctx_queue()
    import pyopencl as cl  # type: ignore
    A = np.random.randn(m, k).astype(np.float32)
    B = np.random.randn(k, n).astype(np.float32)
    mf = cl.mem_flags
    a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
    b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
    tiles = (k + 16 - 1) // 16
    dbg_buf = cl.Buffer(ctx, mf.WRITE_ONLY, size=tiles * 16 * 16 * 4)
    # Build debug kernel
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "uhop", "kernels", "opencl"))
    debug_src = (open(os.path.join(base, "matmul_tiled_debug.cl")).read())
    flip = os.environ.get("UHOP_OPENCL_FLIP_GWS", "0").lower() in ("1", "true")
    build_opts = "-DTILE=16" + (" -DGWS_FLIP=1" if flip else "")
    prg = cl.Program(ctx, debug_src).build(options=build_opts)
    kn = cl.Kernel(prg, "matmul_tiled_debug")
    kn.set_args(a_buf, b_buf, dbg_buf, np.int32(m), np.int32(k), np.int32(n))
    gsz = (m, n) if flip else (n, m)
    lsz = (16, 16)
    evt = cl.enqueue_nd_range_kernel(q, kn, gsz, lsz)
    evt.wait()
    dbg = np.empty((tiles, 16, 16), dtype=np.float32)
    cl.enqueue_copy(q, dbg, dbg_buf)
    q.finish()
    # CPU-side per-tile partials for the top-left C tile (only strict when not flipped)
    if not flip:
        for t in range(tiles):
            L = min(16, k - t * 16)
            At = np.zeros((16, 16), dtype=np.float32)
            Bt = np.zeros((16, 16), dtype=np.float32)
            if L > 0:
                At[:, :L] = A[:16, t * 16 : t * 16 + L]
                Bt[:L, :] = B[t * 16 : t * 16 + L, :16]
            cpu_tile = At @ Bt
            err = float(np.max(np.abs(cpu_tile - dbg[t])))
            assert err < 1e-4, f"Per-tile partial mismatch at t={t}: err={err:.2e}"
        # Sum of per-tile partials should equal the top-left C tile
        cpu_top_left = A[:16, :] @ B[:, :16]
        dbg_sum = np.sum(dbg, axis=0)
        total_err = float(np.max(np.abs(cpu_top_left - dbg_sum)))
        assert total_err < 1e-4, f"Sum of per-tile partials mismatch: err={total_err:.2e}"
    else:
        # Relaxed: ensure non-zero activity only; flipped mapping is exploratory and may differ until kernel adjusted
        assert np.any(dbg != 0), "Flipped GWS produced all-zero debug buffer"

    # Ensure we captured some non-zero activity in first tile
    assert np.any(dbg[0] != 0), "Debug partial accumulators are all zero; kernel may not be executing correctly"

    # Optional deeper dump: compare loaded As/Bs for a specific tile index
    dump_t = int(os.environ.get("UHOP_OCL_DUMP_T", "-1"))
    if 0 <= dump_t < tiles:
        dump_src = (open(os.path.join(base, "matmul_tiled_dump_loads.cl")).read())
        build_opts2 = "-DTILE=16" + (" -DGWS_FLIP=1" if flip else "")
        prg2 = cl.Program(ctx, dump_src).build(options=build_opts2)
        kn2 = cl.Kernel(prg2, "matmul_tiled_dump_loads")
        as_buf = cl.Buffer(ctx, mf.WRITE_ONLY, size=16 * 16 * 4)
        bs_buf = cl.Buffer(ctx, mf.WRITE_ONLY, size=16 * 16 * 4)
        kn2.set_args(a_buf, b_buf, as_buf, bs_buf, np.int32(m), np.int32(k), np.int32(n), np.int32(dump_t))
        gsz2 = (m, n) if flip else (n, m)
        evt2 = cl.enqueue_nd_range_kernel(q, kn2, gsz2, (16, 16))
        evt2.wait()
        As_loaded = np.empty((16, 16), dtype=np.float32)
        Bs_loaded = np.empty((16, 16), dtype=np.float32)
        cl.enqueue_copy(q, As_loaded, as_buf)
        cl.enqueue_copy(q, Bs_loaded, bs_buf)
        q.finish()
        L = min(16, k - dump_t * 16)
        At = np.zeros((16, 16), dtype=np.float32)
        Bt = np.zeros((16, 16), dtype=np.float32)
        if L > 0:
            At[:, :L] = A[:16, dump_t * 16 : dump_t * 16 + L]
            Bt[:L, :] = B[dump_t * 16 : dump_t * 16 + L, :16]
        assert np.allclose(As_loaded, At, atol=1e-6), "Loaded As tile mismatch with CPU slice"
        assert np.allclose(Bs_loaded, Bt, atol=1e-6), "Loaded Bs tile mismatch with CPU slice"


@pytest.mark.skipif(not _RUN_DEBUG_TILED, reason="set UHOP_RUN_DEBUG_TILED=1 to run tiled debug diagnostics")
def test_tiled_debug_partial_accums_groups():
    # Probe multiple workgroup tiles on a rectangular shape to isolate cross-group issues
    m, k, n = 96, 64, 80
    os.environ["UHOP_OPENCL_ENABLE_TILED"] = "1"
    os.environ["UHOP_OPENCL_MATMUL_IMPL"] = "tiled"
    from uhop.backends.opencl_backend import _build_ctx_queue
    ctx, q = _build_ctx_queue()
    import pyopencl as cl  # type: ignore
    A = np.random.randn(m, k).astype(np.float32)
    B = np.random.randn(k, n).astype(np.float32)
    tiles_x = (n + 16 - 1) // 16
    tiles_y = (m + 16 - 1) // 16
    mf = cl.mem_flags
    a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
    b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "uhop", "kernels", "opencl"))
    dbg_src = (open(os.path.join(base, "matmul_tiled_debug_group.cl")).read())
    prg = cl.Program(ctx, dbg_src).build(options="-DTILE=16")
    kn = cl.Kernel(prg, "matmul_tiled_debug_group")
    # Test a few representative groups: top-left, edge, bottom-right
    groups = [(0, 0), (tiles_x - 1, 0), (0, tiles_y - 1), (tiles_x - 1, tiles_y - 1)]
    for gx, gy in groups:
        dbg_buf = cl.Buffer(ctx, mf.WRITE_ONLY, size=((k + 15)//16) * 16 * 16 * 4)
        kn.set_args(a_buf, b_buf, dbg_buf, np.int32(m), np.int32(k), np.int32(n), np.int32(gx), np.int32(gy))
        evt = cl.enqueue_nd_range_kernel(q, kn, (16, 16), (16, 16))
        evt.wait()
        tiles_k = (k + 16 - 1) // 16
        dbg = np.empty((tiles_k, 16, 16), dtype=np.float32)
        cl.enqueue_copy(q, dbg, dbg_buf)
        q.finish()
        # CPU partials for this tile
        row0 = gy * 16
        col0 = gx * 16
        for t in range(tiles_k):
            L = min(16, k - t * 16)
            At = np.zeros((16, 16), dtype=np.float32)
            Bt = np.zeros((16, 16), dtype=np.float32)
            if L > 0:
                rs = min(16, m - row0)
                cs = min(16, n - col0)
                At[:rs, :L] = A[row0:row0 + rs, t * 16: t * 16 + L]
                Bt[:L, :cs] = B[t * 16: t * 16 + L, col0: col0 + cs]
            cpu_tile = At @ Bt
            err = float(np.max(np.abs(cpu_tile - dbg[t])))
            assert err < 1e-4, f"Group ({gx},{gy}) per-tile mismatch at t={t}: err={err:.2e}"
        # Sum-of-partials equals that C tile
        rs = min(16, m - row0)
        cs = min(16, n - col0)
        cpu_C_tile = (A[row0:row0+rs, :] @ B[:, col0:col0+cs])
        dbg_sum = np.sum(dbg, axis=0)[:rs, :cs]
        terr = float(np.max(np.abs(cpu_C_tile - dbg_sum)))
        assert terr < 1e-4, f"Group ({gx},{gy}) sum-of-partials mismatch: err={terr:.2e}"
