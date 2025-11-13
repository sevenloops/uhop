import pytest

from uhop.backends.opencl_backend import is_opencl_available
from uhop.cache import OPENCL_BUFFER_POOL

pytestmark = pytest.mark.skipif(not is_opencl_available(), reason="OpenCL not available")


def test_buffer_pool_stats_and_reuse():
    # Acquire context via backend internal builder
    from uhop.backends.opencl_backend import _build_ctx_queue

    ctx, _ = _build_ctx_queue()
    import pyopencl as cl  # type: ignore

    mf = cl.mem_flags
    # Request same size/flags twice -> one miss then hit
    buf1 = OPENCL_BUFFER_POOL.get(ctx, 1024, mf.READ_ONLY)
    buf2 = OPENCL_BUFFER_POOL.get(ctx, 1024, mf.READ_ONLY)
    assert buf1.int_ptr == buf2.int_ptr
    stats = OPENCL_BUFFER_POOL.stats()
    assert stats["entries"] >= 1
    assert stats["hits"] >= 1
    OPENCL_BUFFER_POOL.clear()
    stats2 = OPENCL_BUFFER_POOL.stats()
    assert stats2["entries"] == 0
