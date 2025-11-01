from pathlib import Path

import numpy as np
import pytest

try:
    import cupy as cp  # type: ignore
except Exception:
    cp = None

from uhop import autotuner
from uhop.backends import cupy_wrapper

pytestmark = pytest.mark.skipif(
    cp is None, reason="cupy required to run CUDA autotuner tests"
)


def test_autotune_add_and_correctness(tmp_path):
    # ensure cache cleared for test device/op
    cache_file = Path("uhop/cache") / "cuda" / "add.json"
    if cache_file.exists():
        cache_file.unlink()

    size = 1_000_00  # 100k elements
    best = autotuner.get_cached_or_tune(
        "add", size=size, dtype="float32", device="cuda"
    )
    assert "latency_s" in best
    assert best["latency_s"] > 0

    # compile the selected kernel and run one correctness check
    template_path = Path("uhop/kernels/cuda/elementwise_add.cu.jinja")
    context = best["kernel_source_context"]
    src = template_path.read_text()
    from jinja2 import Template

    source = Template(src).render(**context)
    kernel = cupy_wrapper.CupyKernel(source, context.get("KERNEL_NAME", "elem_op"))

    a = np.random.rand(size).astype("float32")
    b = np.random.rand(size).astype("float32")
    da = cp.asarray(a)
    db = cp.asarray(b)
    dout = cp.empty_like(da)

    block = best["block"]
    grid = best["grid"]
    kernel.launch((grid, 1, 1), (block, 1, 1), (da, db, dout, size))
    cp.cuda.get_current_stream().synchronize()
    got = dout.get()
    expect = a + b
    # numeric compare
    assert np.allclose(got, expect, atol=1e-5, rtol=1e-4)
