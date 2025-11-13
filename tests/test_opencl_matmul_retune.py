import json
import os

import numpy as np
import pytest

try:
    import pyopencl as cl  # type: ignore
except Exception:  # pragma: no cover
    cl = None  # type: ignore

from uhop.backends.opencl.matmul import MatmulOp
from uhop.cache import CACHE_DIR

pytestmark = pytest.mark.skipif(cl is None, reason="pyopencl not available")

# We focus on the autotune retune trigger logic. Strategy:
# 1. Run a small matmul to create initial params (tuned_at saved).
# 2. Manually mark retune_suggested in autotune entry.
# 3. Run again and assert tuned_at value changed (retune occurred).


def _autotune_key(device: str, shape_key: str) -> str:
    return f"opencl|matmul|matmul_tiled|{device}|{shape_key}"


def test_opencl_matmul_retune_cycle():
    # Skip if environment variable forces naive path
    if str(os.environ.get("UHOP_OPENCL_FORCE_NAIVE") or "0").lower() in ("1", "true", "yes", "on"):
        pytest.skip("Naive forced; retune path irrelevant")
    op = MatmulOp()
    A = np.random.rand(32, 64).astype(np.float32)
    B = np.random.rand(64, 48).astype(np.float32)
    res1 = op.execute(A, B)
    assert res1.output.shape == (32, 48)
    # Load autotune file
    at_file = CACHE_DIR / "autotune.json"
    data = json.loads(at_file.read_text()) if at_file.exists() else {}
    # Find entry
    shape_key = "N32_M64_K48"
    # Device name retrieval is duplicated from current_device_name() logic inline for test robustness
    device = None
    try:
        import pyopencl as _cl

        dev = _cl.get_platforms()[0].get_devices()[0]
        device = dev.name.strip()
    except Exception:
        device = "device"
    k = _autotune_key(device, shape_key)
    entry = data.get(k)
    assert isinstance(entry, dict), "Autotune entry should exist after first run"
    first_tuned_at = entry.get("tuned_at")
    # Inject retune suggestion
    entry["retune_suggested"] = True
    data[k] = entry
    at_file.write_text(json.dumps(data, indent=2))
    # Re-run; should retrigger tuning producing new tuned_at
    res2 = op.execute(A, B)
    data2 = json.loads(at_file.read_text())
    entry2 = data2.get(k)
    assert isinstance(entry2, dict)
    second_tuned_at = entry2.get("tuned_at")
    # tuned_at should differ (retune happened)
    assert second_tuned_at != first_tuned_at, f"Expected retune; tuned_at unchanged ({first_tuned_at})"
    assert res2.output.shape == (32, 48)
