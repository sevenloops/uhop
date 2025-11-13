import numpy as np

from uhop import config as _cfg
from uhop import optimize


@optimize("matmul")
def matmul_np(a, b):
    return np.array(a) @ np.array(b)


def _backend_used(cache_entry):
    return cache_entry.get("backend") if isinstance(cache_entry, dict) else None


def test_force_baseline_env(monkeypatch):
    monkeypatch.setenv("UHOP_FORCE_BASELINE", "1")
    # Clear any prior decision cache
    from uhop.cache import UhopCache as _Cache

    c = _Cache()
    c.clear()
    a = np.random.default_rng(0).random((8, 8), dtype=np.float32)
    b = np.random.default_rng(1).random((8, 8), dtype=np.float32)
    out = matmul_np(a, b)
    ref = a @ b
    assert np.allclose(out, ref, atol=1e-5)
    # Ensure optimizer chose NumPy baseline backend
    entry = c.get("matmul")
    # When forcing baseline, optimizer may return early without caching.
    if entry is not None:
        assert _backend_used(entry) in ("numpy", "cpu", "baseline"), f"Unexpected backend: {entry}"


def test_force_baseline_programmatic(monkeypatch):
    monkeypatch.delenv("UHOP_FORCE_BASELINE", raising=False)
    from uhop.cache import UhopCache as _Cache

    c = _Cache()
    c.clear()
    _cfg.set("UHOP_FORCE_BASELINE", "1")
    a = np.random.default_rng(2).random((4, 4), dtype=np.float32)
    b = np.random.default_rng(3).random((4, 4), dtype=np.float32)
    out = matmul_np(a, b)
    ref = a @ b
    assert np.allclose(out, ref, atol=1e-5)
    entry = c.get("matmul")
    if entry is not None:
        assert _backend_used(entry) in ("numpy", "cpu", "baseline"), f"Unexpected backend: {entry}"
