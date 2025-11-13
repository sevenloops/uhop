import numpy as np

from uhop import UHopOptimizer


def test_env_backend_preference_numpy_baseline(monkeypatch):
    # Force baseline selection regardless of accelerators
    monkeypatch.setenv("UHOP_BACKEND_PREFERENCE", "numpy")

    hop = UHopOptimizer()

    @hop.optimize("matmul")
    def mm(a, b):
        return np.array(a) @ np.array(b)

    A = np.array([[1.0, 2.0]], dtype=np.float32)
    B = np.array([[3.0], [4.0]], dtype=np.float32)
    out = mm(A, B)
    assert np.allclose(out, np.array([[11.0]], dtype=np.float32))
