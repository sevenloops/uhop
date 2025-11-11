# tests/test_optimizer_backend_policy.py
import numpy as np

from uhop.optimizer import UHopOptimizer


def test_gpu_first_policy_cpu_only(monkeypatch):
    hop = UHopOptimizer()
    # Simulate: no triton, no opencl, torch available but CPU only
    monkeypatch.setattr("uhop.backends.triton_backend._TRITON_AVAILABLE", False, raising=False)
    monkeypatch.setattr("uhop.backends.opencl_backend._OPENCL_AVAILABLE", False, raising=False)
    from uhop.backends import torch_backend

    monkeypatch.setattr(torch_backend, "_torch_device_preference", lambda: None, raising=False)

    @hop.optimize("matmul")
    def matmul_np(A, B):
        return np.array(A) @ np.array(B)

    A = np.eye(2, dtype=np.float32)
    B = np.eye(2, dtype=np.float32)
    R = matmul_np(A, B)
    assert np.allclose(R, A @ B)
