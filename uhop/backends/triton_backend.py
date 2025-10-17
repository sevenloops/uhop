# uhop/backends/triton_backend.py
"""
Optional Triton backend examples.

Provides minimal matmul and relu helpers when Triton is installed. These are
demonstrative and intentionally small; UHOP primarily relies on PyTorch and
OpenCL backends. Conv2D via Triton is omitted for brevity.
"""


def is_triton_available():
    try:
        import triton  # type: ignore  # noqa: F401
        return True
    except Exception:
        return False


def triton_matmul(*args, **kwargs):
    if not is_triton_available():
        raise RuntimeError("triton not available")
    # Lazy import to avoid linter errors when Triton isn't installed
    import numpy as np  # type: ignore
    import triton  # type: ignore
    import triton.language as tl  # type: ignore

    @triton.jit
    def _kernel(
            A_ptr,
            B_ptr,
            C_ptr,
            M,
            N,
            K,
            sa0,
            sa1,
            sb0,
            sb1,
            sc0,
            sc1,
            BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs_m = pid * BLOCK + tl.arange(0, BLOCK)
        offs_n = tl.arange(0, BLOCK)
        A = tl.load(A_ptr + offs_m[:, None] * sa0 + offs_n[None, :] * sa1)
        B = tl.load(B_ptr + offs_m[:, None] * sb0 + offs_n[None, :] * sb1)
        C = tl.dot(A, B)
        tl.store(C_ptr + offs_m[:, None] * sc0 + offs_n[None, :] * sc1, C)

    a, b = args
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    C = np.zeros((M, N), dtype=np.float32)
    BLOCK = 64
    grid = ((M + BLOCK - 1) // BLOCK,)
    _kernel[(grid,)](
        a,
        b,
        C,
        M,
        N,
        K,
        a.strides[0] // 4,
        a.strides[1] // 4,
        b.strides[0] // 4,
        b.strides[1] // 4,
        C.strides[0] // 4,
        C.strides[1] // 4,
        BLOCK=BLOCK,
    )
    return C


def triton_relu(x):
    if not is_triton_available():
        raise RuntimeError("triton not available")
    import numpy as np  # type: ignore
    import torch  # type: ignore
    t = torch.from_numpy(np.array(x)).cuda()
    return torch.relu(t).cpu().numpy()


def triton_conv2d(*args, **kwargs):
    raise NotImplementedError("triton conv2d helper not implemented here")
