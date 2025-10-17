import numpy as np

from uhop.validation import (
    validate_callable,
    validate_kernel,
)


def _spec(shape, dtype=np.float32):
    return {"shape": tuple(shape), "dtype": dtype}


def test_validate_callable_pass():
    def ref(A, B):
        return A @ B

    def cand(A, B):
        # identical implementation
        return np.matmul(A, B)

    specs = [_spec((8, 8)), _spec((8, 8))]
    res = validate_callable(cand, ref, specs)
    assert res.ok, f"expected ok, got: {res}"


def test_validate_callable_fail():
    def ref(A, B):
        return A + B

    def cand(A, B):
        # introduce a noticeable error
        return A + 1.1 * B

    specs = [_spec((16,)), _spec((16,))]
    res = validate_callable(cand, ref, specs)
    assert not res.ok, "expected validation to fail"


def test_validate_callable_strict_tightens():
    # Candidate deviates by ~2e-4 relative error -> passes default (rtol=1e-3)
    # but should fail with strict (rtol tightened to 1e-4)
    def ref(x):
        return x * 3.14

    def cand(x):
        return x * 3.14 * (1.0002)

    specs = [_spec((64,), np.float32)]
    loose = validate_callable(cand, ref, specs, strict=False)
    strict = validate_callable(cand, ref, specs, strict=True)
    assert loose.ok, "should pass with default tolerances"
    assert not strict.ok, "should fail with strict tolerances"


def test_validate_kernel_wrapper_basic():
    # Legacy wrapper path: simple add kernel
    def runner(A, B):
        return A + B

    def reference(A, B):
        return np.add(A, B)

    shapes = [((4, 4), np.float32), ((4, 4), np.float32)]
    ok = validate_kernel(runner, reference, shapes, ntests=3)
    assert ok, "validate_kernel should succeed on identical add"
