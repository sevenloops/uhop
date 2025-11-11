"""
uhop/validation.py

Kernel correctness validation utilities.

This module provides helpers to validate candidate kernel callables against
reference implementations over randomized inputs and edge cases. It is
framework-agnostic: inputs/outputs can be NumPy arrays or torch tensors; the
validator converts to/from NumPy as needed for comparison.

Public API:
 - validate_callable(candidate, reference, input_specs,
    tol=None, cases=..., strict=False)
 - dtype_tolerances(dtype)
 - gen_cases(input_specs, extra_edge_cases=True)

Where input_specs is a list of dicts describing each input::
    {"shape": (M, N, ...), "dtype": np.float32}

The candidate/reference callables are invoked as fn(*inputs, **kwargs) and
should return an array-like (NumPy or torch). When strict=True, both abs and
relative tolerances are tightened by 10x as a safety gate.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def _to_numpy(x: Any) -> np.ndarray:
    try:
        import torch  # type: ignore

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.array(x)


def dtype_tolerances(dtype: Any) -> Tuple[float, float]:
    """Return (atol, rtol) suggested for dtype."""
    if dtype in (np.float32, "float32"):  # common default
        return (1e-5, 1e-3)
    if dtype in (np.float16, "float16"):  # fp16 is noisy
        return (5e-3, 1e-2)
    if dtype in (np.float64, "float64"):
        return (1e-8, 1e-6)
    if dtype in (np.int32, np.int64, "int32", "int64"):
        return (0.0, 0.0)
    # fallback
    return (1e-5, 1e-3)


def _edge_values(dtype: Any) -> Iterable[float]:
    if dtype in (
        np.float16,
        np.float32,
        np.float64,
        "float16",
        "float32",
        "float64",
    ):
        return [0.0, 1.0, -1.0, 1e-3, -1e-3]
    return [0, 1, -1]


def gen_cases(
    input_specs: Sequence[Dict[str, Any]],
    extra_edge_cases: bool = True,
    seed: int = 0,
) -> List[List[np.ndarray]]:
    """
    Generate a small set of input cases for validation.

    input_specs: list of {shape, dtype}
    Returns list of test cases (each case is a list of numpy arrays).
    """
    rng = np.random.default_rng(seed)
    cases: List[List[np.ndarray]] = []

    # Randomized cases
    for i in range(3):
        cur: List[np.ndarray] = []
        for spec in input_specs:
            shape = tuple(int(s) for s in spec["shape"])
            dtype = spec["dtype"]
            if np.issubdtype(np.dtype(dtype), np.floating):
                arr = rng.standard_normal(shape).astype(dtype)
            else:
                arr = rng.integers(low=-2, high=3, size=shape, dtype=dtype)
            cur.append(arr)
        cases.append(cur)

    if not extra_edge_cases:
        return cases

    # Edge-case mixtures: zeros and small values
    cur: List[np.ndarray] = []
    for spec in input_specs:
        shape = tuple(int(s) for s in spec["shape"])
        dtype = spec["dtype"]
        arr = np.zeros(shape, dtype=dtype)
        cur.append(arr)
    cases.append(cur)

    cur = []
    for spec in input_specs:
        shape = tuple(int(s) for s in spec["shape"])
        dtype = spec["dtype"]
        ev = list(_edge_values(dtype))
        needed = int(np.prod(shape))
        reps = needed // len(ev) + 1
        flat = np.array(ev * reps, dtype=dtype)[:needed]
        arr = flat.reshape(shape)
        cur.append(arr)
    cases.append(cur)

    return cases


@dataclass
class ValidationResult:
    ok: bool
    max_abs_err: float
    max_rel_err: float
    case_index: int | None = None
    message: str = ""


def validate_callable(
    candidate: Callable[..., Any],
    reference: Callable[..., Any],
    input_specs: Sequence[Dict[str, Any]],
    *,
    tol: Tuple[float, float] | None = None,
    cases: List[List[np.ndarray]] | None = None,
    strict: bool = False,
    kwargs: Dict[str, Any] | None = None,
) -> ValidationResult:
    """
    Validate candidate against reference over a small suite of cases.

    Returns ValidationResult. When strict=True, tolerances are tightened.
    """
    if cases is None:
        cases = gen_cases(input_specs)
    if kwargs is None:
        kwargs = {}

    # Default tolerances by dtype of the first input
    if tol is None:
        first_dtype = input_specs[0]["dtype"] if input_specs else np.float32
        tol = dtype_tolerances(first_dtype)
    atol, rtol = tol
    if strict:
        atol *= 0.1
        rtol *= 0.1

    worst_abs = 0.0
    worst_rel = 0.0
    worst_idx: int | None = None

    for i, args in enumerate(cases):
        try:
            ref = reference(*args, **kwargs)
            cand = candidate(*args, **kwargs)
        except Exception as e:
            msg = f"execution failed: {e}"
            return ValidationResult(False, worst_abs, worst_rel, i, msg)
        ref_np = _to_numpy(ref)
        cand_np = _to_numpy(cand)
        if ref_np.shape != cand_np.shape:
            msg = f"shape mismatch: {ref_np.shape} vs {cand_np.shape}"
            return ValidationResult(False, worst_abs, worst_rel, i, msg)
        abs_err = np.max(np.abs(ref_np - cand_np)) if ref_np.size else 0.0
        denom = np.maximum(np.abs(ref_np), 1e-12)
        rel_err = np.max(np.abs((ref_np - cand_np) / denom)) if ref_np.size else 0.0
        worst_abs = max(worst_abs, float(abs_err))
        worst_rel = max(worst_rel, float(rel_err))
        if not (abs_err <= atol or rel_err <= rtol):
            worst_idx = i
            return ValidationResult(
                False,
                float(abs_err),
                float(rel_err),
                worst_idx,
                "tolerance exceeded",
            )
    return ValidationResult(
        True,
        float(worst_abs),
        float(worst_rel),
        None,
        "ok",
    )


DEFAULT_SEED = 1337


def make_seeded_input(
    seed: int,
    shape: Tuple[int, ...],
    dtype: Any = np.float32,
) -> np.ndarray:
    """Generate deterministic random input based on the seed and shape."""
    rng = np.random.RandomState(seed)
    if np.issubdtype(dtype, np.floating):
        return rng.rand(*shape).astype(dtype)
    elif np.issubdtype(dtype, np.integer):
        return rng.randint(0, 10, size=shape, dtype=dtype)
    else:
        return rng.rand(*shape).astype(np.float32)


def validate_kernel(
    candidate_runner: Callable,
    reference_fn: Callable,
    example_shape: List[Tuple[Tuple[int, ...], Any]],
    ntests: int = 5,
    atol: float = 1e-6,
    rtol: float = 1e-5,
    dump_dir: Optional[str] = None,
    **kwargs: Any,
) -> bool:
    """Validate a kernel against a reference function (compat wrapper).

    Accepts legacy kwargs like ``tests`` and ignores ``nargs`` for
    backward-compatibility with older optimizer code paths.
    """
    # Back-compat for callers using alternative arg names
    if "tests" in kwargs and isinstance(kwargs.get("tests"), int):
        ntests = int(kwargs["tests"])  # type: ignore
    # ``nargs`` was informational only in older flows; ignore if present
    _ = kwargs.get("nargs")
    # Minimal boolean pass/fail API for current optimizer usage
    seed = DEFAULT_SEED
    for test_idx in range(ntests):
        inputs: List[np.ndarray] = []
        case_seed = seed + test_idx
        for shape, dtype in example_shape:
            inp = make_seeded_input(case_seed, shape, dtype)
            inputs.append(inp)
            case_seed += 1  # change seed for next input
        try:
            out_c = candidate_runner(*[np.copy(inp) for inp in inputs])
        except Exception:
            if dump_dir:
                _dump_case(
                    dump_dir,
                    inputs,
                    out_c=None,
                    out_r=None,
                    suffix=f"candidate_exception_{test_idx}",
                )
            return False
        try:
            out_r = reference_fn(*[np.copy(inp) for inp in inputs])
        except Exception:
            if dump_dir:
                _dump_case(
                    dump_dir,
                    inputs,
                    out_c=None,
                    out_r=None,
                    suffix=f"reference_exception_{test_idx}",
                )
            return False
        if not _allclose(out_c, out_r, atol, rtol):
            if dump_dir:
                _dump_case(
                    dump_dir,
                    inputs,
                    out_c=out_c,
                    out_r=out_r,
                    suffix=f"mismatch_{test_idx}",
                )
            return False
    return True


def _allclose(a: np.ndarray, b: np.ndarray, atol: float, rtol: float) -> bool:
    """Check if two arrays are close within given tolerances."""
    try:
        return np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=True)
    except Exception:
        return np.array_equal(a, b)


def _dump_case(
    dump_dir: str,
    inputs: List[np.ndarray],
    out_c: Optional[np.ndarray],
    out_r: Optional[np.ndarray],
    suffix: Optional[str] = None,
):
    os.makedirs(dump_dir, exist_ok=True)
    marker = int(time.time() * 1000)
    fname = f"case_{marker}_{suffix}.npz" if suffix else f"case_{marker}.npz"
    path = os.path.join(dump_dir, fname)
    save_kwargs = {f"in_{i}": v for i, v in enumerate(inputs)}
    if out_c is not None:
        save_kwargs["out_candidate"] = out_c
    if out_r is not None:
        save_kwargs["out_reference"] = out_r
    np.savez_compressed(path, **save_kwargs)
    print(f"[UHOP][Validation] Dumped failing case to {path}")
    return path
