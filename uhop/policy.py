"""Backend policy layer.

Provides a unified interface for selecting the best backend for an op using
preference order, cached benchmarking, and lightweight validation hooks.

This is an initial MVP that can be expanded with richer scoring (variance,
resource constraints, instability flags).
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

import numpy as _np

from . import config as _cfg
from .backends import (
    is_opencl_available,
    is_torch_available,
    is_triton_available,
    opencl_conv2d,
    opencl_matmul,
    opencl_relu,
    torch_conv2d,
    torch_matmul,
    torch_relu,
    triton_matmul,
    triton_relu,
)
from .backends.registry import ensure_default_backends_registered
from .cache import UhopAutotune
from .hardware import detect_hardware
from .utils.logging import get_logger as _get_logger

_log = _get_logger("uhop.policy")


@dataclass
class PolicySelection:
    backend_name: str
    policy_reason: str
    run: Callable[..., Any]
    latency_ms: float | None = None


class BackendPolicy:
    def __init__(self, backend_manager, cache):
        self.backend_manager = backend_manager
        self.cache = cache
        # Static preference order fallback if no registry yet
        self.default_order: List[str] = ["torch", "triton", "opencl", "cpu", "numpy"]
        # Benchmark mode configuration (can be overridden via env):
        # UHOP_POLICY_BENCH_ITERS, UHOP_POLICY_BENCH_WARMUP, UHOP_POLICY_BENCH_EARLY, UHOP_POLICY_MODE

    def _candidate_backends(self, op: str) -> List[str]:
        pref = _cfg.get("UHOP_BACKEND_PREFERENCE")
        if pref:
            return [p.strip().lower() for p in str(pref).split(",") if p.strip()]
        # If backend manager has real registrations, derive order by availability
        try:
            ensure_default_backends_registered()
            names = []
            for name in ["torch", "triton", "opencl"]:
                try:
                    b = getattr(self.backend_manager, "get_backend", lambda x: None)(name)
                    if b and b.capabilities.available:
                        names.append(name)
                except Exception:
                    continue
            if names:
                # Append cpu/numpy fallbacks implicitly
                names.extend(["cpu", "numpy"])  # maintain baseline entries
                return self._apply_hardware_bias(names)
        except Exception:
            pass
        return self._apply_hardware_bias(list(self.default_order))

    def _apply_hardware_bias(self, order: List[str]) -> List[str]:
        try:
            hw = detect_hardware()
        except Exception:
            hw = None
        if hw is None:
            return order

        vendor = (getattr(hw, "vendor", "") or "").lower()
        kind = (getattr(hw, "kind", "") or "").lower()
        if vendor == "amd" and kind.startswith("opencl"):
            return self._move_front(order, "opencl")
        return order

    @staticmethod
    def _move_front(order: List[str], backend: str) -> List[str]:
        if backend not in order:
            return order
        reordered = [backend]
        reordered.extend([name for name in order if name != backend])
        return reordered

    def select(self, op: str, args, kwargs) -> Optional[PolicySelection]:
        # Forced baseline bypass
        if bool(_cfg.get("UHOP_FORCE_BASELINE")):
            return None
        mode = str(_cfg.get("UHOP_POLICY_MODE") or "order_probe").lower()
        order = self._candidate_backends(op)
        at = None
        try:
            at = UhopAutotune()
        except Exception:
            at = None
        hw = None
        try:
            hw = detect_hardware()
        except Exception:
            hw = None

        def _shape_key_from_args(aargs) -> str:
            def _sig_from_val(v):
                try:
                    import torch  # type: ignore

                    if isinstance(v, torch.Tensor):
                        dev = getattr(v.device, "type", "cpu")
                        return (
                            "torch",
                            tuple(int(x) for x in v.shape),
                            str(v.dtype).replace("torch.", ""),
                            dev,
                        )
                except Exception:
                    pass
                try:
                    if isinstance(v, _np.ndarray):
                        return ("numpy", tuple(int(x) for x in v.shape), str(v.dtype))
                except Exception:
                    pass
                return (type(v).__name__,)

            parts = [str(_sig_from_val(aargs[0]))]
            if len(aargs) > 1:
                parts.append(str(_sig_from_val(aargs[1])))
            return ";".join(parts)

        device_name = getattr(hw, "name", "unknown")
        shape_key = _shape_key_from_args(list(args)) if args else op

        if mode != "benchmark":
            # Original order-probe logic
            for name in order:
                runner = self._get_runner(name, op, args, kwargs)
                if runner is None:
                    continue
                t0 = time.perf_counter()
                try:
                    _ = runner(*args, **kwargs)
                except Exception:
                    continue
                dt_ms = (time.perf_counter() - t0) * 1000.0
                # Record profile sample and decision trace for winner
                try:
                    if at is not None:
                        at.record_profile(
                            name, op, "policy_select", device_name, shape_key, gflops=0.0, ms=float(dt_ms)
                        )
                        at.set_params(
                            name,
                            op,
                            "policy_select",
                            device_name,
                            shape_key,
                            {
                                "decision_trace": {
                                    "mode": "order_probe",
                                    "winner": name,
                                    "latency_ms": float(dt_ms),
                                },
                                "thresholds": {"retune_rel_var": 0.15},
                            },
                        )
                except Exception:
                    pass
                return PolicySelection(backend_name=name, policy_reason="order_probe", run=runner, latency_ms=dt_ms)
            return None
        # Benchmark mode: run limited iterations for each viable backend and pick fastest median
        try:
            iters = int(_cfg.get("UHOP_POLICY_BENCH_ITERS") or 3)
        except Exception:
            iters = 3
        try:
            warmup = int(_cfg.get("UHOP_POLICY_BENCH_WARMUP") or 1)
        except Exception:
            warmup = 1
        try:
            early_factor = float(
                _cfg.get("UHOP_POLICY_BENCH_EARLY") or 2.5
            )  # early exit if current winner < next candidate first run / factor
        except Exception:
            early_factor = 2.5
        timings: List[tuple[str, float, Callable]] = []
        winner_latency = None
        winner_name = None
        winner_runner = None
        for name in order:
            runner = self._get_runner(name, op, args, kwargs)
            if runner is None:
                continue
            # Warmup
            for _w in range(max(0, warmup)):
                try:
                    _ = runner(*args, **kwargs)
                except Exception:
                    runner = None
                    break
            if runner is None:
                continue
            times = []
            failed = False
            for i in range(max(1, iters)):
                t0 = time.perf_counter()
                try:
                    _ = runner(*args, **kwargs)
                except Exception:
                    failed = True
                    break
                dt_ms = (time.perf_counter() - t0) * 1000.0
                times.append(dt_ms)
                # Early exit heuristic: if we already have a winner and current first iteration slower by large factor, skip remaining iterations
                if i == 0 and winner_latency is not None and dt_ms > winner_latency * early_factor:
                    break
            if failed or not times:
                continue
            # Median latency
            med = statistics.median(times)
            # Record profile sample for each candidate (median)
            try:
                if at is not None:
                    at.record_profile(name, op, "policy_select", device_name, shape_key, gflops=0.0, ms=float(med))
            except Exception:
                pass
            timings.append((name, med, runner))
            if winner_latency is None or med < winner_latency:
                winner_latency = med
                winner_name = name
                winner_runner = runner
        if winner_name is None:
            return None
        # Store decision trace for winner including competing candidates
        try:
            if at is not None:
                at.set_params(
                    winner_name,
                    op,
                    "policy_select",
                    device_name,
                    shape_key,
                    {
                        "decision_trace": {
                            "mode": "benchmark",
                            "winner": winner_name,
                            "winner_ms": float(winner_latency),
                            "candidates": [{"name": n, "median_ms": float(m)} for (n, m, _r) in timings],
                        },
                        "thresholds": {"retune_rel_var": 0.15},
                    },
                )
        except Exception:
            pass
        return PolicySelection(
            backend_name=winner_name, policy_reason="benchmark", run=winner_runner, latency_ms=winner_latency
        )

    def _get_runner(self, backend: str, op: str, args, kwargs) -> Optional[Callable]:
        try:

            def _ensure_np32(x):
                try:
                    return _np.array(x, dtype=_np.float32)
                except Exception:
                    return x

            if backend in ("torch", "cpu") and is_torch_available():
                if op == "matmul":
                    return lambda a, b, **kw: torch_matmul(a, b, keep_format=kw.get("keep_format"))
                if op == "conv2d":
                    return lambda a, b, **kw: torch_conv2d(
                        a,
                        b,
                        stride=kwargs.get("stride", 1),
                        padding=kwargs.get("padding", 0),
                        keep_format=kw.get("keep_format"),
                    )
                if op == "relu":
                    return lambda a, **kw: torch_relu(a, keep_format=kw.get("keep_format"))
            if backend == "triton" and is_triton_available():
                if op == "matmul":
                    # Ensure NumPy float32 inputs for fair probing
                    return lambda a, b, **kw: triton_matmul(_ensure_np32(a), _ensure_np32(b))
                if op == "relu":
                    return lambda a, **kw: triton_relu(_ensure_np32(a))
            if backend == "opencl" and is_opencl_available():
                if op == "matmul":
                    return lambda a, b, **kw: opencl_matmul(a, b)
                if op == "conv2d":
                    return lambda a, b, **kw: opencl_conv2d(
                        a, b, stride=kwargs.get("stride", 1), padding=kwargs.get("padding", 0)
                    )
                if op == "relu":
                    return lambda a, **kw: opencl_relu(a)
            if backend in ("numpy", "baseline"):
                # Defer to baseline by returning None (optimizer will call fn)
                return None
        except Exception:
            return None
        return None

    def explain(
        self, op: str, args, kwargs, *, warmup: int = 0, iterations: int = 1, collect_stats: bool = False
    ) -> dict:
        """Explain selection by probing candidate backends in order.

        Returns a dict with fields:
          - order: list[str] preference order considered
          - candidates: [{name, ok, latency_ms, error}]
          - selected: {name, latency_ms} | None
          - reason: textual reason for selection
          - env: environment-affecting variables
        """
        info: dict = {
            "order": self._candidate_backends(op),
            "candidates": [],
            "selected": None,
            "reason": None,
            "env": {
                "UHOP_BACKEND_PREFERENCE": _cfg.get("UHOP_BACKEND_PREFERENCE"),
                "UHOP_FORCE_BASELINE": bool(_cfg.get("UHOP_FORCE_BASELINE")),
                "UHOP_STRICT_VALIDATE": bool(_cfg.get("UHOP_STRICT_VALIDATE")),
            },
            "params": {
                "warmup": warmup,
                "iterations": iterations,
                "collect_stats": collect_stats,
            },
        }
        if bool(_cfg.get("UHOP_FORCE_BASELINE")):
            info["reason"] = "forced_baseline"
            return info
        for name in info["order"]:
            runner = self._get_runner(name, op, args, kwargs)
            if runner is None:
                info["candidates"].append(
                    {
                        "name": name,
                        "ok": False,
                        "latency_ms": None,
                        "error": "unavailable",
                    }
                )
                continue
            try:
                # Warmup runs (not counted)
                for _w in range(max(0, warmup)):
                    _ = runner(*args, **kwargs)
                times: List[float] = []
                for _i in range(max(1, iterations)):
                    t0 = time.perf_counter()
                    _ = runner(*args, **kwargs)
                    dt_ms = (time.perf_counter() - t0) * 1000.0
                    times.append(dt_ms)
                # Stats
                median_ms = float(statistics.median(times)) if times else None
                rec = {"name": name, "ok": True, "latency_ms": median_ms}
                if collect_stats:
                    rec["stats"] = {
                        "runs": len(times),
                        "median_ms": median_ms,
                        "mean_ms": float(statistics.mean(times)) if len(times) else None,
                        "min_ms": float(min(times)) if times else None,
                        "max_ms": float(max(times)) if times else None,
                        "std_ms": float(statistics.stdev(times)) if len(times) > 1 else 0.0,
                        "times_ms": [float(x) for x in times],
                    }
                info["candidates"].append(rec)
                # Select the first successful candidate (mirrors order-probe policy)
                if info["selected"] is None:
                    info["selected"] = {"name": name, "latency_ms": median_ms}
                    info["reason"] = "order_probe"
            except Exception as e:
                info["candidates"].append(
                    {
                        "name": name,
                        "ok": False,
                        "latency_ms": None,
                        "error": str(e)[:240],
                    }
                )
                continue
        return info


# Global backend manager placeholder (future expansion)
_backend_manager_singleton = None


class DummyBackendManager:
    pass


def get_backend_manager():
    global _backend_manager_singleton
    if _backend_manager_singleton is None:
        _backend_manager_singleton = DummyBackendManager()
    return _backend_manager_singleton
