# uhop/core/benchmark.py
"""
Benchmark helpers for UHOP. Ensures warm-up and correct device synchronization.
"""
from time import perf_counter
from statistics import median
from typing import Callable, Iterable, Optional, Any, List


def benchmark_callable(
    fn: Callable[..., Any],
    args: Iterable = (),
    runs: int = 5,
    warmup: int = 3,
    sync_fn: Optional[Callable[[], None]] = None,
    start_event: Optional[Callable[[], None]] = None,
    end_event: Optional[Callable[[], float]] = None,
) -> float:
    """
    Benchmark callable `fn(*args)`:
      - runs warmup times (not timed)
      - runs `runs` times and returns median time in seconds
      - if sync_fn is provided, it's called before starting and after each run to ensure device completion
    """
    timings: List[float] = []
    # warmup
    for _ in range(warmup):
        res = fn(*args)
        if sync_fn:
            try:
                sync_fn()
            except Exception:
                pass

    # timed runs
    for _ in range(runs):
        if sync_fn:
            try:
                sync_fn()
            except Exception:
                pass
        if start_event:
            try:
                start_event()
            except Exception:
                start_event = None
        t0 = perf_counter()
        res = fn(*args)
        if end_event:
            try:
                # If end_event returns a device timestamp (seconds), prefer it
                ev_t = end_event()
                if isinstance(ev_t, (int, float)):
                    timings.append(float(ev_t))
                    continue
            except Exception:
                end_event = None
        if sync_fn:
            try:
                sync_fn()
            except Exception:
                pass
        t1 = perf_counter()
        timings.append(t1 - t0)
    return median(timings)


def measure_time(func: Callable[[], Any], sync_fn: Optional[Callable[[], None]] = None, warmup=3, repeat=10) -> float:
    """Deprecated convenience wrapper; prefer benchmark_callable.
    Keeps API but adds optional sync_fn.
    """
    for _ in range(warmup):
        func()
        if sync_fn:
            try: sync_fn()
            except Exception: pass
    times = []
    for _ in range(repeat):
        if sync_fn:
            try: sync_fn()
            except Exception: pass
        t0 = perf_counter()
        func()
        if sync_fn:
            try: sync_fn()
            except Exception: pass
        times.append(perf_counter() - t0)
    return median(times)
