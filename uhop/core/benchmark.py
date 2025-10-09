# uhop/core/benchmark.py
import time
import numpy as np
from typing import Callable

def benchmark_callable(fn: Callable, args=(), kwargs=None, runs: int = 5) -> float:
    kwargs = kwargs or {}
    # warmup
    try:
        fn(*args, **kwargs)
    except Exception:
        pass
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return float(np.median(times))
