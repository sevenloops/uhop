# uhop/profiler.py
import time
from typing import Callable, Tuple

import numpy as np


def time_function(
    func: Callable, args: Tuple = (), kwargs: dict = None, repeats: int = 3
) -> float:
    kwargs = kwargs or {}
    # warm-up
    for _ in range(1):
        func(*args, **kwargs)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        func(*args, **kwargs)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return float(np.median(times))


def baseline_matmul(a, b):
    # simple numpy baseline
    return a @ b
