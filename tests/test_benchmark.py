from uhop.core.benchmark import benchmark_callable


def test_benchmark_respects_warmup_and_sync():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1

    # With warmup=3 and runs=5, total calls should be 8
    _ = benchmark_callable(fn, runs=5, warmup=3)
    assert calls["n"] == 8
