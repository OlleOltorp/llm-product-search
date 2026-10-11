import pytest

from product_search.eval.cost import latency_summary, time_queries, timed


def test_time_queries_warms_up_then_times_every_query_repeatedly() -> None:
    calls: list[str] = []
    latencies = time_queries(calls.append, ["a", "b", "c"], warmup=4, repeats=2)
    assert len(latencies) == 6
    assert len(calls) == 4 + 6
    assert all(ms >= 0 for ms in latencies)


def test_latency_summary_percentiles() -> None:
    summary = latency_summary([float(i) for i in range(1, 101)])
    assert summary["p50_ms"] == pytest.approx(50.5)
    assert summary["p99_ms"] == pytest.approx(99.01)
    assert summary["n"] == 100


def test_timed_records_seconds() -> None:
    with timed() as t:
        sum(range(1000))
    assert t["seconds"] >= 0
