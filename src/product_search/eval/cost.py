"""Cost measurement: query latency, indexing time, index size and memory.

Protocol (tower only): close background apps, fix thread counts, warm up,
then time every query several times so p99 rests on more than a handful of
measurements.
"""

import os
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import psutil

WARMUP_QUERIES = 100
REPEATS = 5


def time_queries(
    search: Callable[[str], object],
    queries: Sequence[str],
    warmup: int = WARMUP_QUERIES,
    repeats: int = REPEATS,
    sync: Callable[[], None] | None = None,
) -> list[float]:
    """Time a search function one query at a time.

    Args:
        search: Runs one query end to end (text in, ranked ids out).
        queries: Query texts.
        warmup: Untimed queries run first, to fill caches and finish
            one-time compilation.
        repeats: Times every query is timed.
        sync: Called before each clock stop, e.g. torch.cuda.synchronize,
            so GPU work is included.

    Returns:
        Latencies in milliseconds, len(queries) * repeats values.
    """
    for i in range(warmup):
        search(queries[i % len(queries)])

    latencies = []
    for _ in range(repeats):
        for query in queries:
            start = time.perf_counter_ns()
            search(query)
            if sync is not None:
                sync()
            latencies.append((time.perf_counter_ns() - start) / 1e6)
    return latencies


def latency_summary(latencies_ms: Sequence[float]) -> dict[str, float]:
    """Mean and percentiles (p50, p75, p95, p99) of latencies in milliseconds."""
    values = np.asarray(latencies_ms, dtype=float)
    p50, p75, p95, p99 = np.percentile(values, [50, 75, 95, 99])
    return {
        "mean_ms": float(values.mean()),
        "p50_ms": float(p50),
        "p75_ms": float(p75),
        "p95_ms": float(p95),
        "p99_ms": float(p99),
        "n": float(values.size),
    }


@contextmanager
def timed() -> Iterator[dict[str, float]]:
    """Measure wall-clock seconds of a block, e.g. building an index.

    Example:
        with timed() as t:
            build_index()
        t["seconds"]
    """
    result: dict[str, float] = {}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["seconds"] = time.perf_counter() - start


def dir_size_mb(path: Path) -> float:
    """Total size of the files under path, in MB."""
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


def peak_ram_gb() -> float:
    """Peak resident memory of this process so far, in GB."""
    if sys.platform == "win32":
        return float(psutil.Process().memory_info().peak_wset) / 1e9
    import resource

    # ru_maxrss is in kilobytes on Linux.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def thread_settings() -> dict[str, str]:
    """Thread-count environment variables that affect timing."""
    names = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")
    return {name: os.environ.get(name, "unset") for name in names}
