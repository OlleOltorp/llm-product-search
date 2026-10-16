"""Uncertainty for metrics: bootstrap intervals and paired permutation tests.

Both work over queries: each query is one observation.
"""

import numpy as np
import pandas as pd

N_RESAMPLES = 10_000


def bootstrap_ci(
    values: pd.Series,
    n_resamples: int = N_RESAMPLES,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Mean with a percentile bootstrap interval.

    Draws the queries with replacement n_resamples times and takes the
    spread of the resampled means. NaN values (undefined for a query) are
    dropped first.

    Args:
        values: One value per query.
        n_resamples: Number of resamples.
        confidence: Interval coverage, e.g. 0.95.
        seed: Random seed.

    Returns:
        Mean, lower bound, upper bound.

    Raises:
        ValueError: If no value is left after dropping NaN.
    """
    x = values.dropna().to_numpy(dtype=float)
    if x.size == 0:
        raise ValueError("No values to bootstrap")
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_resamples, x.size), replace=True).mean(axis=1)
    tail = (1 - confidence) / 2
    low, high = np.quantile(means, [tail, 1 - tail])
    return float(x.mean()), float(low), float(high)


def paired_permutation_test(
    a: pd.Series,
    b: pd.Series,
    n_permutations: int = N_RESAMPLES,
    seed: int = 0,
) -> tuple[float, float]:
    """Two-sided paired test of whether b differs from a.

    Under "no real difference", which system scored higher on a query is
    chance, so each per-query difference is equally likely to have either
    sign. Random sign flips give the spread of mean differences under that
    assumption; the p-value is the share at least as extreme as observed.

    Args:
        a: Per-query values for system A, indexed by query id.
        b: Per-query values for system B, indexed by query id.
        n_permutations: Number of random sign flips.
        seed: Random seed.

    Returns:
        Mean difference (b - a) and p-value.

    Raises:
        ValueError: If no query has a value for both systems.
    """
    a, b = a.align(b, join="inner")
    diff = (b - a).dropna().to_numpy(dtype=float)
    if diff.size == 0:
        raise ValueError("No queries with values for both systems")
    observed = diff.mean()
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_permutations, diff.size))
    permuted = (signs * diff).mean(axis=1)
    extreme = np.sum(np.abs(permuted) >= abs(observed) - 1e-12)
    return float(observed), float((extreme + 1) / (n_permutations + 1))


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    """Benjamini-Hochberg adjusted p-values (q-values).

    With many tests, some p < 0.05 by chance. Keeping the tests with
    q < 0.05 keeps the expected share of false findings among them below 5%.

    Args:
        p_values: One p-value per test.

    Returns:
        q-values in the same order and index.
    """
    p = p_values.to_numpy(dtype=float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    # Each q is the smallest ranked value at or above its position, capped at 1.
    q = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1.0)
    out = np.empty_like(q)
    out[order] = q
    return pd.Series(out, index=p_values.index)


def summarise_metrics(
    per_query: pd.DataFrame, n_resamples: int = N_RESAMPLES, seed: int = 0
) -> dict[str, dict[str, float]]:
    """Mean, bootstrap interval and query count for every metric column.

    Args:
        per_query: One row per query, one column per metric.
        n_resamples: Bootstrap resamples.
        seed: Random seed.

    Returns:
        metric -> {"mean", "low", "high", "n"}.
    """
    summary = {}
    for metric in per_query.columns:
        mean, low, high = bootstrap_ci(per_query[metric], n_resamples, seed=seed)
        n = int(per_query[metric].notna().sum())
        summary[metric] = {"mean": mean, "low": low, "high": high, "n": n}
    return summary
