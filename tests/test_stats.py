import numpy as np
import pandas as pd
import pytest

from product_search.eval.stats import bootstrap_ci, paired_permutation_test, summarise_metrics


def test_bootstrap_constant_values_give_zero_width() -> None:
    assert bootstrap_ci(pd.Series([0.5] * 20), n_resamples=200) == (0.5, 0.5, 0.5)


def test_bootstrap_interval_contains_mean_and_drops_nan() -> None:
    values = pd.Series(np.random.default_rng(1).random(240).tolist() + [np.nan])
    mean, low, high = bootstrap_ci(values, n_resamples=2000)
    assert mean == pytest.approx(values.dropna().mean())
    assert low < mean < high


def test_bootstrap_is_reproducible() -> None:
    values = pd.Series(np.random.default_rng(2).random(50))
    assert bootstrap_ci(values, 500, seed=3) == bootstrap_ci(values, 500, seed=3)


def test_permutation_identical_systems() -> None:
    a = pd.Series([0.1, 0.5, 0.9, 0.3])
    diff, p = paired_permutation_test(a, a.copy(), n_permutations=500)
    assert diff == 0.0
    assert p == 1.0


def test_permutation_detects_consistent_gain() -> None:
    rng = np.random.default_rng(4)
    a = pd.Series(rng.random(240))
    b = a + 0.05 + rng.normal(0, 0.01, 240)
    diff, p = paired_permutation_test(a, b, n_permutations=2000)
    assert diff == pytest.approx(0.05, abs=0.01)
    assert p < 0.01


def test_permutation_aligns_by_query_id() -> None:
    a = pd.Series([0.2, 0.4], index=["q1", "q2"])
    b = pd.Series([0.4, 0.2], index=["q2", "q1"])  # same values, other order
    diff, _ = paired_permutation_test(a, b, n_permutations=100)
    assert diff == 0.0


def test_summarise_metrics_counts_defined_queries() -> None:
    table = pd.DataFrame({"R@10": [1.0, 0.0, np.nan]})
    summary = summarise_metrics(table, n_resamples=100)
    assert summary["R@10"]["n"] == 2
    assert summary["R@10"]["mean"] == 0.5
