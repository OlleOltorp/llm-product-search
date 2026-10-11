import math
import random

import pandas as pd
import pytest

from product_search.eval.metrics import (
    ndcg_at_k,
    per_query_metrics,
    rank,
    recall_at_k,
    reciprocal_rank_at_k,
)


def test_rank_orders_by_score_and_breaks_ties_by_id() -> None:
    results = {"q": {"b": 1.0, "a": 1.0, "c": 2.0}}
    assert rank(results) == {"q": ["c", "a", "b"]}
    assert rank(results, depth=2) == {"q": ["c", "a"]}


def test_recall_plain_and_capped() -> None:
    ranking = ["a", "b", "c", "d"]
    relevant = {"a", "c", "x"}
    assert recall_at_k(ranking, relevant, 2) == pytest.approx(1 / 3)
    assert recall_at_k(ranking, relevant, 4) == pytest.approx(2 / 3)
    # Capped: 1 found in the top 2, denominator min(2, 3) = 2.
    assert recall_at_k(ranking, relevant, 2, capped=True) == pytest.approx(1 / 2)
    assert math.isnan(recall_at_k(ranking, set(), 2))


def test_ndcg_by_hand() -> None:
    grades = {"a": 2, "b": 1, "c": 0}
    # DCG = 1/log2(2) + 2/log2(3); ideal = 2/log2(2) + 1/log2(3).
    expected = (1 + 2 / math.log2(3)) / (2 + 1 / math.log2(3))
    assert ndcg_at_k(["b", "a", "c"], grades, 10) == pytest.approx(expected)
    assert ndcg_at_k(["a", "b"], grades, 10) == pytest.approx(1.0)
    assert math.isnan(ndcg_at_k(["a"], {"a": 0}, 10))


def test_reciprocal_rank() -> None:
    assert reciprocal_rank_at_k(["x", "y", "a"], {"a"}, 10) == pytest.approx(1 / 3)
    assert reciprocal_rank_at_k(["x", "y", "a"], {"a"}, 2) == 0.0
    assert math.isnan(reciprocal_rank_at_k(["a"], set(), 10))


def test_per_query_metrics_handles_missing_and_undefined_queries() -> None:
    labels = {"q1": {"a": 2, "b": 1}, "q2": {"c": 1}, "q3": {"d": 2}}
    results = {"q1": {"a": 2.0, "b": 1.0}, "q2": {"c": 1.0}}
    table = per_query_metrics(results, labels, ["q1", "q2", "q3"], ks=(10,))

    assert table.loc["q1", "R@10_exact"] == 1.0
    assert pd.isna(table.loc["q2", "R@10_exact"])  # no Exact product
    assert table.loc["q2", "R@10_any"] == 1.0
    assert table.loc["q3", "R@10_exact"] == 0.0  # nothing retrieved
    assert list(table.columns) == [
        "R@10_exact",
        "Rcap@10_exact",
        "R@10_any",
        "Rcap@10_any",
        "nDCG@10",
        "MRR@10",
    ]


def test_matches_ir_measures_on_random_data() -> None:
    ir_measures = pytest.importorskip("ir_measures")
    rng = random.Random(0)
    products = [f"p{i}" for i in range(300)]
    labels = {
        f"q{q}": {p: rng.choice([0, 0, 1, 2]) for p in rng.sample(products, 80)} for q in range(30)
    }
    results = {f"q{q}": {p: rng.random() for p in rng.sample(products, 150)} for q in range(30)}

    ours = per_query_metrics(results, labels, list(labels), ks=(10, 100))
    measures = [
        ir_measures.parse_measure("R(rel=2)@100"),
        ir_measures.parse_measure("R(rel=1)@10"),
        ir_measures.parse_measure("nDCG@10"),
        ir_measures.parse_measure("RR(rel=2)@10"),
    ]
    columns = ["R@100_exact", "R@10_any", "nDCG@10", "MRR@10"]
    reference = {
        (m.query_id, str(m.measure)): m.value
        for m in ir_measures.iter_calc(measures, labels, results)
    }

    for measure, column in zip(measures, columns, strict=True):
        for query_id in labels:
            if not pd.isna(ours.loc[query_id, column]):
                assert ours.loc[query_id, column] == pytest.approx(
                    reference[(query_id, str(measure))]
                ), (query_id, column)
