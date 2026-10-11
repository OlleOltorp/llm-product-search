"""Retrieval metrics, computed per query from ranked results and graded labels.

Grades: Exact 2, Partial 1, Irrelevant 0. Products without a label count as
not relevant. A query is left out of a metric (NaN) when it has no relevant
product at that level, since recall and MRR are undefined there.

Metric names:
    R@k_exact, R@k_any      Recall at k; "any" means Exact or Partial.
    Rcap@k_exact, Rcap@k_any  Recall with denominator min(k, #relevant).
    nDCG@10                 Graded, linear gain, as in trec_eval.
    MRR@10                  Reciprocal rank of the first Exact product.
"""

import math
from collections.abc import Iterable, Sequence

import pandas as pd

from product_search.types import Labels, Results

RECALL_KS = (10, 20, 50, 100, 200, 500, 1000)
EXACT = 2
LEVELS = {"exact": 2, "any": 1}  # name -> lowest grade counted as relevant


def rank(results: Results, depth: int | None = None) -> dict[str, list[str]]:
    """Order each query's products by score, highest first.

    Ties are broken by product id so the order is always the same.

    Args:
        results: query_id -> {product_id: score}.
        depth: Keep only the top depth products; None keeps all.

    Returns:
        query_id -> product ids, best first.
    """
    ranked = {}
    for query_id, scores in results.items():
        order = sorted(scores, key=lambda product_id: (-scores[product_id], product_id))
        ranked[query_id] = order if depth is None else order[:depth]
    return ranked


def recall_at_k(ranking: Sequence[str], relevant: set[str], k: int, capped: bool = False) -> float:
    """Share of relevant products found in the top k.

    Args:
        ranking: Product ids, best first.
        relevant: Ids of relevant products.
        k: Cut-off rank.
        capped: Divide by min(k, #relevant) instead of #relevant, so a query
            with more relevant products than k can still reach 1.0.

    Returns:
        Recall, or NaN if there are no relevant products.
    """
    if not relevant:
        return math.nan
    found = len(relevant.intersection(ranking[:k]))
    return found / (min(k, len(relevant)) if capped else len(relevant))


def ndcg_at_k(ranking: Sequence[str], grades: dict[str, int], k: int) -> float:
    """Normalised discounted cumulative gain at k.

    Gain is the grade itself (Exact 2, Partial 1) with a log2 rank discount,
    divided by the best possible score for this query.

    Args:
        ranking: Product ids, best first.
        grades: product_id -> grade for this query's labelled products.
        k: Cut-off rank.

    Returns:
        nDCG in [0, 1], or NaN if no product has a grade above 0.
    """
    ideal = sorted((g for g in grades.values() if g > 0), reverse=True)[:k]
    if not ideal:
        return math.nan
    dcg = sum(grades.get(p, 0) / math.log2(i + 2) for i, p in enumerate(ranking[:k]))
    ideal_dcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / ideal_dcg


def reciprocal_rank_at_k(ranking: Sequence[str], relevant: set[str], k: int) -> float:
    """1 / rank of the first relevant product in the top k, 0 if none.

    Returns:
        Reciprocal rank, or NaN if there are no relevant products.
    """
    if not relevant:
        return math.nan
    for i, product_id in enumerate(ranking[:k]):
        if product_id in relevant:
            return 1 / (i + 1)
    return 0.0


def per_query_metrics(
    results: Results,
    labels: Labels,
    query_ids: Iterable[str],
    ks: Sequence[int] = RECALL_KS,
) -> pd.DataFrame:
    """Compute every metric for every query.

    Args:
        results: query_id -> {product_id: score}. A query missing here
            retrieved nothing and scores 0.
        labels: query_id -> {product_id: grade}.
        query_ids: Queries to score, e.g. the dev split.
        ks: Cut-offs for the recall metrics.

    Returns:
        One row per query (index query_id), one column per metric, NaN where
        a metric is undefined for that query.
    """
    ranked = rank(results, depth=max(ks))
    rows = {}
    for query_id in query_ids:
        ranking = ranked.get(query_id, [])
        grades = labels.get(query_id, {})
        row = {}
        for level, min_grade in LEVELS.items():
            relevant = {p for p, g in grades.items() if g >= min_grade}
            for k in ks:
                row[f"R@{k}_{level}"] = recall_at_k(ranking, relevant, k)
            for k in ks:
                row[f"Rcap@{k}_{level}"] = recall_at_k(ranking, relevant, k, capped=True)
        exact = {p for p, g in grades.items() if g >= EXACT}
        row["nDCG@10"] = ndcg_at_k(ranking, grades, 10)
        row["MRR@10"] = reciprocal_rank_at_k(ranking, exact, 10)
        rows[query_id] = row
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("query_id")
