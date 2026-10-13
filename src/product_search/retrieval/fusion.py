"""Combine finished ranked lists from several retrievers (late fusion).

Each retriever's saved top-k list is merged into one ranking. A product
missing from a list gets nothing from that list: no RRF points, and the
lowest normalised score (0) in a weighted sum.
"""

from collections.abc import Sequence

from product_search.eval.metrics import rank
from product_search.types import Results

RRF_K = 60


def _top(scores: dict[str, float], depth: int) -> dict[str, float]:
    """The depth highest scores, ties broken by product id."""
    order = sorted(scores, key=lambda product_id: (-scores[product_id], product_id))
    return {product_id: scores[product_id] for product_id in order[:depth]}


def rrf(runs: Sequence[Results], k: int = RRF_K, depth: int = 1000) -> Results:
    """Reciprocal rank fusion: each list adds 1 / (k + rank) for every product it ranks.

    Args:
        runs: One result set per retriever.
        k: Damping constant; 60 is the usual default.
        depth: Products kept per query after fusion.
    """
    rankings = [rank(run) for run in runs]
    fused: Results = {}
    for query_id in set().union(*runs):
        scores: dict[str, float] = {}
        for ranking in rankings:
            for position, product_id in enumerate(ranking.get(query_id, []), start=1):
                scores[product_id] = scores.get(product_id, 0.0) + 1 / (k + position)
        fused[query_id] = _top(scores, depth)
    return fused


def min_max(run: Results) -> Results:
    """Scale each query's scores to 0-1 (all 1 when every score is equal)."""
    scaled: Results = {}
    for query_id, scores in run.items():
        if not scores:
            scaled[query_id] = {}
            continue
        low, high = min(scores.values()), max(scores.values())
        span = high - low
        scaled[query_id] = {
            product_id: (score - low) / span if span > 0 else 1.0
            for product_id, score in scores.items()
        }
    return scaled


def weighted_sum(runs: Sequence[Results], weights: Sequence[float], depth: int = 1000) -> Results:
    """Sum of min-max scaled scores, one weight per retriever.

    Raises:
        ValueError: If there is not exactly one weight per run.
    """
    if len(weights) != len(runs):
        raise ValueError(f"{len(runs)} runs but {len(weights)} weights")
    scaled = [min_max(run) for run in runs]
    fused: Results = {}
    for query_id in set().union(*runs):
        scores: dict[str, float] = {}
        for run, weight in zip(scaled, weights, strict=True):
            for product_id, score in run.get(query_id, {}).items():
                scores[product_id] = scores.get(product_id, 0.0) + weight * score
        fused[query_id] = _top(scores, depth)
    return fused
