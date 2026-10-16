"""Shared pieces of the expansion scripts: retrievers, sources and combinations.

A grid config names expansion runs by short keys (p1_t5, p2, q1, ...).
Product combinations: one P1 variant or none, x P2 on/off x P3 on/off.
Query combinations: every subset of the query versions.

Separate-mode lists:  runs/lists/<expansion run>__<retriever>/
Append-mode lists:    runs/lists/append/<grid>/<products>__<queries>__<retriever>/
Append indexes:       runs/append_index/<grid>/<products>__<retriever>/ (dense only)
"""

from pathlib import Path
from typing import Any

import pandas as pd

from product_search.eval.metrics import rank
from product_search.expansion.generate import load_expansion_config
from product_search.expansion.grid import subsets
from product_search.retrieval.bm25 import BM25Retriever, load_bm25_config
from product_search.retrieval.dense import DenseRetriever, load_dense_config
from product_search.retrieval.fusion import rrf
from product_search.retrieval.rerank import FIRST_STAGE
from product_search.runs import REPO_DIR, RUNS_DIR, load_results
from product_search.types import Results

LISTS_DIR = RUNS_DIR / "lists"
APPEND_DIR = LISTS_DIR / "append"
INDEX_DIR = RUNS_DIR / "append_index"
RETRIEVERS = {
    "bm25": REPO_DIR / "configs/bm25_default.yaml",
    "inf15": REPO_DIR / "configs/dense_inf_15b.yaml",
    "arcticm": REPO_DIR / "configs/dense_arctic_m.yaml",
}

Combination = tuple[str, ...]


def load_retriever(name: str) -> tuple[BM25Retriever | DenseRetriever, tuple[str, ...]]:
    """The retriever and the product fields its original index uses."""
    path = RETRIEVERS[name]
    if name == "bm25":
        _, bm25 = load_bm25_config(path)
        return BM25Retriever(bm25), bm25.fields
    _, dense = load_dense_config(path)
    return DenseRetriever(dense), dense.fields


def is_ready(run: str) -> bool:
    """Whether an expansion run has finished (expansions.parquet is written last)."""
    return (RUNS_DIR / run / "expansions.parquet").exists()


def first_stage_titles(
    products: pd.DataFrame, query_ids: list[str], k: int
) -> dict[str, list[str]]:
    """Names of the top k products per query in the fused first stage (Q4 grounding).

    Test queries are covered once every first-stage run has its test run
    (<run>_test); until then they are left out.
    """
    names = dict(zip(products["product_id"], products["name"], strict=True))
    # All or none: a test query fused from only some retrievers would be grounded differently.
    with_test = all((RUNS_DIR / f"{run}_test" / "results.parquet").exists() for run in FIRST_STAGE)
    fused = rrf([_base_results(run, with_test) for run in FIRST_STAGE])
    top = rank({q: fused[q] for q in query_ids if q in fused}, k)
    return {q: [names[p] for p in ranking] for q, ranking in top.items()}


def _base_results(run: str, with_test: bool) -> Results:
    """A base run's dev results, plus its test run's when with_test."""
    results = load_results(RUNS_DIR / run / "results.parquet")
    if with_test:
        results |= load_results(RUNS_DIR / f"{run}_test" / "results.parquet")
    return results


def load_source(run: str) -> tuple[str, dict[str, str]]:
    """An expansion run's target ("products" or "queries") and id -> cleaned text."""
    _, config = load_expansion_config(RUNS_DIR / run / "config.yaml")
    table = pd.read_parquet(RUNS_DIR / run / "expansions.parquet")
    return config.target, dict(zip(table["id"], table["text"], strict=True))


def expansion_keys(config: dict[str, Any]) -> dict[str, str]:
    """Key -> expansion run, for all product and query expansions in a grid config."""
    return config["p1"] | config["products"] | config["queries"]


def product_combinations(config: dict[str, Any]) -> list[Combination]:
    """Every product combination, the empty one first."""
    return [
        tuple(k for k in (p1, *extra) if k)
        for p1 in [None, *config["p1"]]
        for extra in subsets(list(config["products"]))
    ]


def query_combinations(config: dict[str, Any]) -> list[Combination]:
    """Every subset of the query versions, the empty one first."""
    return list(subsets(list(config["queries"])))


def label(keys: Combination) -> str:
    """p1_t5+p2, or none."""
    return "+".join(keys) or "none"


def append_list_dir(grid: str, products: Combination, queries: Combination, retriever: str) -> Path:
    """Folder of one append-mode list. Keys only mean something within a grid
    config, so lists live under the grid's name."""
    return APPEND_DIR / grid / f"{label(products)}__{label(queries)}__{retriever}"
