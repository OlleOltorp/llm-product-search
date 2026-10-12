"""Grid search for BM25 on dev, with ablations around the best setting.

Grid: 5 field choices x stemming on/off x 7 k1 x 6 b = 420 settings.
Stopwords on and minimum word length 2 stay fixed (bm25s defaults).

Writes to runs/bm25_grid/:
    grid.parquet       Every metric for every (setting, dev query).
    summary.csv        One row per setting, best first.
    ablations.csv      Best setting with one change at a time.
    meta.json          Git commit and machine.
and configs/bm25_tuned.yaml for the best setting.

Usage:
    uv run python scripts/tune_bm25.py
"""

import itertools
import logging
import time
from dataclasses import replace
from typing import Any

import pandas as pd

from product_search.data.splits import load_split
from product_search.data.wands import label_grades, load_wands, product_texts, query_texts
from product_search.eval.metrics import per_query_metrics
from product_search.eval.stats import bootstrap_ci, paired_permutation_test
from product_search.retrieval.bm25 import (
    BM25Config,
    BM25Retriever,
    save_bm25_config,
    tokenize,
)
from product_search.runs import (
    REPO_DIR,
    RESULT_DEPTH,
    RUNS_DIR,
    git_info,
    machine_info,
    write_json,
)

OBJECTIVE = "Rcap@100_exact"
REPORT = ("nDCG@10", "MRR@10", "R@100_exact", "R@1000_exact")

FIELD_OPTIONS = {
    "name": ("name",),
    "name+cat": ("name", "categories"),
    "name+cat+path": ("name", "categories", "category_path"),
    "name+cat+path+desc": ("name", "categories", "category_path", "description"),
    "all": ("name", "categories", "category_path", "description", "features"),
}
STEMMING = (True, False)
K1_VALUES = (0.3, 0.6, 0.9, 1.2, 1.5, 2.0, 3.0)
B_VALUES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
SETTINGS = ("fields", "stemming", "k1", "b")
DEFAULT_K1_B = {"k1": 0.9, "b": 0.4}

OUT_DIR = RUNS_DIR / "bm25_grid"


def run_grid() -> pd.DataFrame:
    """Score every setting on the dev queries.

    Words are split once per (fields, stemming); each k1/b only rebuilds
    the index.

    Returns:
        One row per (setting, query) with setting and metric columns.
    """
    data = load_wands()
    dev = load_split().dev
    labels = label_grades(data.labels)
    queries = query_texts(data.queries)
    dev_texts = [queries[q] for q in dev]

    tables = []
    for fields_name, stemming in itertools.product(FIELD_OPTIONS, STEMMING):
        start = time.perf_counter()
        base = BM25Config(fields=FIELD_OPTIONS[fields_name], stemming=stemming)
        products = product_texts(data.products, base.fields)
        product_ids = list(products)
        texts = [products[p] for p in product_ids]
        product_tokens = tokenize(texts, stemming, base.stopwords, base.min_word_length)
        query_tokens = tokenize(dev_texts, stemming, base.stopwords, base.min_word_length)

        for k1, b in itertools.product(K1_VALUES, B_VALUES):
            retriever = BM25Retriever(replace(base, k1=k1, b=b))
            retriever.index_tokens(product_ids, product_tokens)
            results = retriever.search_tokens(dev, query_tokens, RESULT_DEPTH)
            table = per_query_metrics(results, labels, dev).reset_index()
            tables.append(table.assign(fields=fields_name, stemming=stemming, k1=k1, b=b))
        logging.info(
            "%-20s stemming=%-5s  %d settings in %.0fs",
            fields_name,
            stemming,
            len(K1_VALUES) * len(B_VALUES),
            time.perf_counter() - start,
        )
    return pd.concat(tables, ignore_index=True)


def summarise(grid: pd.DataFrame) -> pd.DataFrame:
    """One row per setting: objective mean with 95% interval and other means, best first."""
    rows = []
    for key, group in grid.groupby(list(SETTINGS), sort=False):
        mean, low, high = bootstrap_ci(group[OBJECTIVE])
        row = dict(zip(SETTINGS, key, strict=True))
        row |= {OBJECTIVE: mean, "low": low, "high": high}
        row |= {metric: group[metric].mean() for metric in REPORT}
        rows.append(row)
    return pd.DataFrame(rows).sort_values(OBJECTIVE, ascending=False, ignore_index=True)


def scores(grid: pd.DataFrame, setting: dict[str, Any]) -> pd.DataFrame:
    """Per-query metrics of one setting, indexed by query id."""
    mask = pd.Series(True, index=grid.index)
    for column, value in setting.items():
        mask &= grid[column] == value
    return grid[mask].set_index("query_id")


def ablations(grid: pd.DataFrame, best: dict[str, Any]) -> pd.DataFrame:
    """The best setting, then one change at a time, each tested against the best.

    Returns:
        One row per variant: what changed, objective and nDCG@10 means, the
        objective difference from the best and its permutation-test p-value.
    """
    variants: dict[str, dict[str, Any]] = {"best": best}
    variants[f"stemming {'off' if best['stemming'] else 'on'}"] = best | {
        "stemming": not best["stemming"]
    }
    for fields_name in FIELD_OPTIONS:
        if fields_name != best["fields"]:
            variants[f"fields: {fields_name}"] = best | {"fields": fields_name}
    variants["default k1 0.9, b 0.4"] = best | DEFAULT_K1_B

    best_scores = scores(grid, best)
    rows = []
    for change, setting in variants.items():
        s = scores(grid, setting)
        diff, p = paired_permutation_test(best_scores[OBJECTIVE], s[OBJECTIVE])
        rows.append(
            {
                "change": change,
                **setting,
                OBJECTIVE: s[OBJECTIVE].mean(),
                "nDCG@10": s["nDCG@10"].mean(),
                "diff_vs_best": diff,
                "p_value": p,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("bm25s").setLevel(logging.WARNING)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    grid = run_grid()
    grid.to_parquet(OUT_DIR / "grid.parquet", index=False)
    summary = summarise(grid)
    summary.to_csv(OUT_DIR / "summary.csv", index=False)

    top = summary.iloc[0]
    best: dict[str, Any] = {
        "fields": str(top["fields"]),
        "stemming": bool(top["stemming"]),
        "k1": float(top["k1"]),
        "b": float(top["b"]),
    }
    ablation_table = ablations(grid, best)
    ablation_table.to_csv(OUT_DIR / "ablations.csv", index=False)
    write_json({"git": git_info(), "machine": machine_info()}, OUT_DIR / "meta.json")

    tuned = BM25Config(
        fields=FIELD_OPTIONS[best["fields"]],
        stemming=best["stemming"],
        k1=best["k1"],
        b=best["b"],
    )
    save_bm25_config(REPO_DIR / "configs" / "bm25_tuned.yaml", "bm25_tuned", tuned)

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary.head(10).round(3).to_string())
        print()
        print(ablation_table.round(3).to_string())
    if best["k1"] == max(K1_VALUES):
        print(f"\nBest k1 is the largest tried ({best['k1']}); consider extending the range.")


if __name__ == "__main__":
    main()
