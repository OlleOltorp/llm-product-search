"""Combine saved dev runs and score every combination.

Step "rrf": reciprocal rank fusion for every combination of 2 or more systems.
Step "weighted": weighted sum over a weight grid for the best combinations
from the RRF step, each compared with RRF on the same systems.

Latency of a combination: BM25 runs on the CPU alongside the GPU, and the
dense models share one GPU, so it is max(BM25 p99, sum of dense p99).
This adds p99s, which slightly overstates the true p99 of the sum.

Writes to runs/fusion/:
    rrf.csv, rrf_per_query.parquet           one row per combination
    weighted.csv, weighted_per_query.parquet one row per (combination, weights)

Usage:
    uv run python scripts/fuse.py rrf
    uv run python scripts/fuse.py weighted --top 3
"""

import argparse
import itertools
import logging

import pandas as pd

from product_search.data.splits import load_split
from product_search.data.wands import label_grades, load_wands
from product_search.eval.metrics import per_query_metrics
from product_search.eval.stats import bootstrap_ci, paired_permutation_test
from product_search.retrieval.fusion import rrf, weighted_sum
from product_search.runs import RUNS_DIR, load_results, read_json
from product_search.types import Results

SYSTEMS = {  # short name: run
    "bm25": "bm25_default",
    "inf15": "dense_inf_15b",
    "arcticm": "dense_arctic_m",
    "arcticl": "dense_arctic_l",
    "mxbai": "dense_mxbai_large",
    "bgel": "dense_bge_large",
    "pplx4": "dense_pplx_4b",
}
BASELINES = {"bm25": "bm25_default", "inf15": "dense_inf_15b"}
METRICS = ("Rcap@100_exact", "nDCG@10")
OBJECTIVE = "Rcap@100_exact"
BUDGET_MS = 50
WEIGHT_STEPS = 10  # weights in steps of 0.1
OUT_DIR = RUNS_DIR / "fusion"


def p99(run: str) -> float:
    """Saved p99 latency of a run, in ms."""
    return float(read_json(RUNS_DIR / run / "meta.json")["cost"]["latency"]["p99_ms"])


def latency(names: tuple[str, ...]) -> float:
    """Estimated p99 of a combination: BM25 in parallel, dense models one after another."""
    dense = sum(p99(SYSTEMS[n]) for n in names if n != "bm25")
    return max(p99(SYSTEMS["bm25"]) if "bm25" in names else 0.0, dense)


def score(
    results: Results, labels: dict[str, dict[str, int]], dev: list[str], baselines: dict
) -> tuple[dict, pd.DataFrame]:
    """Means, intervals and tests against each baseline for one fused result."""
    table = per_query_metrics(results, labels, dev)
    row: dict = {}
    for metric in METRICS:
        mean, low, high = bootstrap_ci(table[metric])
        row |= {metric: mean, f"{metric}_low": low, f"{metric}_high": high}
        for short, base in baselines.items():
            diff, p = paired_permutation_test(base[metric], table[metric])
            row |= {f"Δ {metric} vs {short}": diff * 100, f"p {metric} vs {short}": p}
    return row, table


def weight_grid(n: int) -> list[tuple[float, ...]]:
    """All weights in steps of 0.1 that sum to 1, each system weighted above 0."""
    return [
        tuple(c / WEIGHT_STEPS for c in combo)
        for combo in itertools.product(range(1, WEIGHT_STEPS), repeat=n)
        if sum(combo) == WEIGHT_STEPS
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Score fused combinations on dev.")
    parser.add_argument("step", choices=["rrf", "weighted"])
    parser.add_argument("--top", type=int, default=3, help="Combinations for the weighted step.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_wands()
    dev = load_split().dev
    labels = label_grades(data.labels)
    runs = {n: load_results(RUNS_DIR / run / "results.parquet") for n, run in SYSTEMS.items()}
    base_tables = {
        short: pd.read_parquet(RUNS_DIR / run / "metrics_per_query_dev.parquet")
        for short, run in BASELINES.items()
    }

    rows, tables = [], []
    if args.step == "rrf":
        combos = [
            names
            for size in range(2, len(SYSTEMS) + 1)
            for names in itertools.combinations(SYSTEMS, size)
        ]
        for names in combos:
            row, table = score(rrf([runs[n] for n in names]), labels, dev, base_tables)
            label = "+".join(names)
            rows.append(
                {"combination": label, "systems": len(names), "p99_ms": latency(names)} | row
            )
            tables.append(table.assign(combination=label))
        logging.info("%d RRF combinations scored", len(combos))
    else:
        rrf_rows = pd.read_csv(OUT_DIR / "rrf.csv")
        within = rrf_rows[rrf_rows["p99_ms"] <= BUDGET_MS].sort_values(OBJECTIVE, ascending=False)
        chosen = list(within["combination"].head(args.top))
        bm25_pairs = within[
            (within["systems"] == 2) & within["combination"].str.startswith("bm25+")
        ]
        if bm25_pairs["combination"].iloc[0] not in chosen:
            chosen.append(bm25_pairs["combination"].iloc[0])
        rrf_tables = pd.read_parquet(OUT_DIR / "rrf_per_query.parquet").set_index("query_id")
        for label in chosen:
            names = tuple(label.split("+"))
            reference = {
                "rrf": rrf_tables[rrf_tables["combination"] == label].drop(columns="combination")
            }
            for weights in weight_grid(len(names)):
                fused = weighted_sum([runs[n] for n in names], weights)
                row, table = score(fused, labels, dev, base_tables | reference)
                text = " ".join(f"{w:.1f}" for w in weights)
                rows.append({"combination": label, "weights": text} | row)
                tables.append(table.assign(combination=label, weights=text))
            logging.info("%s: %d weight settings", label, len(weight_grid(len(names))))

    result = pd.DataFrame(rows).sort_values(OBJECTIVE, ascending=False)
    result.to_csv(OUT_DIR / f"{args.step}.csv", index=False)
    pd.concat(tables).reset_index().to_parquet(OUT_DIR / f"{args.step}_per_query.parquet")
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(result.head(15).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
