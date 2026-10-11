"""Score a run on dev (or, once, on test) and record the result.

Writes metrics_<split>.json and metrics_per_query_<split>.parquet into the
run folder, and the metric means into results/<split>.csv.

Usage:
    uv run python scripts/evaluate.py runs/bm25_default
    uv run python scripts/evaluate.py runs/final --split test --confirm-test
"""

import argparse
from pathlib import Path

from product_search.data.splits import load_split
from product_search.data.wands import label_grades, load_wands
from product_search.eval.metrics import per_query_metrics
from product_search.eval.stats import N_RESAMPLES, summarise_metrics
from product_search.runs import RESULTS_DIR, load_results, update_results_table, write_json

HEADLINE = ("R@100_exact", "Rcap@100_exact", "R@100_any", "nDCG@10", "MRR@10")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a run and record the result.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--confirm-test", action="store_true")
    parser.add_argument("--resamples", type=int, default=N_RESAMPLES)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    if args.split == "test" and not args.confirm_test:
        parser.error("The test set is used once, at the end. Add --confirm-test to use it.")

    split = load_split()
    query_ids = split.dev if args.split == "dev" else split.test
    labels = label_grades(load_wands().labels)
    results = load_results(args.run_dir / "results.parquet")

    per_query = per_query_metrics(results, labels, query_ids)
    summary = summarise_metrics(per_query, n_resamples=args.resamples)

    per_query.to_parquet(args.run_dir / f"metrics_per_query_{args.split}.parquet")
    write_json(summary, args.run_dir / f"metrics_{args.split}.json")
    update_results_table(args.results_dir / f"{args.split}.csv", args.run_dir.name, summary)

    print(f"{args.run_dir.name} on {args.split} ({len(query_ids)} queries)")
    for metric in HEADLINE:
        s = summary[metric]
        print(f"  {metric:<15} {s['mean']:.3f} [{s['low']:.3f}, {s['high']:.3f}]  n={s['n']}")


if __name__ == "__main__":
    main()
