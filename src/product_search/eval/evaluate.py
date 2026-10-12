"""Score a saved run on dev or test and record the result."""

from pathlib import Path

from product_search.data.splits import load_split
from product_search.data.wands import label_grades, load_wands
from product_search.eval.metrics import per_query_metrics
from product_search.eval.stats import N_RESAMPLES, summarise_metrics
from product_search.runs import RESULTS_DIR, load_results, update_results_table, write_json

HEADLINE = ("Rcap@100_exact", "R@100_exact", "R@1000_exact", "nDCG@10", "MRR@10")


def evaluate_run(
    run_dir: Path,
    split: str = "dev",
    results_dir: Path = RESULTS_DIR,
    n_resamples: int = N_RESAMPLES,
) -> dict[str, dict[str, float]]:
    """Score run_dir/results.parquet on one side of the split.

    Writes metrics_<split>.json and metrics_per_query_<split>.parquet into the
    run folder and the metric means into results_dir/<split>.csv.

    Args:
        run_dir: Run folder holding results.parquet.
        split: "dev" or "test".
        results_dir: Folder of the committed results tables.
        n_resamples: Bootstrap resamples.

    Returns:
        metric -> {"mean", "low", "high", "n"}.
    """
    fixed_split = load_split()
    query_ids = fixed_split.dev if split == "dev" else fixed_split.test
    labels = label_grades(load_wands().labels)
    results = load_results(run_dir / "results.parquet")

    per_query = per_query_metrics(results, labels, query_ids)
    summary = summarise_metrics(per_query, n_resamples=n_resamples)

    per_query.to_parquet(run_dir / f"metrics_per_query_{split}.parquet")
    write_json(summary, run_dir / f"metrics_{split}.json")
    update_results_table(results_dir / f"{split}.csv", run_dir.name, summary)
    return summary


def format_summary(
    summary: dict[str, dict[str, float]], metrics: tuple[str, ...] = HEADLINE
) -> str:
    """One line per metric: mean [low, high] n."""
    lines = []
    for metric in metrics:
        s = summary[metric]
        lines.append(
            f"  {metric:<15} {s['mean']:.3f} [{s['low']:.3f}, {s['high']:.3f}]  n={s['n']}"
        )
    return "\n".join(lines)
