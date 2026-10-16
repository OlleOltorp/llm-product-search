"""Score every combination of product and query expansions, in both modes, on three systems.

Systems: BM25 alone, dense alone (inf-1.5B + arctic-m) and the hybrid (all
three). For each system the base is its retrievers' original lists fused
with RRF. Each combination is compared with its own system's base (paired
permutation test), which tests whether expansion helps the weaker system
more than the stronger ones.

Modes (one mode for both sides of a combination):
    separate  base lists + per expansion that expansion's list for every
              retriever (scripts/expansion_lists.py), all fused
    append    one list per retriever, searched with the appended query over
              the appended product text (scripts/expansion_append.py), fused
              like the base

Product combinations: P1 (one of its generator variants, or none) x P2 on/off
x P3 on/off. Query combinations: every subset of the query versions. The
combination with neither is the base and is not scored.

Writes runs/<name>/grid.csv (one row per mode x system x combination) and
per_query.parquet (nDCG@10 and Rcap@100 per query, for the notebook).
p-values are paired permutation tests; q-values adjust them for the number of
tests in the grid (Benjamini-Hochberg). Use q < 0.05 to call a finding.
With --select, only the rows that were significantly better in an earlier
grid are scored, so the q-values cover that smaller family.

Usage:
    uv run python scripts/expansion_grid.py --config configs/expansion_grid.yaml
    uv run python scripts/expansion_grid.py --config configs/expansion_grid.yaml --split test
    uv run python scripts/expansion_grid.py --config configs/expansion_grid_qwen3_0.6b.yaml \
        --select runs/expansion_grid_qwen3_4b/grid.csv
"""

import argparse
import itertools
import logging
from pathlib import Path

import pandas as pd
import yaml

from product_search.data.splits import load_split
from product_search.data.wands import label_grades, load_wands
from product_search.eval.metrics import per_query_metrics
from product_search.eval.stats import benjamini_hochberg, paired_permutation_test
from product_search.expansion.grid import ListBank
from product_search.expansion.lists import (
    LISTS_DIR,
    append_list_dir,
    expansion_keys,
    label,
    product_combinations,
    query_combinations,
)
from product_search.runs import RUNS_DIR, create_run_dir

METRICS = ("nDCG@10", "Rcap@100_exact")
MODES = ("separate", "append")
SELECT_Q = 0.05


def selected_rows(path: Path) -> set[tuple[str, str, str, str]]:
    """(mode, system, products, queries) of the rows significantly better than base."""
    earlier = pd.read_csv(path)
    better = earlier[(earlier["q nDCG@10"] < SELECT_Q) & (earlier["Δ nDCG@10"] > 0)]
    return set(better[["mode", "system", "products", "queries"]].itertuples(index=False, name=None))


def main() -> None:
    parser = argparse.ArgumentParser(description="Score all expansion combinations.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing run.")
    parser.add_argument(
        "--select",
        type=Path,
        help="grid.csv of an earlier run: score only its significantly better rows "
        "(same mode, system and combination keys), e.g. for a query size sweep.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    grid = config["name"]
    # The test split is scored once, at the end, into its own run.
    run_dir = create_run_dir(grid if args.split == "dev" else f"{grid}_test", force=args.force)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    data = load_wands()
    queries_split = getattr(load_split(), args.split)
    labels = label_grades(data.labels)
    bank = ListBank(queries_split, list(data.products["product_id"]))
    retrievers: dict[str, str] = config["retrievers"]
    systems: dict[str, list[str]] = config["systems"]
    for retriever, run in retrievers.items():
        bank.add(f"base__{retriever}", RUNS_DIR / run / "results.parquet")
        for key, source in expansion_keys(config).items():
            path = LISTS_DIR / f"{source}__{retriever}" / "results.parquet"
            if path.exists():
                bank.add(f"{key}__{retriever}", path)

    base = {
        system: per_query_metrics(
            bank.fuse([f"base__{r}" for r in members]), labels, queries_split
        )[list(METRICS)]
        for system, members in systems.items()
    }
    combinations = [
        (products, queries)
        for products, queries in itertools.product(
            product_combinations(config), query_combinations(config)
        )
        if products or queries
    ]

    selected = selected_rows(args.select) if args.select else None

    rows, per_query, waiting = [], [], 0
    for mode, (products, queries) in itertools.product(args.modes, combinations):
        wanted = {
            system
            for system in systems
            if selected is None or (mode, system, label(products), label(queries)) in selected
        }
        if not wanted:
            continue
        if mode == "append":
            # Append lists are many; hold only this combination's in memory.
            for r in retrievers:
                path = append_list_dir(grid, products, queries, r) / "results.parquet"
                if path.exists():
                    bank.add(f"append__{r}", path)
        for system, members in systems.items():
            if system not in wanted:
                continue
            if mode == "append":
                names = [f"append__{r}" for r in members]
            else:
                names = [f"base__{r}" for r in members]
                names += [f"{key}__{r}" for key in (*products, *queries) for r in members]
            if any(n not in bank for n in names):
                # Lists not built yet (a queue scores the grid after each stage).
                waiting += 1
                continue
            table = per_query_metrics(bank.fuse(names), labels, queries_split)[list(METRICS)]
            keys = {"mode": mode, "system": system, "products": label(products)}
            keys |= {"queries": label(queries)}
            row = keys | {"lists": len(names)}
            for metric in METRICS:
                diff, p = paired_permutation_test(base[system][metric], table[metric])
                row |= {
                    metric: table[metric].mean(),
                    f"base {metric}": base[system][metric].mean(),
                    f"Δ {metric}": diff * 100,
                    f"p {metric}": p,
                }
            rows.append(row)
            per_query.append(table.assign(**keys).reset_index())
        if mode == "append":
            for r in retrievers:
                bank.drop(f"append__{r}")
    logging.info("%d rows scored, %d waiting for lists", len(rows), waiting)
    if not rows:
        return

    grid_table = pd.DataFrame(rows)
    # One family per metric: every mode x system x combination in this grid.
    for metric in METRICS:
        grid_table[f"q {metric}"] = benjamini_hochberg(grid_table[f"p {metric}"])
    grid_table.to_csv(run_dir / "grid.csv", index=False)
    pd.concat(per_query).to_parquet(run_dir / "per_query.parquet", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        best = grid_table.sort_values("Δ nDCG@10", ascending=False)
        print(best.groupby(["mode", "system"]).head(3).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
