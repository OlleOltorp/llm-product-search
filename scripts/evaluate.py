"""Score a run on dev (or, once, on test) and record the result.

Usage:
    uv run python scripts/evaluate.py runs/bm25_default
    uv run python scripts/evaluate.py runs/final --split test --confirm-test
"""

import argparse
from pathlib import Path

from product_search.eval.evaluate import evaluate_run, format_summary
from product_search.eval.stats import N_RESAMPLES
from product_search.runs import RESULTS_DIR


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

    summary = evaluate_run(args.run_dir, args.split, args.results_dir, args.resamples)
    print(f"{args.run_dir.name} on {args.split}")
    print(format_summary(summary))


if __name__ == "__main__":
    main()
