"""Make the fixed dev/test split of WANDS queries and save it to splits/wands.json.

Usage:
    uv run python scripts/make_split.py [--seed N] [--force]
"""

import argparse
import logging

from product_search.data.splits import DEFAULT_SPLIT_PATH, make_split, save_split
from product_search.data.wands import load_wands


def main() -> None:
    parser = argparse.ArgumentParser(description="Make the dev/test split of WANDS queries.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--force", action="store_true", help="Overwrite an existing split.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    data = load_wands()
    split = make_split(data, seed=args.seed)
    save_split(split, DEFAULT_SPLIT_PATH, force=args.force)

    queries = dict(zip(data.queries["query_id"], data.queries["query"], strict=True))
    print(f"dev {len(split.dev)} / test {len(split.test)} -> {DEFAULT_SPLIT_PATH}")
    for group in split.duplicate_groups:
        side = "dev" if group[0] in split.dev else "test"
        print(f"  kept together ({side}): {[queries[q] for q in group]}")


if __name__ == "__main__":
    main()
