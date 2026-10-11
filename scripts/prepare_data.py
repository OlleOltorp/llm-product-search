"""Download, clean and cache WANDS as Parquet.

Usage:
    uv run python scripts/prepare_data.py [--force] [--data-dir DIR]
"""

import argparse
import logging
from pathlib import Path

from product_search.data.wands import DEFAULT_DATA_DIR, prepare_wands


def main() -> None:
    parser = argparse.ArgumentParser(description="Download, clean and cache WANDS.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--force", action="store_true", help="Rebuild processed files.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    prepare_wands(args.data_dir, force=args.force)


if __name__ == "__main__":
    main()
