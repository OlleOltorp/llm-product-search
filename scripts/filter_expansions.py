"""Drop the worst generated queries, as in Doc2Query-- (Gospodinov et al., ECIR 2023).

Every generated query is scored against its own product with a cross-encoder;
queries below a corpus-wide score threshold are dropped, keeping the given
share of all queries. Writes a new run <source>_filtered with the same files.

Usage:
    uv run python scripts/filter_expansions.py --source expand_p1_t5 \
        --scorer configs/rerank_gte_modernbert.yaml --keep 0.7
"""

import argparse
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import CrossEncoder

from product_search.data.wands import load_wands, product_texts
from product_search.eval.cost import timed
from product_search.retrieval.dense import DTYPES
from product_search.retrieval.rerank import load_rerank_config
from product_search.runs import RUNS_DIR, create_run_dir, read_json, write_json

MAX_LENGTH = 256
BATCH_SIZE = 128


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter generated queries with a cross-encoder.")
    parser.add_argument("--source", required=True, help="Expansion run with one query per line.")
    parser.add_argument("--scorer", type=Path, required=True, help="cross_encoder rerank config.")
    parser.add_argument("--keep", type=float, default=0.7, help="Share of queries kept.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing run.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    source_dir = RUNS_DIR / args.source
    expansions = pd.read_parquet(source_dir / "expansions.parquet")
    _, scorer_config = load_rerank_config(args.scorer)
    if scorer_config.method != "cross_encoder":
        raise ValueError(f"{args.scorer}: the filter needs a cross_encoder config")
    texts = product_texts(load_wands().products, scorer_config.fields)

    pairs, owners = [], []
    for row, (product_id, text) in enumerate(
        zip(expansions["id"], expansions["text"], strict=True)
    ):
        for query in text.splitlines():
            pairs.append((query, texts[product_id]))
            owners.append(row)

    model = CrossEncoder(
        scorer_config.model,
        revision=scorer_config.revision,
        max_length=MAX_LENGTH,
        model_kwargs={"dtype": DTYPES[scorer_config.dtype]},
    )
    with timed() as scoring:
        scores = model.predict(pairs, batch_size=BATCH_SIZE, show_progress_bar=False)
    threshold = float(np.quantile(scores, 1 - args.keep))

    kept: list[list[str]] = [[] for _ in range(len(expansions))]
    for (query, _), row, score in zip(pairs, owners, scores, strict=True):
        if score >= threshold:
            kept[row].append(query)
    filtered = expansions.assign(text=["\n".join(queries) for queries in kept])

    name = f"{args.source}_filtered"
    run_dir = create_run_dir(name, force=args.force)
    shutil.copy(source_dir / "config.yaml", run_dir / "config.yaml")
    filtered.to_parquet(run_dir / "expansions.parquet", index=False)
    meta = read_json(source_dir / "meta.json") | {
        "filter": {
            "scorer": scorer_config.model,
            "keep": args.keep,
            "threshold": threshold,
            "queries_before": len(pairs),
            "queries_after": sum(len(q) for q in kept),
            "products_left_empty": sum(not q for q in kept),
            "seconds": scoring["seconds"],
        }
    }
    write_json(meta, run_dir / "meta.json")
    print(f"{name}: kept {meta['filter']['queries_after']} of {len(pairs)} queries")
    print(
        f"  {scoring['seconds']:.0f}s, products left empty {meta['filter']['products_left_empty']}"
    )


if __name__ == "__main__":
    main()
