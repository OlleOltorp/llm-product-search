"""Append-mode lists: every product x query combination of a grid config.

For each product combination the generated texts are written into the
product text and indexed once; every query combination then searches that
index with the appended query (see product_search.expansion.append). Each
(products, queries, retriever) gives one list; the grid fuses the lists of a
system's retrievers with RRF, exactly as the base is fused.

Dense product embeddings are saved per product combination, so a crashed run
resumes without re-encoding, and later analysis can reuse them. Lists that
exist are skipped, and so are combinations whose expansion runs have not
finished yet, so the script can be called again after each stage of a queue.

Writes, under the grid config's name:
    runs/lists/append/<grid>/<products>__<queries>__<retriever>/
        results.parquet  top 1,000 per query (dev and test)
        meta.json        sources, seconds, commit
    runs/lists/append/<grid>/queries__<queries>__<retriever>.parquet
        the query texts searched (query_id, text)
    runs/append_index/<grid>/<products>__<retriever>/  (dense only)
        embeddings.npy, product_ids.json, meta.json (seconds, share of texts cut)

inf-1.5B needs the transformers 4.57 overlay, so it runs in its own call.

Usage:
    uv run python scripts/expansion_append.py --config configs/expansion_grid.yaml \
        --retrievers bm25 arcticm
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml

from product_search.data.splits import load_split
from product_search.data.wands import load_wands, product_texts, query_texts
from product_search.eval.cost import timed
from product_search.expansion.append import (
    BM25_QUERY_REPEATS,
    DENSE_QUERY_REPEATS,
    appended_product_texts,
    appended_query,
)
from product_search.expansion.lists import (
    APPEND_DIR,
    INDEX_DIR,
    RETRIEVERS,
    append_list_dir,
    expansion_keys,
    is_ready,
    label,
    load_retriever,
    load_source,
    product_combinations,
    query_combinations,
)
from product_search.retrieval.dense import DenseRetriever
from product_search.runs import RESULT_DEPTH, git_info, read_json, save_results, write_json


def check_config(grid_dir: Path, config: dict, force: bool) -> None:
    """Keep a copy of the grid config; refuse to mix lists from a changed one."""
    path = grid_dir / "config.yaml"
    if path.exists() and yaml.safe_load(path.read_text(encoding="utf-8")) != config and not force:
        raise ValueError(f"{path} differs from the given config; use a new grid name or --force")
    grid_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def share_cut(retriever: DenseRetriever, texts: list[str]) -> float:
    """Share of product texts longer than the model's max_seq_length."""
    tokenizer = retriever.model.tokenizer
    lengths = [len(ids) for ids in tokenizer(texts, add_special_tokens=True)["input_ids"]]
    return sum(n > retriever.config.max_seq_length for n in lengths) / len(lengths)


def index_dense(
    retriever: DenseRetriever, texts: dict[str, str], index_dir: Path, force: bool
) -> None:
    """Load saved embeddings, or encode the texts and save them."""
    if (index_dir / "embeddings.npy").exists() and not force:
        retriever.load(index_dir, read_json(index_dir / "product_ids.json")["product_ids"])
        logging.info("  loaded %s", index_dir.name)
        return
    with timed() as took:
        retriever.index(texts)
    index_dir.mkdir(parents=True, exist_ok=True)
    retriever.save(index_dir)
    write_json({"product_ids": list(texts)}, index_dir / "product_ids.json")
    cut = share_cut(retriever, list(texts.values()))
    write_json({"seconds": took["seconds"], "share_cut": cut}, index_dir / "meta.json")
    logging.info(
        "  encoded %s: %.0fs, %.1f%% of texts cut", index_dir.name, took["seconds"], cut * 100
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Append-mode lists for a grid config.")
    parser.add_argument("--config", type=Path, required=True, help="Grid config.")
    parser.add_argument("--retrievers", nargs="+", choices=list(RETRIEVERS), required=True)
    parser.add_argument("--force", action="store_true", help="Overwrite lists and indexes.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    grid = config["name"]
    grid_dir = APPEND_DIR / grid
    check_config(grid_dir, config, args.force)

    data = load_wands()
    split = load_split()
    query_ids = split.dev + split.test
    raw_queries = query_texts(data.queries)
    keys = expansion_keys(config)
    # The queue runs the fastest expansions first and calls this script after
    # each stage; combinations that need an unfinished run wait for a later call.
    ready = {key for key, run in keys.items() if is_ready(run)}
    if ready != set(keys):
        logging.info("not ready yet: %s", sorted(set(keys) - ready))
    texts = {key: load_source(keys[key])[1] for key in ready}
    products_options = [p for p in product_combinations(config) if set(p) <= ready]
    query_options = [q for q in query_combinations(config) if set(q) <= ready]

    for name in args.retrievers:
        retriever, fields = load_retriever(name)
        repeats = BM25_QUERY_REPEATS if name == "bm25" else DENSE_QUERY_REPEATS
        for products in products_options:
            # No products and no queries is the base run itself.
            todo = [
                queries
                for queries in query_options
                if (products or queries)
                and (
                    args.force
                    or not (
                        append_list_dir(grid, products, queries, name) / "results.parquet"
                    ).exists()
                )
            ]
            if not todo:
                logging.info("%s %s: all lists exist, skipped", name, label(products))
                continue
            logging.info("%s %s: %d lists", name, label(products), len(todo))
            product_text = (
                appended_product_texts(data.products, fields, [texts[k] for k in products])
                if products
                else product_texts(data.products, fields)
            )
            if isinstance(retriever, DenseRetriever):
                index_dir = INDEX_DIR / grid / f"{label(products)}__{name}"
                index_dense(retriever, product_text, index_dir, args.force)
            else:
                retriever.index(product_text)

            for queries in todo:
                searched = {
                    q: appended_query(
                        raw_queries[q], [texts[k].get(q, "") for k in queries], repeats
                    )
                    for q in query_ids
                }
                with timed() as took:
                    results = retriever.search(searched, RESULT_DEPTH)
                out = append_list_dir(grid, products, queries, name)
                out.mkdir(parents=True, exist_ok=True)
                save_results(results, out / "results.parquet")
                query_file = grid_dir / f"queries__{label(queries)}__{name}.parquet"
                if not query_file.exists() or args.force:
                    pd.DataFrame(
                        {"query_id": list(searched), "text": list(searched.values())}
                    ).to_parquet(query_file, index=False)
                meta = {
                    "mode": "append",
                    "grid": grid,
                    "retriever": name,
                    "products": {k: keys[k] for k in products},
                    "queries": {k: keys[k] for k in queries},
                    "query_repeats": repeats,
                    "search_seconds": took["seconds"],
                    "git": git_info(),
                }
                write_json(meta, out / "meta.json")


if __name__ == "__main__":
    main()
