"""Turn expansion runs into search lists, one per retriever.

Product expansion -> a separate index over the generated text only, searched
with the raw query. Query expansion -> the original product index, searched
with "query + generated text". Original texts and indexes are never changed;
the grid later fuses these lists with the original ones.

Lists cover dev and test queries; the grid picks the split. A query whose
expansion is empty is left out of that expansion's list, so it does not repeat
the original search with double weight.

Writes runs/lists/<source>__<retriever>/results.parquet (top 1,000 per query)
and meta.json (source, retriever, seconds, commit).
inf-1.5B needs the transformers 4.57 overlay (see configs/dense_inf_15b.yaml),
so it runs in its own call with --retrievers inf15.

Usage:
    uv run python scripts/expansion_lists.py --retrievers bm25 arcticm \
        --sources expand_p2_qwen3_8b expand_q1_qwen3_4b
"""

import argparse
import logging

from product_search.data.splits import load_split
from product_search.data.wands import load_wands, product_texts, query_texts
from product_search.eval.cost import timed
from product_search.expansion.lists import (
    LISTS_DIR,
    RETRIEVERS,
    is_ready,
    load_retriever,
    load_source,
)
from product_search.runs import RESULT_DEPTH, git_info, save_results, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Search lists from expansion runs.")
    parser.add_argument("--retrievers", nargs="+", choices=list(RETRIEVERS), required=True)
    parser.add_argument("--sources", nargs="+", required=True, help="Expansion run names.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing lists.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    data = load_wands()
    split = load_split()
    query_ids = split.dev + split.test
    raw_queries = query_texts(data.queries)
    waiting = [source for source in args.sources if not is_ready(source)]
    if waiting:
        logging.info("not finished yet, skipped: %s", waiting)
    sources = {source: load_source(source) for source in args.sources if is_ready(source)}
    # Query expansions all search the original index: do them first, build it once.
    sources = dict(sorted(sources.items(), key=lambda item: item[1][0] != "queries"))

    for name in args.retrievers:
        retriever, fields = load_retriever(name)
        original_indexed = False
        for source, (target, texts) in sources.items():
            out = LISTS_DIR / f"{source}__{name}"
            if (out / "results.parquet").exists() and not args.force:
                logging.info("%s exists, skipped", out.name)
                continue
            with timed() as took:
                if target == "products":
                    retriever.index({p: t for p, t in texts.items() if t})
                    queries = {q: raw_queries[q] for q in query_ids}
                else:
                    if not original_indexed:
                        retriever.index(product_texts(data.products, fields))
                        original_indexed = True
                    queries = {
                        q: f"{raw_queries[q]}\n{texts[q]}" for q in query_ids if texts.get(q)
                    }
                results = retriever.search(queries, RESULT_DEPTH)
            out.mkdir(parents=True, exist_ok=True)
            save_results(results, out / "results.parquet")
            meta = {"mode": "separate", "source": source, "target": target, "retriever": name}
            meta |= {"queries_searched": len(queries)}
            write_json(meta | {"seconds": took["seconds"], "git": git_info()}, out / "meta.json")
            logging.info("%s: %.0fs", out.name, took["seconds"])


if __name__ == "__main__":
    main()
