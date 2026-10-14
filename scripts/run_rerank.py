"""Rerank the first stage's candidates for the dev queries and score the result.

Candidates: the configured first-stage runs fused with RRF (default BM25 +
inf-1.5B + arctic-m). Queries are reranked one at a time, as in serving, and
each is timed; with --timing the machine was prepared (apps closed), so the
latency is reportable.

Creates runs/<name>/ with the config, the reranked top 1,000 per dev query,
meta.json (load and indexing time, reranking latency, memory) and the dev
metrics.

Usage:
    uv run python scripts/run_rerank.py --config configs/rerank_bge_v2_m3.yaml [--timing]
        [--depth 50]
"""

import argparse
import dataclasses
import logging
from pathlib import Path

import torch

from product_search.data.splits import load_split
from product_search.data.wands import load_wands, product_texts, query_texts
from product_search.eval.cost import latency_summary, peak_gpu_gb, peak_ram_gb, timed
from product_search.eval.evaluate import evaluate_run, format_summary
from product_search.eval.metrics import rank
from product_search.retrieval.fusion import rrf
from product_search.retrieval.rerank import (
    load_rerank_config,
    load_scorer,
    rerank,
    save_rerank_config,
)
from product_search.runs import (
    RUNS_DIR,
    create_run_dir,
    git_info,
    load_results,
    machine_info,
    save_results,
    write_json,
)

WARMUP_QUERIES = 10


def main() -> None:
    parser = argparse.ArgumentParser(description="Rerank first-stage candidates on dev.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--timing", action="store_true", help="Machine prepared for timing.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing run.")
    parser.add_argument("--queries", type=int, help="Only the first n dev queries (trial run).")
    parser.add_argument(
        "--depth", type=int, help="Override the config's depth; the run is named <name>_d<depth>."
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    name, config = load_rerank_config(args.config)
    if args.depth is not None:
        name, config = f"{name}_d{args.depth}", dataclasses.replace(config, depth=args.depth)
    run_dir = create_run_dir(name, force=args.force)
    save_rerank_config(run_dir / "config.yaml", name, config)

    data = load_wands()
    dev = load_split().dev[: args.queries]
    products = product_texts(data.products, config.fields)
    queries = {q: text for q, text in query_texts(data.queries).items() if q in set(dev)}

    fused = rrf([load_results(RUNS_DIR / run / "results.parquet") for run in config.first_stage])
    first_stage = {q: fused[q] for q in queries}
    pool = {p for ranking in rank(first_stage, config.depth).values() for p in ranking}

    with timed() as loading:
        scorer = load_scorer(config)
    with timed() as indexing:
        scorer.index({p: products[p] for p in pool})

    sync = torch.cuda.synchronize if torch.cuda.is_available() else None
    warmup = dict(list(queries.items())[:WARMUP_QUERIES])
    rerank(scorer, warmup, first_stage, config.depth, sync)
    results, latencies = rerank(scorer, queries, first_stage, config.depth, sync)
    save_results(results, run_dir / "results.parquet")

    latency = latency_summary(latencies)
    cost = {
        "load_seconds": loading["seconds"],
        "indexing_seconds": indexing["seconds"],
        "indexed_products": len(pool),
        "rerank_latency": latency,
        "peak_ram_gb": peak_ram_gb(),
        "peak_gpu_gb": peak_gpu_gb(),
    }
    write_json(
        {"git": git_info(), "machine": machine_info(), "timed": args.timing, "cost": cost},
        run_dir / "meta.json",
    )
    print(f"{name} on dev ({len(queries)} queries, depth {config.depth})")
    # A trial run covers only some queries, so it is not scored or added to the results table.
    if args.queries is None:
        print(format_summary(evaluate_run(run_dir, "dev")))
    print(f"  load {loading['seconds']:.0f}s, indexing {indexing['seconds']:.0f}s")
    print(f"  rerank latency p50 {latency['p50_ms']:.0f} ms, p99 {latency['p99_ms']:.0f} ms")


if __name__ == "__main__":
    main()
