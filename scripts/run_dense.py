"""Run one dense config on the dev queries and score it.

Creates runs/<name>/ with the config, the top 1,000 products per dev query,
meta.json (git, machine, load and indexing time, index size, memory,
optional latency) and the dev metrics.

Usage:
    uv run python scripts/run_dense.py --config configs/dense_bge_small.yaml [--timing]
"""

import argparse
import logging
import tempfile
import time
from pathlib import Path

import torch

from product_search.data.splits import load_split
from product_search.data.wands import load_wands, product_texts, query_texts
from product_search.eval.cost import (
    dir_size_mb,
    latency_summary,
    peak_gpu_gb,
    peak_ram_gb,
    time_queries,
    timed,
)
from product_search.eval.evaluate import evaluate_run, format_summary
from product_search.retrieval.dense import DenseRetriever, load_dense_config, save_dense_config
from product_search.runs import (
    RESULT_DEPTH,
    create_run_dir,
    git_info,
    machine_info,
    save_results,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one dense config on dev.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--timing", action="store_true", help="Measure per-query latency.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing run.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    name, config = load_dense_config(args.config)
    run_dir = create_run_dir(name, force=args.force)
    save_dense_config(run_dir / "config.yaml", name, config)

    data = load_wands()
    dev = load_split().dev
    products = product_texts(data.products, config.fields)
    queries = query_texts(data.queries)
    dev_queries = {q: queries[q] for q in dev}

    with timed() as loading:
        retriever = DenseRetriever(config)
    with timed() as indexing:
        retriever.index(products)
    with tempfile.TemporaryDirectory() as tmp:
        retriever.save(Path(tmp))
        index_size = dir_size_mb(Path(tmp))

    start = time.perf_counter()
    results = retriever.search(dev_queries, RESULT_DEPTH)
    batch_seconds = time.perf_counter() - start
    save_results(results, run_dir / "results.parquet")

    latency = None
    if args.timing:
        texts = list(dev_queries.values())
        sync = torch.cuda.synchronize if torch.cuda.is_available() else None
        latency = latency_summary(
            time_queries(lambda text: retriever.search({"q": text}, RESULT_DEPTH), texts, sync=sync)
        )

    cost = {
        "load_seconds": loading["seconds"],
        "indexing_seconds": indexing["seconds"],
        "index_size_mb": index_size,
        "batch_queries_per_second": len(dev) / batch_seconds,
        "latency": latency,
        "peak_ram_gb": peak_ram_gb(),
        "peak_gpu_gb": peak_gpu_gb(),
    }
    write_json(
        {"git": git_info(), "machine": machine_info(), "timed": args.timing, "cost": cost},
        run_dir / "meta.json",
    )
    summary = evaluate_run(run_dir, "dev")

    print(f"{name} on dev ({len(dev)} queries)")
    print(format_summary(summary))
    print(f"  indexing {indexing['seconds']:.1f}s, index {index_size:.0f} MB")
    if latency is not None:
        print(f"  latency p50 {latency['p50_ms']:.1f} ms, p99 {latency['p99_ms']:.1f} ms")


if __name__ == "__main__":
    main()
