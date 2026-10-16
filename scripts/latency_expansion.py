"""Live latency of query-time expansion, one query at a time (dev queries).

Three steps (see product_search.expansion.latency for how pieces add up):
    llm      time to generate each query version (Q1-Q4) for one query, per
             model and engine (transformers in this process, or a vLLM server
             started with scripts/serve_vllm.ps1 for the same model)
    search   time of every search a combination can need, per retriever
             (with vLLM also every set of versions sent at once)
    compose  per query, the time of every significantly better combination in
             the expansion grids; p50 and p99, and whether p99 fits the budget

Writes runs/latency_expansion/ (or --out):
    llm_<generator>_<engine>.parquet   query_id, version, ms, tokens
    search_<retriever>.parquet         query_id, list, ms (median of repeats)
    compose.csv                        one row per combination x generator x engine

Usage (tower, quiet machine; inf15 needs the transformers 4.57 overlay):
    uv run python scripts/latency_expansion.py llm --generator qwen3_0.6b
    uv run python scripts/latency_expansion.py llm --generator qwen3_0.6b --engine vllm
    uv run python scripts/latency_expansion.py search --retriever bm25
    uv run python scripts/latency_expansion.py compose
"""

import argparse
import dataclasses
import itertools
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from product_search.data.splits import load_split
from product_search.data.wands import load_wands, product_texts, query_texts
from product_search.eval.cost import latency_summary, time_queries
from product_search.expansion.generate import Generator, load_expansion_config, query_inputs
from product_search.expansion.latency import combination_ms
from product_search.expansion.lists import (
    APPEND_DIR,
    INDEX_DIR,
    RETRIEVERS,
    first_stage_titles,
    load_retriever,
    load_source,
)
from product_search.retrieval.bm25 import BM25Retriever
from product_search.retrieval.dense import DenseRetriever
from product_search.runs import REPO_DIR, RESULT_DEPTH, RUNS_DIR, read_json

OUT_DIR = RUNS_DIR / "latency_expansion"
CONFIGS = REPO_DIR / "configs"
GENERATORS = CONFIGS / "generators.yaml"
VERSIONS = ("q1", "q2", "q3", "q4")
# Query generator -> its grid (configs/expansion_grid*.yaml names).
GRIDS = {"qwen3_4b": "expansion_grid_qwen3_4b", "qwen3_0.6b": "expansion_grid_qwen3_0.6b"}
BUDGET_MS = 500.0
LLM_WARMUP = 10
SELECT_Q = 0.05


def dev_queries(limit: int | None) -> tuple[list[str], dict[str, str]]:
    data = load_wands()
    return load_split().dev[:limit], query_texts(data.queries)


def run_llm(args: argparse.Namespace) -> None:
    """Generate each query version for one query at a time and time it.

    With vLLM, every set of two or more versions is also timed sent at once
    (until the last reply), as a service would send them.
    """
    ids, raw = dev_queries(args.limit)
    grounding = first_stage_titles(load_wands().products, ids, 5)
    sync = torch.cuda.synchronize if torch.cuda.is_available() else None
    configs = {}
    for version in VERSIONS:
        _, config = load_expansion_config(
            CONFIGS / f"expand_{version}.yaml", args.generator, GENERATORS
        )
        configs[version] = dataclasses.replace(config, engine=args.engine, batch_size=1)
    if args.engine == "vllm":
        # Only a tokenizer each; the server holds the model.
        generators = {v: Generator(c) for v, c in configs.items()}
    else:
        # One model; the config (prompt, output length) is swapped per version below.
        shared = Generator(configs[VERSIONS[0]])
        generators = dict.fromkeys(VERSIONS, shared)

    rows: list[tuple[str, str, float, int]] = []
    prompts: dict[str, dict[str, str]] = {}

    def log(label: str) -> None:
        summary = latency_summary([ms for _, v, ms, _ in rows if v == label])
        logging.info(
            "%s %s %s: p50 %.0f ms, p99 %.0f ms",
            args.generator,
            args.engine,
            label,
            summary["p50_ms"],
            summary["p99_ms"],
        )

    for version, config in configs.items():
        generator = generators[version]
        generator.config = config
        kept = [q for q in ids if q in grounding] if config.grounding_k else ids
        titles = [grounding[q][: config.grounding_k] for q in kept] if config.grounding_k else None
        prompts[version] = dict(
            zip(kept, query_inputs(generator, [raw[q] for q in kept], titles), strict=True)
        )
        texts = list(prompts[version].values())
        for i in range(LLM_WARMUP):
            generator.generate([texts[i % len(texts)]])
        for query_id, prompt in prompts[version].items():
            start = time.perf_counter_ns()
            _, tokens = generator.generate([prompt])
            if sync is not None and args.engine == "transformers":
                sync()
            rows.append((query_id, version, (time.perf_counter_ns() - start) / 1e6, tokens))
        log(version)

    if args.engine == "vllm":
        with ThreadPoolExecutor(len(VERSIONS)) as pool:
            for size in range(2, len(VERSIONS) + 1):
                for subset in itertools.combinations(VERSIONS, size):
                    label = "+".join(subset)
                    kept = [q for q in ids if all(q in prompts[v] for v in subset)]

                    def together(query_id: str, subset: tuple[str, ...] = subset) -> int:
                        futures = [
                            pool.submit(generators[v].generate, [prompts[v][query_id]])
                            for v in subset
                        ]
                        return sum(f.result()[1] for f in futures)

                    for query_id in kept[:LLM_WARMUP]:
                        together(query_id)
                    for query_id in kept:
                        start = time.perf_counter_ns()
                        tokens = together(query_id)
                        rows.append(
                            (query_id, label, (time.perf_counter_ns() - start) / 1e6, tokens)
                        )
                    log(label)

    table = pd.DataFrame(rows, columns=["query_id", "version", "ms", "tokens"])
    table.to_parquet(args.out / f"llm_{args.generator}_{args.engine}.parquet", index=False)


def original_index(
    name: str, retriever: BM25Retriever | DenseRetriever, fields: tuple[str, ...]
) -> None:
    """Index the original product texts; dense models reuse saved embeddings when present."""
    if isinstance(retriever, DenseRetriever):
        saved = INDEX_DIR / GRIDS["qwen3_4b"] / f"none__{name}"
        if (saved / "embeddings.npy").exists():
            retriever.load(saved, read_json(saved / "product_ids.json")["product_ids"])
            return
    retriever.index(product_texts(load_wands().products, fields))


def run_search(args: argparse.Namespace) -> None:
    """Time every search a combination can need, for one retriever."""
    name = args.retriever
    retriever, fields = load_retriever(name)
    ids, raw = dev_queries(args.limit)
    lists: dict[str, dict[str, str]] = {"base": {q: raw[q] for q in ids}}
    for generator, grid in GRIDS.items():
        for version in VERSIONS:
            run = f"expand_{version}_{generator}"
            if not (RUNS_DIR / run / "expansions.parquet").exists():
                continue
            texts = load_source(run)[1]
            lists[f"{version}@{generator}"] = {
                q: f"{raw[q]}\n{texts[q]}" for q in ids if texts.get(q)
            }
        if name == "bm25":
            # The exact appended queries the grid searched.
            for path in sorted((APPEND_DIR / grid).glob("queries__*__bm25.parquet")):
                label = path.stem.split("__")[1]
                table = pd.read_parquet(path)
                texts = dict(zip(table["query_id"], table["text"], strict=True))
                lists[f"append:{label}@{generator}"] = {q: texts[q] for q in ids}

    sync = torch.cuda.synchronize if torch.cuda.is_available() and name != "bm25" else None
    rows: list[tuple[str, str, float]] = []

    def time_lists(selected: dict[str, dict[str, str]]) -> None:
        for list_name, queries in selected.items():
            kept = list(queries)
            ms = time_queries(
                lambda text: retriever.search({"q": text}, RESULT_DEPTH),
                [queries[q] for q in kept],
                warmup=min(100, len(kept)),
                sync=sync,
            )
            # time_queries runs all queries once per repeat; keep each query's median.
            medians = np.median(np.asarray(ms).reshape(-1, len(kept)), axis=0)
            rows.extend((q, list_name, float(m)) for q, m in zip(kept, medians, strict=True))
            logging.info("%s %s: p99 %.1f ms", name, list_name, np.percentile(medians, 99))

    original_index(name, retriever, fields)
    time_lists(lists)
    if name == "bm25":
        grid = yaml.safe_load((CONFIGS / "expansion_grid.yaml").read_text(encoding="utf-8"))
        for key, run in (grid["p1"] | grid["products"]).items():
            texts = load_source(run)[1]
            retriever.index({p: t for p, t in texts.items() if t})
            time_lists({f"product:{key}": lists["base"]})
    pd.DataFrame(rows, columns=["query_id", "list", "ms"]).to_parquet(
        args.out / f"search_{name}.parquet", index=False
    )


def run_compose(args: argparse.Namespace) -> None:
    """Time every significantly better combination per query; p50, p99, budget."""
    systems = yaml.safe_load((CONFIGS / "expansion_grid.yaml").read_text(encoding="utf-8"))[
        "systems"
    ]
    search = {
        # A query with an empty expansion is not searched in that list: 0 ms.
        r: pd.read_parquet(args.out / f"search_{r}.parquet")
        .pivot(index="query_id", columns="list", values="ms")
        .fillna(0.0)
        for r in RETRIEVERS
    }
    rows = []
    for path in sorted(args.out.glob("llm_*.parquet")):
        generator, engine = path.stem.removeprefix("llm_").rsplit("_", 1)
        llm = pd.read_parquet(path).pivot(index="query_id", columns="version", values="ms")
        grid_csv = RUNS_DIR / GRIDS[generator] / "grid.csv"
        if not grid_csv.exists():
            logging.warning("%s missing, skipped", grid_csv)
            continue
        grid = pd.read_csv(grid_csv)
        better = grid[(grid["q nDCG@10"] < SELECT_Q) & (grid["Δ nDCG@10"] > 0)]
        queries = [q for q in llm.index if all(q in s.index for s in search.values())]
        for _, row in better.iterrows():
            members = systems[row["system"]]
            products = () if row["products"] == "none" else tuple(row["products"].split("+"))
            versions = () if row["queries"] == "none" else tuple(row["queries"].split("+"))
            if row["mode"] == "append" and any(r != "bm25" for r in members):
                continue
            # Versions sent at once, when that was timed (vLLM); else one after another.
            together = "+".join(versions) if len(versions) > 1 else ""
            concurrent = together in llm.columns
            times = [
                combination_ms(
                    row["mode"],
                    members,
                    products,
                    versions,
                    generator,
                    llm.loc[q].to_dict(),
                    {r: search[r].loc[q].to_dict() for r in search},
                    llm_total=float(llm[together][q]) if concurrent else None,
                )
                for q in queries
            ]
            summary = latency_summary(times)
            rows.append(
                {
                    "generator": generator,
                    "engine": engine,
                    "mode": row["mode"],
                    "system": row["system"],
                    "products": row["products"],
                    "queries": row["queries"],
                    "nDCG@10": row["nDCG@10"],
                    "Δ nDCG@10": row["Δ nDCG@10"],
                    "q nDCG@10": row["q nDCG@10"],
                    "llm_calls": "together" if concurrent else "one by one",
                    "p50_ms": summary["p50_ms"],
                    "p99_ms": summary["p99_ms"],
                    "within_budget": summary["p99_ms"] <= BUDGET_MS,
                    "queries_timed": len(times),
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(args.out / "compose.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        best = table.sort_values("nDCG@10", ascending=False)
        print(
            best.groupby(["generator", "engine", "system"]).head(3).round(3).to_string(index=False)
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Live latency of query-time expansion.")
    parser.add_argument("step", choices=("llm", "search", "compose"))
    parser.add_argument("--generator", choices=list(GRIDS), help="llm: query generator.")
    parser.add_argument("--engine", choices=("transformers", "vllm"), default="transformers")
    parser.add_argument("--retriever", choices=list(RETRIEVERS), help="search: retriever.")
    parser.add_argument("--limit", type=int, help="Only the first n dev queries (smoke test).")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("bm25s").setLevel(logging.WARNING)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.step == "llm":
        if args.generator is None:
            parser.error("llm needs --generator")
        run_llm(args)
    elif args.step == "search":
        if args.retriever is None:
            parser.error("search needs --retriever")
        run_search(args)
    else:
        run_compose(args)


if __name__ == "__main__":
    main()
