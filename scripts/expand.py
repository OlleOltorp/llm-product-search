"""Generate expansions for all products or for all queries (dev and test).

Creates runs/<name>/ with the config, expansions.parquet (id, raw, text) and
meta.json (time, generated tokens, tokens per second, share of empty outputs).
Prints three examples so the output can be checked by eye.

Usage:
    uv run python scripts/expand.py --config configs/expand_p1_t5.yaml
    uv run python scripts/expand.py --config configs/expand_p2.yaml --generator qwen3_8b
    uv run python scripts/expand.py --config configs/expand_q1.yaml --generator qwen3_4b --limit 20
    uv run python scripts/expand.py --config configs/expand_q4.yaml --generator qwen3_0.6b \
        --add-missing
"""

import argparse
import logging
import random
from pathlib import Path

import pandas as pd

from product_search.data.splits import load_split
from product_search.data.wands import load_wands, product_texts, query_texts
from product_search.eval.cost import peak_gpu_gb, timed
from product_search.expansion.generate import (
    Generator,
    clean,
    load_expansion_config,
    product_inputs,
    query_inputs,
    save_expansion_config,
)
from product_search.expansion.lists import first_stage_titles
from product_search.runs import (
    REPO_DIR,
    RUNS_DIR,
    create_run_dir,
    git_info,
    machine_info,
    write_json,
)

GENERATORS = REPO_DIR / "configs/generators.yaml"
SAMPLE_SEED = 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate product or query expansions.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--generator", help="Model key in configs/generators.yaml.")
    parser.add_argument("--limit", type=int, help="Only n inputs (trial run, name gets _trial).")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing run.")
    parser.add_argument(
        "--add-missing",
        action="store_true",
        help="Add only the inputs an existing run lacks (Q4 test queries, once the first "
        "stage covers test); earlier outputs are kept as they are.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    name, config = load_expansion_config(args.config, args.generator, GENERATORS)
    if args.limit is not None:
        name = f"{name}_trial"
    existing = None
    if args.add_missing:
        run_dir = RUNS_DIR / name
        if load_expansion_config(run_dir / "config.yaml")[1] != config:
            raise ValueError(f"{name}: config differs from the existing run")
        existing = pd.read_parquet(run_dir / "expansions.parquet")
    else:
        run_dir = create_run_dir(name, force=args.force)
        save_expansion_config(run_dir / "config.yaml", name, config)

    data = load_wands()
    if config.target == "products":
        texts = product_texts(data.products, config.fields)
        ids = sorted(texts)
        if args.limit is not None:
            ids = sorted(random.Random(SAMPLE_SEED).sample(ids, args.limit))
    else:
        queries = query_texts(data.queries)
        # Dev and test: expansion uses no labels, and the test run later needs
        # the same outputs. A trial takes dev queries first.
        split = load_split()
        ids = (split.dev + split.test)[: args.limit]
    if existing is not None:
        done = set(existing["id"])
        ids = [i for i in ids if i not in done]
        if not ids:
            print(f"{name}: nothing to add")
            return

    with timed() as loading:
        generator = Generator(config)
    if config.target == "products":
        inputs = product_inputs(generator, [texts[i] for i in ids])
    else:
        titles = None
        if config.grounding_k:
            grounding = first_stage_titles(data.products, ids, config.grounding_k)
            # The base runs cover dev only until the final test run.
            if len(grounding) < len(ids):
                logging.warning(
                    "no first-stage results for %d queries, left out", len(ids) - len(grounding)
                )
                ids = [q for q in ids if q in grounding]
            titles = [grounding[q] for q in ids]
        inputs = query_inputs(generator, [queries[q] for q in ids], titles)

    with timed() as generating:
        raw, new_tokens = generator.generate(inputs)
    cleaned = [clean(r, config.parse, config.dedupe) for r in raw]
    table = pd.DataFrame({"id": ids, "raw": raw, "text": cleaned})
    if existing is not None:
        table = pd.concat([existing, table], ignore_index=True)
    table.to_parquet(run_dir / "expansions.parquet", index=False)

    empty = sum(not t for t in cleaned) / len(cleaned)
    thinking = sum("<think>" in r for r in raw) / len(raw)
    seconds_per_input = generating["seconds"] / len(ids)
    cost = {
        "inputs": len(ids),
        "load_seconds": loading["seconds"],
        "generate_seconds": generating["seconds"],
        "generated_tokens": new_tokens,
        "tokens_per_second": new_tokens / generating["seconds"],
        "seconds_per_input": seconds_per_input,
        "peak_gpu_gb": peak_gpu_gb(),
    }
    write_json(
        {
            "git": git_info(),
            "machine": machine_info(),
            "cost": cost,
            "empty_share": empty,
            "think_share": thinking,
        },
        run_dir / ("meta.json" if existing is None else "meta_added.json"),
    )

    print(f"{name}: {len(ids)} {config.target}")
    print(
        f"  {generating['seconds']:.0f}s, {new_tokens / generating['seconds']:.0f} tokens/s, "
        f"{seconds_per_input * 1000:.0f} ms per input, empty {empty:.1%}, "
        f"thinking {thinking:.1%}"
    )
    for i in range(min(3, len(ids))):
        print(f"  --- {ids[i]}\n  {cleaned[i][:400]}")


if __name__ == "__main__":
    main()
