"""Quick check that reranker configs load and score sensibly, before a full run.

Per config:
    - loads the pinned model and reports load time;
    - four toy queries each score their obvious product highest;
    - products scored alone come out in the same order as in a batch
      (catches padding bugs; bf16 rounding may swap near-ties, hence a
      threshold; skipped for listwise, where scores depend on the list);
    - one query against the 100 longest products: time and peak GPU memory.
Prints one line per model and writes nothing. Exits 1 if any check fails.
Run one config per process so a crash or a bad GPU state cannot affect the next.

Usage:
    uv run python scripts/smoke_rerank.py configs/rerank_bge_v2_m3.yaml
"""

import argparse
import itertools
import sys
import time
from pathlib import Path

import torch

from product_search.data.wands import load_wands, product_texts
from product_search.eval.cost import peak_gpu_gb, timed
from product_search.retrieval.rerank import load_rerank_config, load_scorer

TOY_PRODUCTS = {
    "t0": "solid wood dining table",
    "t1": "blue velvet sofa",
    "t2": "king size bed frame",
    "t3": "outdoor patio umbrella",
}
TOY_QUERIES = ["wooden dining table", "blue sofa", "bed frame king", "patio umbrella"]
LONG_PRODUCTS = 100
MIN_AGREEMENT = 0.9


def check(path: Path, texts: dict[str, str]) -> list[str]:
    """Run all checks for one config; return the failed checks (empty if all pass)."""
    name, config = load_rerank_config(path)
    failures = []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    longest = dict(
        sorted(texts.items(), key=lambda item: len(item[1]), reverse=True)[:LONG_PRODUCTS]
    )
    with timed() as loading:
        scorer = load_scorer(config)
    scorer.index(TOY_PRODUCTS | longest)

    toy_ids = list(TOY_PRODUCTS)
    correct = 0
    for i, query in enumerate(TOY_QUERIES):
        scores = scorer.score(query, toy_ids)
        correct += int(max(range(len(scores)), key=scores.__getitem__) == i)

    sample = list(longest)[:8]
    batched = scorer.score(TOY_QUERIES[1], sample)
    alone = [scorer.score(TOY_QUERIES[1], [p])[0] for p in sample]
    pairs = list(itertools.combinations(range(len(sample)), 2))
    same_order = sum((batched[i] - batched[j]) * (alone[i] - alone[j]) > 0 for i, j in pairs)
    agreement = same_order / len(pairs)

    sync = torch.cuda.synchronize if torch.cuda.is_available() else lambda: None
    start = time.perf_counter()
    scorer.score(TOY_QUERIES[0], list(longest))
    sync()
    long_ms = (time.perf_counter() - start) * 1000
    peak = peak_gpu_gb()

    if correct < len(TOY_QUERIES):
        failures.append(f"toy {correct}/{len(TOY_QUERIES)}")
    if config.method != "listwise" and agreement < MIN_AGREEMENT:
        failures.append(f"alone vs batch order {agreement:.2f}")

    peak_text = f"{peak:.1f} GB" if peak is not None else "n/a"
    print(
        f"{name:24s} {'OK  ' if not failures else 'FAIL'} load {loading['seconds']:5.1f}s  "
        f"toy {correct}/{len(TOY_QUERIES)}  alone/batch order {agreement:.2f}  "
        f"100 longest {long_ms:6.0f} ms  peak GPU {peak_text}  {'; '.join(failures)}",
        flush=True,
    )
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test reranker configs.")
    parser.add_argument("configs", type=Path, nargs="+")
    args = parser.parse_args()

    products = load_wands().products
    failed = []
    for path in args.configs:
        _, config = load_rerank_config(path)
        texts = product_texts(products, config.fields)
        try:
            if check(path, texts):
                failed.append(path.name)
        except Exception as error:  # report and move on to the next model
            print(f"{path.stem:24s} ERROR {type(error).__name__}: {error}", flush=True)
            failed.append(path.name)
    if failed:
        print(f"\nFailed: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
