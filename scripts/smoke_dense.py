"""Quick check that dense configs load and embed sensibly, before a full run.

Per config:
    - loads the pinned model and reports dtype, embedding size, load time;
    - vectors have unit length;
    - a text embedded alone matches the same text in a padded batch
      (catches padding and pooling bugs);
    - four toy queries each rank their obvious product first;
    - one batch of the longest product texts fits in GPU memory.
Prints one line per model and writes nothing. Exits 1 if any check fails.

Usage:
    uv run python scripts/smoke_dense.py configs/dense_*.yaml
"""

import argparse
import gc
import sys
from pathlib import Path

import torch

from product_search.data.wands import load_wands, product_texts
from product_search.eval.cost import peak_gpu_gb, timed
from product_search.retrieval.dense import DenseRetriever, load_dense_config

TOY_PRODUCTS = [
    "solid wood dining table",
    "blue velvet sofa",
    "king size bed frame",
    "outdoor patio umbrella",
]
TOY_QUERIES = ["wooden dining table", "blue sofa", "bed frame king", "patio umbrella"]
MIN_COSINE = 0.999
NORM_TOLERANCE = 1e-3


def check(path: Path, texts: list[str]) -> list[str]:
    """Run all checks for one config; return the failed checks (empty if all pass)."""
    name, config = load_dense_config(path)
    failures = []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    with timed() as loading:
        retriever = DenseRetriever(config)
    dtype = next(retriever.model.parameters()).dtype

    sample = texts[:8]
    batched = retriever.encode_documents(sample)
    alone = torch.cat([retriever.encode_documents([t]) for t in sample])
    min_cosine = (batched * alone).sum(dim=1).min().item()
    norm_error = (batched.norm(dim=1) - 1).abs().max().item()

    scores = retriever.encode_queries(TOY_QUERIES) @ retriever.encode_documents(TOY_PRODUCTS).T
    toy_correct = int((scores.argmax(dim=1).cpu() == torch.arange(len(TOY_QUERIES))).sum())

    longest = sorted(texts, key=len, reverse=True)[: config.batch_size]
    retriever.encode_documents(longest)
    peak = peak_gpu_gb()

    if dtype != getattr(torch, config.dtype):
        failures.append(f"dtype {dtype}")
    if retriever.model.max_seq_length != config.max_seq_length:
        failures.append(f"max_seq_length {retriever.model.max_seq_length}")
    if norm_error > NORM_TOLERANCE:
        failures.append(f"norm off by {norm_error:.4f}")
    if min_cosine < MIN_COSINE:
        failures.append(f"alone vs batch cosine {min_cosine:.4f}")
    if toy_correct < len(TOY_QUERIES):
        failures.append(f"toy {toy_correct}/{len(TOY_QUERIES)}")

    peak_text = f"{peak:.1f} GB" if peak is not None else "n/a"
    print(
        f"{name:20s} {'OK  ' if not failures else 'FAIL'} {str(dtype):15s} "
        f"dim {batched.shape[1]:4d}  load {loading['seconds']:5.1f}s  "
        f"alone/batch {min_cosine:.5f}  toy {toy_correct}/{len(TOY_QUERIES)}  "
        f"peak GPU {peak_text}  {'; '.join(failures)}",
        flush=True,
    )
    del retriever
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test dense configs.")
    parser.add_argument("configs", type=Path, nargs="+")
    args = parser.parse_args()

    products = load_wands().products
    failed = []
    for path in args.configs:
        _, config = load_dense_config(path)
        texts = list(product_texts(products, config.fields).values())
        try:
            if check(path, texts):
                failed.append(path.name)
        except Exception as error:  # report and move on to the next model
            print(f"{path.stem:20s} ERROR {type(error).__name__}: {error}", flush=True)
            failed.append(path.name)
    if failed:
        print(f"\nFailed: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
