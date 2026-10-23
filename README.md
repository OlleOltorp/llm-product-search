# Product search on a small catalogue

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![Dataset: WANDS](https://img.shields.io/badge/dataset-WANDS-orange)](https://github.com/wayfair/WANDS)
[![Repo size](https://img.shields.io/github/repo-size/OlleOltorp/llm-product-search)](https://github.com/OlleOltorp/llm-product-search)
[![Last commit](https://img.shields.io/github/last-commit/OlleOltorp/llm-product-search)](https://github.com/OlleOltorp/llm-product-search/commits/main)

This was an exploratory study in preparation for a possible master's thesis in search. I tested the leading open-source models from different benchmarks, hybrid setups, and pipelines with rerankers and with query and product expansion. All models were used out of the box, but every setup was tested thoroughly on the key ranking metrics and on latency for production use.

## Results

![Quality against latency, each stage](docs/img/overview.png)

*Best setting at each stage, chosen on dev (hollow) and checked once on test (filled). Figure 2 in [notebook 08](notebooks/08_test.ipynb).*

| Stage | nDCG@10 (test) | Change | p99 latency | Kept |
|---|---|---|---|---|
| BM25 | 0.710 | – | 1.3 ms | baseline |
| + dense (inf-retriever-v1-1.5b) | 0.786 | +7.6 vs BM25 | 28 ms | ✓ |
| + hybrid (BM25 + 2 dense models, RRF) | **0.804** | **+1.9 vs dense** | **37 ms** | ✓ |
| + reranker (zerank-2) | dev only | +0.9 on dev | ~470 ms | ✗ too slow |
| + LLM expansion | 0.808 | +0.3, not significant | 467 ms | ✗ |

- Hybrid search gave the reliable gain: +1.9 nDCG@10 on test, 37 ms.
- Rerankers and LLM expansion each cost ~10× the latency for at most one point, and the expansion gain did not hold on test.
- LLM expansion helps plain BM25 a lot: 151 of 161 combinations still significantly better on test.

*nDCG@10: quality of the top 10 results (0–1). Rcap@100: share of the right products found in the top 100. Change: test-set difference in points against the stage above (paired permutation test).*

## How it was measured

| | |
|---|---|
| Data | [WANDS](https://github.com/wayfair/WANDS) (Wayfair): 42,994 products, 480 real queries, 233,448 human labels |
| Split | 241 dev queries for every choice, 239 test queries used once at the end |
| Rule | Keep a change only if significantly better (p < 0.05; q-values when many settings are compared) |
| Hardware | One RTX 4070 (12 GB), one query at a time. Budget: p99 ≤ 50 ms first stage, ≤ 500 ms in total |

## Steps

### 1. BM25 · [notebook 03](notebooks/03_bm25.ipynb)

- 420 settings tried. Stemming (+4.5 Rcap@100) and the description field (+5.6) matter.
- Tuning k1 and b does not help (+1.6, not significant), so we use the default.

![BM25: Rcap@100 for every k1 and b](docs/img/bm25_k1_b.png)

*Rcap@100 for every k1 and b. Every setting lands between 0.69 and 0.72, so we use the default (k1 0.9, b 0.4). Figure 2 in [notebook 03](notebooks/03_bm25.ipynb).*

### 2. Dense retrieval · [notebook 04](notebooks/04_dense.ipynb)

- 11 off-the-shelf embedding models, 33M to 7B parameters.
- 7 of 11 beat BM25 on nDCG@10, only 3 of 11 on Rcap@100.
- inf-retriever-v1-1.5b is the best model within 50 ms.

![Dense models, quality against latency](docs/img/dense_quality_latency.png)

*Models significantly better than BM25 on at least one metric. Filled: significant on that metric. Figure 4 in [notebook 04](notebooks/04_dense.ipynb).*

### 3. Hybrid · [notebook 05](notebooks/05_fusion.ipynb)

- All 120 combinations of BM25 and 6 dense models, merged with reciprocal rank fusion (RRF).
- No combination finds more right products than inf-1.5B alone, but several rank the top 10 better.
- Pick: BM25 + inf-1.5B + arctic-embed-m, +2.1 nDCG@10 on dev, 37 ms.

![Combinations against inf-1.5B](docs/img/fusion_vs_inf.png)

*The 34 combinations within 50 ms against inf-1.5B alone (dashed). Green: significantly better, red: significantly worse. Figure 1 in [notebook 05](notebooks/05_fusion.ipynb).*

### 4. Rerankers · [notebook 06](notebooks/06_rerankers.ipynb)

- 15 off-the-shelf rerankers on the hybrid's top 10, 20, 50 and 100.
- Only zerank-2 helped: +0.9 at depth 10, at ~470 ms. Not kept.
- The rest got worse the more candidates they reranked.

![Change in nDCG@10 by reranking depth](docs/img/rerank_depth.png)

*Change against the hybrid alone, one line per model. Figure 2 in [notebook 06](notebooks/06_rerankers.ipynb).*

### 5. LLM expansion · [notebook 07](notebooks/07_expansion.ipynb)

- Product side, generated once offline: P1 likely searches, P2 missing keywords, P3 attributes.
- Query side, generated live: Q1 keywords, Q2 attributes, Q3 an imagined product, Q4 keywords after seeing the top 5 results.
- Every combination, added to the text or as extra RRF lists: 1,530 tests. Qwen3-0.6B on vLLM was fast enough; the same model was ~70× faster on vLLM than in transformers.

![Significant wins and losses per system and mode](docs/img/expansion_wins_losses.png)

*Combinations significantly better (right) or worse (left) than the same system without expansion, out of 255 per row. Figure 1 in [notebook 07](notebooks/07_expansion.ipynb).*

### 6. Test set · [notebook 08](notebooks/08_test.ipynb)

- Run once, on everything chosen on dev.
- Held: dense beats BM25, hybrid beats dense, expansion helps BM25.
- Did not hold: expansion's top-10 gain on the hybrid (0 of 14 significant). It raised recall in the top 100 instead (14 of 14).

![Hybrid with expansion, dev against test](docs/img/test_expansion.png)

*The 14 hybrid combinations chosen on dev, from dev (orange) to test (blue). Filled: significant. Figure 1 in [notebook 08](notebooks/08_test.ipynb).*

## Limits

- One dataset, built around its 480 queries.
- Latency for one user; expansion pieces timed separately, not on one shared GPU.
- No fine-tuning.

## Next

- Fine-tune an embedding model or reranker on synthetic queries.
- Move the expansion gain offline, into the product index.
- Check on a larger catalogue (TREC Product Search 2023, 1.66M products).

## Repository

```
src/product_search/   data, retrieval (BM25, dense, fusion, rerank), expansion, evaluation
scripts/              one entry point per experiment
configs/              one YAML file per model or run
notebooks/            01–08, one per step
docs/evaluation.md    evaluation rules
results/              metrics for every run (dev.csv, test.csv)
tests/                pytest
```

## Running it

Python 3.12 and [uv](https://docs.astral.sh/uv/). Experiments ran on Windows with CUDA 13.

```bash
uv sync --extra cu130          # or --extra cpu without a GPU
uv run --extra cu130 python scripts/prepare_data.py
uv run --extra cu130 python scripts/make_split.py
uv run --extra cu130 python scripts/run_bm25.py --config configs/bm25_default.yaml
uv run --extra cu130 python scripts/run_dense.py --config configs/dense_inf_15b.yaml
uv run --extra cu130 python scripts/fuse.py rrf
```

Runs are written to `runs/<name>/`; `scripts/evaluate.py` scores a run and adds it to `results/dev.csv`. Query expansion on vLLM: `scripts/serve_vllm.ps1` (Docker) + `scripts/expand.py`.

## Licence

Code: [MIT](LICENSE). WANDS (MIT, Wayfair) is downloaded by `scripts/prepare_data.py`, not stored here.
