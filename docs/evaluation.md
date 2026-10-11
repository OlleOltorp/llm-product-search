# Evaluation

## Split
- 480 WANDS queries: dev 241, test 239 (`splits/wands.json`, seed 2026).
- Random, balanced by number of Exact matches (0 / 1–10 / 11–100 / >100).
- Near-duplicates stay on one side: same words ignoring plurals, or ≥ 50% shared Exact products (5 pairs).
- All tuning on dev. Test is run once, at the end.

## Relevance
- Grades: Exact 2, Partial 1, Irrelevant 0. Unlabelled products count as not relevant.
- Main level: Exact. Exact + Partial ("any") reported alongside.
- A query with no relevant product at a level is left out of that level's metrics.

## Metrics
| Metric | Measures |
|---|---|
| R@k, k = 10 … 1000 | Share of relevant products in the top k (recall curve) |
| Rcap@k | Same, divided by min(k, #relevant), so a query with more relevant products than k can reach 1 |
| nDCG@10 | Quality of the top 10, graded (gain = grade) |
| MRR@10 | 1 / rank of the first Exact product |

## Uncertainty
- 95% bootstrap interval over queries (10,000 resamples) on every reported number.
- Paired permutation test (10,000 sign flips) for planned comparisons only, not for every grid-search setting.

## Cost (tower only)
- Latency per single query, end to end: p50, p75, p95, p99. 100 warm-up queries, then every dev query 5 times.
- Throughput, indexing time, index size, peak RAM and GPU memory.
- Background apps closed; threads, power plan and package versions recorded per run.

## Runs
`runs/<name>/` (not committed): config, top 1,000 results per query, metadata, metrics overall and per query. Metric means per run go to `results/<split>.csv` (committed).
