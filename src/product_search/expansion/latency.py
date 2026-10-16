"""Query-time cost of an expansion combination, built from timed pieces.

Expansion adds work at query time: an LLM call per query version, and extra
searches. As for fusion (scripts/fuse.py), the pieces are timed one by one and
added up, here per query so the percentiles describe whole queries:

- BM25 runs on the CPU while the dense models share the GPU, so a set of
  searches takes max(BM25 total, dense total).
- Q4 shows the LLM the first stage's top products, so it needs the hybrid
  first stage before the LLM call (search, then LLM, then search). Without Q4
  the LLM runs alongside the searches that do not need its output.
- Query versions are generated one after another (an upper bound), unless
  the time of sending them at once was measured (llm_total).
- A dense search over a generated-text index costs the same as over the
  original index (same number of products, same raw query), so it reuses that
  timing. BM25 over a generated-text index is timed on its own.
- Append mode searches one appended query per retriever and no original list.

Search timings are keyed by list name:
    base                raw query, original index
    <version>@<gen>     raw + expansion (separate mode), e.g. q4@qwen3_4b
    product:<key>       BM25 over a generated-text index, raw query
    append:<queries>@<gen>  BM25 with the appended query (raw x5 + expansions)
"""

from collections.abc import Mapping, Sequence

FIRST_STAGE_RETRIEVERS = ("bm25", "inf15", "arcticm")
GROUNDED = "q4"


def _searches(
    lists: Sequence[tuple[str, str]], search_ms: Mapping[str, Mapping[str, float]]
) -> float:
    """Time of a set of (retriever, list) searches: BM25 on the CPU, dense on one GPU."""
    bm25 = sum(search_ms[r][name] for r, name in lists if r == "bm25")
    dense = sum(search_ms[r][name] for r, name in lists if r != "bm25")
    return max(bm25, dense)


def combination_ms(
    mode: str,
    members: Sequence[str],
    products: Sequence[str],
    queries: Sequence[str],
    generator: str,
    llm_ms: Mapping[str, float],
    search_ms: Mapping[str, Mapping[str, float]],
    llm_total: float | None = None,
) -> float:
    """Milliseconds for one query through one combination.

    Args:
        mode: "separate" or "append".
        members: Retrievers of the system, e.g. ("bm25", "inf15", "arcticm").
        products: Product expansion keys, e.g. ("p1_qwen",).
        queries: Query versions, e.g. ("q3", "q4").
        generator: Query generator key the timings belong to, e.g. "qwen3_4b".
        llm_ms: Query version -> generation time for this query.
        search_ms: Retriever -> list name -> search time for this query.
        llm_total: Measured time of all versions sent at once; by default the
            versions are added up (one after another).

    Raises:
        ValueError: For append mode with a dense retriever (not timed: no
            such combination was significantly better).
    """
    llm = llm_total if llm_total is not None else sum(llm_ms[v] for v in queries)
    grounded = GROUNDED in queries
    first_stage = [(r, "base") for r in FIRST_STAGE_RETRIEVERS] if grounded else []

    if mode == "append":
        if any(r != "bm25" for r in members):
            raise ValueError("append-mode latency is timed for BM25 only")
        label = "+".join(queries) or "none"
        search = _searches([("bm25", f"append:{label}@{generator}")], search_ms)
        return _searches(first_stage, search_ms) + llm + search

    # Lists that do not need the LLM: the original searches and product indexes.
    originals = [(r, "base") for r in members]
    product_lists = [
        (r, "base" if r != "bm25" else f"product:{p}") for p in products for r in members
    ]
    late = [(r, f"{v}@{generator}") for v in queries for r in members]
    if grounded:
        # Search, then LLM, then search; the first stage covers the members' originals.
        before = first_stage + product_lists
        return _searches(before, search_ms) + llm + _searches(late, search_ms)
    return max(llm, _searches(originals + product_lists, search_ms)) + _searches(late, search_ms)
