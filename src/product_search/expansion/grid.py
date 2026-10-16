"""Fast reciprocal rank fusion over many saved lists, for the expansion grid.

The grid fuses hundreds of list combinations. Each saved list is held as
arrays of (cell in a queries x products table, 1 / (k + rank)); a
combination's scores are those arrays added into the table, and the top
products per query come from argpartition. Ties are broken by product id, as in
retrieval.fusion.rrf.
"""

import itertools
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from product_search.retrieval.fusion import RRF_K
from product_search.types import Results


class ListBank:
    """Saved result lists for a fixed set of queries, ready to be fused."""

    def __init__(self, query_ids: Sequence[str], product_ids: Sequence[str], k: int = RRF_K):
        self.query_ids = list(query_ids)
        # Sorted, so column order is product id order and ties break by id.
        self.product_ids = sorted(product_ids)
        self._query_row = {q: i for i, q in enumerate(self.query_ids)}
        self._product_col = {p: j for j, p in enumerate(self.product_ids)}
        self.k = k
        self._lists: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def add(self, name: str, path: Path) -> None:
        """Load a results.parquet (query_id, product_id, rank, score) under a name."""
        table = pd.read_parquet(path, columns=["query_id", "product_id", "rank"])
        table = table[table["query_id"].isin(self._query_row)]
        cells = (
            table["query_id"].map(self._query_row).to_numpy() * len(self.product_ids)
            + table["product_id"].map(self._product_col).to_numpy()
        )
        self._lists[name] = (cells, 1.0 / (self.k + table["rank"].to_numpy()))

    def drop(self, name: str) -> None:
        """Forget a list (no error if it was never added)."""
        self._lists.pop(name, None)

    def __contains__(self, name: str) -> bool:
        return name in self._lists

    def fuse(self, names: Sequence[str], depth: int = 1000) -> Results:
        """RRF of the named lists; top depth products per query."""
        size = len(self.query_ids) * len(self.product_ids)
        scores = np.zeros(size)
        for name in names:
            cells, contributions = self._lists[name]
            # A product appears once per query in a list, so the cells are unique.
            scores[cells] += contributions
        table = scores.reshape(len(self.query_ids), len(self.product_ids))
        depth = min(depth, len(self.product_ids))
        top = np.argpartition(-table, depth - 1, axis=1)[:, :depth]
        results: Results = {}
        for row, query_id in enumerate(self.query_ids):
            columns = top[row][table[row, top[row]] > 0]
            # Highest score first, then lowest column (= product id) on ties.
            order = np.lexsort((columns, -table[row, columns]))
            results[query_id] = {self.product_ids[c]: float(table[row, c]) for c in columns[order]}
        return results


def subsets(items: Sequence[str]) -> Iterator[tuple[str, ...]]:
    """Every subset of items, the empty one first."""
    for size in range(len(items) + 1):
        yield from itertools.combinations(items, size)
