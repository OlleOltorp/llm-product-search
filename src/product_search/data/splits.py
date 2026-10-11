"""Split WANDS queries into dev and test.

Simple random split, balanced by difficulty (number of Exact matches), with
near-duplicate queries kept on the same side. The split is saved once to
splits/wands.json and never changed.
"""

import itertools
import json
import random
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from product_search.data.wands import WandsData
from product_search.types import Queries

DEFAULT_SPLIT_PATH = Path(__file__).resolve().parents[3] / "splits" / "wands.json"

# Two queries are near-duplicates if they share at least this share of their
# Exact products (Jaccard). 0.5 finds true rewordings; 0.3 already links
# different needs such as "sugar canister" and "coffee container".
DUPLICATE_EXACT_OVERLAP = 0.5

# Difficulty buckets by number of Exact matches, easiest first.
BUCKETS = ("0", "1-10", "11-100", ">100")


@dataclass(frozen=True)
class Split:
    """A fixed dev/test split of query ids.

    Attributes:
        seed: Random seed used to make the split.
        test_share: Target share of queries in test.
        dev: Dev query ids.
        test: Test query ids.
        duplicate_groups: Near-duplicate groups (two or more queries), each
            kept on one side.
    """

    seed: int
    test_share: float
    dev: list[str]
    test: list[str]
    duplicate_groups: list[list[str]]


def normalise_words(query: str) -> frozenset[str]:
    """Return the query's words, lowercased, without punctuation or a plural 's'."""
    words = re.sub(r"[^a-z0-9 ]", " ", query.lower()).split()
    return frozenset(w[:-1] if w.endswith("s") and len(w) > 3 else w for w in words)


def difficulty_bucket(n_exact: int) -> str:
    """Bucket a query by its number of Exact matches: 0, 1-10, 11-100 or >100."""
    if n_exact == 0:
        return "0"
    if n_exact <= 10:
        return "1-10"
    if n_exact <= 100:
        return "11-100"
    return ">100"


def find_duplicates(
    queries: Queries,
    exact: dict[str, set[str]],
    min_overlap: float = DUPLICATE_EXACT_OVERLAP,
) -> list[tuple[str, str]]:
    """Find near-duplicate query pairs.

    Two queries are near-duplicates if they have the same words (ignoring
    plurals), or share at least min_overlap of their Exact products.

    Args:
        queries: query_id -> query text.
        exact: query_id -> ids of its Exact products (queries without any
            Exact product may be missing).
        min_overlap: Minimum Jaccard overlap of Exact products.

    Returns:
        Pairs of query ids.
    """
    words = {query_id: normalise_words(text) for query_id, text in queries.items()}
    pairs = []
    for a, b in itertools.combinations(sorted(queries), 2):
        if words[a] == words[b]:
            pairs.append((a, b))
            continue
        exact_a, exact_b = exact.get(a, set()), exact.get(b, set())
        if exact_a and exact_b and len(exact_a & exact_b) / len(exact_a | exact_b) >= min_overlap:
            pairs.append((a, b))
    return pairs


def group_queries(query_ids: list[str], pairs: list[tuple[str, str]]) -> list[list[str]]:
    """Merge linked queries into groups; every query ends up in exactly one group.

    Args:
        query_ids: All query ids.
        pairs: Linked pairs; links are transitive (a-b and b-c give a-b-c).

    Returns:
        Groups of query ids, each sorted, in a fixed order.
    """
    parent = {query_id: query_id for query_id in query_ids}

    def root(query_id: str) -> str:
        while parent[query_id] != query_id:
            query_id = parent[query_id]
        return query_id

    for a, b in pairs:
        parent[root(a)] = root(b)

    groups: dict[str, list[str]] = defaultdict(list)
    for query_id in query_ids:
        groups[root(query_id)].append(query_id)
    return sorted(
        (sorted(g, key=_id_order) for g in groups.values()), key=lambda g: _id_order(g[0])
    )


def split_groups(
    groups: list[list[str]], buckets: dict[str, str], test_share: float, seed: int
) -> tuple[list[str], list[str]]:
    """Randomly assign whole groups to dev or test, separately per difficulty bucket.

    Args:
        groups: Query groups; a group is never split.
        buckets: query_id -> difficulty bucket (from BUCKETS). A group takes
            the bucket with the most Exact matches among its queries.
        test_share: Target share of queries in test, per bucket.
        seed: Random seed.

    Returns:
        Sorted dev and test query ids.
    """
    rng = random.Random(seed)
    by_bucket: dict[str, list[list[str]]] = defaultdict(list)
    for group in groups:
        by_bucket[max((buckets[q] for q in group), key=BUCKETS.index)].append(group)

    dev: list[str] = []
    test: list[str] = []
    for bucket in BUCKETS:
        bucket_groups = by_bucket[bucket]
        rng.shuffle(bucket_groups)
        target = round(sum(len(g) for g in bucket_groups) * test_share)
        n_test = 0
        for group in bucket_groups:
            if n_test + len(group) <= target:
                test.extend(group)
                n_test += len(group)
            else:
                dev.extend(group)
    return sorted(dev, key=_id_order), sorted(test, key=_id_order)


def make_split(data: WandsData, seed: int, test_share: float = 0.5) -> Split:
    """Build the dev/test split from cleaned WANDS data.

    Args:
        data: Cleaned WANDS tables.
        seed: Random seed.
        test_share: Target share of queries in test.

    Returns:
        The split.
    """
    queries = dict(zip(data.queries["query_id"], data.queries["query"], strict=True))
    exact_labels = data.labels[data.labels["grade"] == 2]
    exact: dict[str, set[str]] = {}
    for query_id, product_id in zip(
        exact_labels["query_id"], exact_labels["product_id"], strict=True
    ):
        exact.setdefault(query_id, set()).add(product_id)

    groups = group_queries(list(queries), find_duplicates(queries, exact))
    buckets = {q: difficulty_bucket(len(exact.get(q, set()))) for q in queries}
    dev, test = split_groups(groups, buckets, test_share, seed)
    return Split(
        seed=seed,
        test_share=test_share,
        dev=dev,
        test=test,
        duplicate_groups=[g for g in groups if len(g) > 1],
    )


def save_split(split: Split, path: Path = DEFAULT_SPLIT_PATH, force: bool = False) -> None:
    """Write the split as JSON.

    Raises:
        FileExistsError: If the file exists and force is not set. The split
            must not change once results have been reported on it.
    """
    if path.exists() and not force:
        raise FileExistsError(f"{path} exists; the split is fixed. Use force to overwrite.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(split), indent=2) + "\n", encoding="utf-8")


def load_split(path: Path = DEFAULT_SPLIT_PATH) -> Split:
    """Read a split written by save_split."""
    return Split(**json.loads(path.read_text(encoding="utf-8")))


def _id_order(query_id: str) -> tuple[int, str]:
    """Sort numeric ids as numbers ("2" before "10")."""
    return (len(query_id), query_id)
