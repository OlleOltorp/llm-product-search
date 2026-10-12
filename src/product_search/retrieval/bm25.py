"""BM25 retrieval with bm25s (Lucene variant) and our own word splitting.

Word splitting is ours so the minimum word length is a setting: bm25s's
default drops one-character words such as "l" in "l shaped desk" or "8" in
"8 light". Queries and products always go through the same tokenize().
"""

import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import bm25s
import Stemmer
import yaml
from bm25s.stopwords import STOPWORDS_EN

from product_search.data.wands import DEFAULT_TEXT_FIELDS
from product_search.types import Products, Queries, Results

_STEMMER = Stemmer.Stemmer("english")


@dataclass(frozen=True)
class BM25Config:
    """BM25 settings.

    Attributes:
        fields: Product fields joined into the indexed text.
        stemming: Reduce words to their stem with Snowball English
            ("chairs" -> "chair").
        stopwords: Drop bm25s's 33 English stopwords ("a", "for", "with", ...).
        min_word_length: Shortest word kept; 1 keeps "l" and "8".
        k1: Term frequency saturation.
        b: Length normalisation, 0 (none) to 1 (full).
    """

    fields: tuple[str, ...] = DEFAULT_TEXT_FIELDS
    stemming: bool = True
    stopwords: bool = True
    min_word_length: int = 2
    k1: float = 0.9
    b: float = 0.4


def load_bm25_config(path: Path) -> tuple[str, BM25Config]:
    """Read a run name and BM25 settings from YAML.

    Raises:
        ValueError: If 'name' is missing or a setting is unknown.
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    name = raw.pop("name", None)
    if name is None:
        raise ValueError(f"{path}: missing 'name'")
    unknown = set(raw) - {f.name for f in fields(BM25Config)}
    if unknown:
        raise ValueError(f"{path}: unknown BM25 settings {sorted(unknown)}")
    if "fields" in raw:
        raw["fields"] = tuple(raw["fields"])
    return name, BM25Config(**raw)


def save_bm25_config(path: Path, name: str, config: BM25Config) -> None:
    """Write a run name and BM25 settings as YAML."""
    settings = asdict(config) | {"fields": list(config.fields)}
    path.write_text(yaml.safe_dump({"name": name, **settings}, sort_keys=False), encoding="utf-8")


def tokenize(
    texts: Sequence[str], stemming: bool, stopwords: bool, min_word_length: int
) -> list[list[str]]:
    """Split texts into lowercase words, optionally dropping stopwords and stemming.

    Args:
        texts: Texts to split.
        stemming: Apply the Snowball English stemmer.
        stopwords: Drop English stopwords (before stemming).
        min_word_length: Shortest word kept, in characters.

    Returns:
        One list of words per text.
    """
    pattern = re.compile(rf"\b\w{{{min_word_length},}}\b")
    drop = set(STOPWORDS_EN) if stopwords else set()
    tokens = []
    for text in texts:
        words = [w for w in pattern.findall(text.lower()) if w not in drop]
        tokens.append(_STEMMER.stemWords(words) if stemming else words)
    return tokens


class BM25Retriever:
    """BM25 over product texts.

    index() and search() take raw text. index_tokens() and search_tokens()
    take already split words, so tuning can split once and try many k1/b.
    """

    def __init__(self, config: BM25Config) -> None:
        self.config = config
        self._model: bm25s.BM25 | None = None
        self._product_ids: list[str] = []

    def _tokenize(self, texts: Sequence[str]) -> list[list[str]]:
        c = self.config
        return tokenize(texts, c.stemming, c.stopwords, c.min_word_length)

    def index(self, products: Products) -> None:
        """Build the index from product_id -> text."""
        product_ids = list(products)
        self.index_tokens(product_ids, self._tokenize([products[p] for p in product_ids]))

    def index_tokens(self, product_ids: list[str], tokens: list[list[str]]) -> None:
        """Build the index from already split product texts."""
        self._product_ids = product_ids
        self._model = bm25s.BM25(k1=self.config.k1, b=self.config.b, method="lucene")
        self._model.index(tokens, show_progress=False)

    def save(self, path: Path) -> None:
        """Write the index to a folder (used to measure its size on disk)."""
        if self._model is None:
            raise RuntimeError("Call index() before save()")
        self._model.save(str(path))

    def search(self, queries: Queries, k: int) -> Results:
        """Return the top k products per query; products with no matching word are left out."""
        query_ids = list(queries)
        return self.search_tokens(query_ids, self._tokenize([queries[q] for q in query_ids]), k)

    def search_tokens(self, query_ids: list[str], tokens: list[list[str]], k: int) -> Results:
        """Search with already split queries."""
        if self._model is None:
            raise RuntimeError("Call index() before search()")
        k = min(k, len(self._product_ids))
        documents, scores = self._model.retrieve(tokens, k=k, show_progress=False)
        results: Results = {}
        for query_id, doc_row, score_row in zip(query_ids, documents, scores, strict=True):
            # Score 0 means no query word matched; their order would be arbitrary.
            results[query_id] = {
                self._product_ids[d]: float(s)
                for d, s in zip(doc_row, score_row, strict=True)
                if s > 0
            }
        return results
