"""Append mode: generated text is written into the product text and the query.

As in docTTTTTquery (products) and query2doc (queries), the retriever sees
one blended text instead of extra lists.

Products: the generated text goes before the description, so the 512-token
cut of the dense models drops the end of a long description, not the
expansion. BM25 does not depend on the order.

Queries: BM25 gets the raw query five times, then the expansion, so the
expansion cannot outweigh the user's words (query2doc, Wang et al. 2023);
dense models get the raw query, then the expansion.
"""

from collections.abc import Mapping, Sequence

import pandas as pd

from product_search.data.wands import product_texts
from product_search.types import Products

BM25_QUERY_REPEATS = 5
DENSE_QUERY_REPEATS = 1


def appended_product_texts(
    products: pd.DataFrame, fields: Sequence[str], extras: Sequence[Mapping[str, str]]
) -> Products:
    """Product texts with generated texts placed before the description.

    Args:
        products: Cleaned products table.
        fields: Fields of the original text, in order.
        extras: One product_id -> generated text mapping per expansion;
            products missing from a mapping get nothing from it.

    Returns:
        product_id -> text.
    """
    fields = list(fields)
    tail = ["description"] if fields and fields[-1] == "description" else []
    head = product_texts(products, fields[: len(fields) - len(tail)])
    end = product_texts(products, tail) if tail else {}
    texts = {}
    for product_id, text in head.items():
        parts = [text, *(extra.get(product_id, "") for extra in extras), end.get(product_id, "")]
        texts[product_id] = "\n".join(part for part in parts if part)
    return texts


def appended_query(raw: str, extras: Sequence[str], repeats: int) -> str:
    """The raw query repeated, then each non-empty expansion on its own line."""
    return "\n".join([" ".join([raw] * repeats), *(e for e in extras if e)])
