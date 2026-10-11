import pandas as pd
import pytest

from product_search.data.wands import (
    deduplicate_labels,
    label_grades,
    product_texts,
    split_categories,
)


def test_deduplicate_keeps_newest_label() -> None:
    labels = pd.DataFrame(
        {
            "id": [21650, 65, 7],
            "query_id": ["0", "0", "1"],
            "product_id": ["31306", "31306", "5"],
            "label": ["Partial", "Irrelevant", "Exact"],
        }
    )
    result = deduplicate_labels(labels)
    assert len(result) == 2
    assert result.set_index("query_id").loc["0", "label"] == "Partial"


def test_split_categories() -> None:
    raw = pd.Series(["Sofas|Sectionals", "Beds", None])
    assert split_categories(raw).tolist() == [["Sofas", "Sectionals"], ["Beds"], []]


def test_product_texts_joins_fields_and_skips_empty() -> None:
    products = pd.DataFrame(
        {
            "product_id": ["1", "2"],
            "name": ["velvet sofa", "oak bed"],
            "categories": [["Sofas", "Sectionals"], []],
            "description": ["soft", ""],
        }
    )
    texts = product_texts(products, fields=["name", "categories", "description"])
    assert texts == {"1": "velvet sofa\nSofas, Sectionals\nsoft", "2": "oak bed"}


def test_product_texts_rejects_unknown_field() -> None:
    products = pd.DataFrame({"product_id": ["1"], "name": ["sofa"]})
    with pytest.raises(ValueError, match="Unknown fields"):
        product_texts(products, fields=["name", "colour"])


def test_label_grades() -> None:
    labels = pd.DataFrame(
        {"query_id": ["0", "0", "1"], "product_id": ["a", "b", "a"], "grade": [2, 0, 1]}
    )
    assert label_grades(labels) == {"0": {"a": 2, "b": 0}, "1": {"a": 1}}
