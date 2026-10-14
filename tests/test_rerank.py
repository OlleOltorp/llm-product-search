from pathlib import Path

import pytest

from product_search.retrieval.rerank import load_rerank_config, rerank
from product_search.types import Products


class LengthScorer:
    """Scores a product by the length of its text."""

    def __init__(self) -> None:
        self.texts: Products = {}

    def index(self, products: Products) -> None:
        self.texts = products

    def score(self, query: str, product_ids: list[str]) -> list[float]:
        return [float(len(self.texts[p])) for p in product_ids]


def test_rerank_reorders_head_and_keeps_tail_order() -> None:
    scorer = LengthScorer()
    scorer.index({"a": "x", "b": "xxx", "c": "xx", "d": "xxxx"})
    first_stage = {"q": {"a": 4.0, "b": 3.0, "c": 2.0, "d": 1.0}}
    results, latencies = rerank(scorer, {"q": "query"}, first_stage, depth=2)
    order = sorted(results["q"], key=lambda p: -results["q"][p])
    assert order == ["b", "a", "c", "d"]
    assert len(latencies) == 1


def test_rerank_depth_beyond_list_reranks_everything() -> None:
    scorer = LengthScorer()
    scorer.index({"a": "x", "b": "xx"})
    results, _ = rerank(scorer, {"q": "query"}, {"q": {"a": 2.0, "b": 1.0}}, depth=100)
    assert max(results["q"], key=results["q"].__getitem__) == "b"


def test_config_rejects_unknown_method(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("name: x\nmodel: m\nrevision: r\nmethod: magic\n", encoding="utf-8")
    with pytest.raises(ValueError, match="method"):
        load_rerank_config(path)


def test_configs_in_repo_load() -> None:
    for path in Path(__file__).parents[1].glob("configs/rerank_*.yaml"):
        name, config = load_rerank_config(path)
        assert name == path.stem
        assert config.depth == 100
