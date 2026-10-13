import pytest

from product_search.retrieval.fusion import min_max, rrf, weighted_sum


def test_rrf_rewards_products_high_in_both_lists() -> None:
    a = {"q": {"x": 3.0, "y": 2.0, "z": 1.0}}
    b = {"q": {"y": 0.9, "x": 0.8}}
    fused = rrf([a, b], k=60)
    assert list(fused["q"]) == ["x", "y", "z"]
    assert fused["q"]["x"] == pytest.approx(1 / 61 + 1 / 62)
    assert fused["q"]["z"] == pytest.approx(1 / 63)


def test_rrf_keeps_queries_missing_from_one_run() -> None:
    fused = rrf([{"q1": {"x": 1.0}}, {"q2": {"y": 1.0}}])
    assert set(fused) == {"q1", "q2"}


def test_rrf_cuts_to_depth() -> None:
    run = {"q": {str(i): float(i) for i in range(10)}}
    assert len(rrf([run, run], depth=3)["q"]) == 3


def test_min_max_scales_per_query() -> None:
    scaled = min_max({"q": {"x": 10.0, "y": 5.0, "z": 0.0}, "r": {"x": 2.0, "y": 2.0}})
    assert scaled["q"] == {"x": 1.0, "y": 0.5, "z": 0.0}
    assert scaled["r"] == {"x": 1.0, "y": 1.0}


def test_weighted_sum_uses_weights() -> None:
    a = {"q": {"x": 1.0, "y": 0.0}}
    b = {"q": {"x": 0.0, "y": 1.0}}
    assert list(weighted_sum([a, b], [0.7, 0.3])["q"]) == ["x", "y"]
    assert list(weighted_sum([a, b], [0.3, 0.7])["q"]) == ["y", "x"]


def test_weighted_sum_needs_one_weight_per_run() -> None:
    with pytest.raises(ValueError, match="weights"):
        weighted_sum([{"q": {}}], [0.5, 0.5])
