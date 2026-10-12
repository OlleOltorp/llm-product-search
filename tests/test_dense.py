from pathlib import Path

import pytest
import torch

from product_search.retrieval.dense import (
    DenseConfig,
    load_dense_config,
    save_dense_config,
    top_k,
)

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"


def test_top_k_ranks_by_dot_product() -> None:
    queries = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    products = torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.6, 0.8]])
    results = top_k(["q1", "q2"], queries, ["a", "b", "c"], products, k=2)
    assert list(results["q1"]) == ["b", "c"]
    assert list(results["q2"]) == ["a", "c"]
    assert results["q1"]["c"] == pytest.approx(0.6)


def test_top_k_caps_k_at_catalogue_size() -> None:
    results = top_k(["q"], torch.ones(1, 2), ["a", "b"], torch.ones(2, 2), k=1000)
    assert len(results["q"]) == 2


def test_config_round_trip(tmp_path: Path) -> None:
    config = DenseConfig(
        model="org/model",
        revision="abc123",
        query_prompt="Instruct: find\nQuery:",
        config_kwargs={"unpad_inputs": False},
    )
    path = tmp_path / "dense.yaml"
    save_dense_config(path, "dense_test", config)
    assert load_dense_config(path) == ("dense_test", config)


def test_config_rejects_unknown_setting(tmp_path: Path) -> None:
    path = tmp_path / "dense.yaml"
    path.write_text("name: x\nmodel: m\nrevision: r\npooling: mean\n", encoding="utf-8")
    with pytest.raises(ValueError, match="pooling"):
        load_dense_config(path)


def test_config_rejects_unknown_dtype(tmp_path: Path) -> None:
    path = tmp_path / "dense.yaml"
    path.write_text("name: x\nmodel: m\nrevision: r\ndtype: float16\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dtype"):
        load_dense_config(path)


@pytest.mark.parametrize("path", sorted(CONFIGS_DIR.glob("dense_*.yaml")), ids=lambda p: p.stem)
def test_repo_configs_load(path: Path) -> None:
    name, config = load_dense_config(path)
    assert name == path.stem
    assert len(config.revision) == 40
