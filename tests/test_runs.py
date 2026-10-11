from pathlib import Path

import pandas as pd
import pytest

from product_search.runs import create_run_dir, load_results, save_results, update_results_table


def test_save_and_load_results_keeps_top_depth(tmp_path: Path) -> None:
    results = {"q1": {"a": 0.9, "b": 0.5, "c": 0.1}, "q2": {"d": 1.0}}
    save_results(results, tmp_path / "results.parquet", depth=2)
    assert load_results(tmp_path / "results.parquet") == {
        "q1": {"a": 0.9, "b": 0.5},
        "q2": {"d": 1.0},
    }


def test_create_run_dir_refuses_to_overwrite(tmp_path: Path) -> None:
    create_run_dir("bm25", runs_dir=tmp_path)
    with pytest.raises(FileExistsError):
        create_run_dir("bm25", runs_dir=tmp_path)


def test_update_results_table_replaces_existing_row(tmp_path: Path) -> None:
    path = tmp_path / "dev.csv"
    update_results_table(path, "bm25", {"R@100": {"mean": 0.5}})
    update_results_table(path, "dense", {"R@100": {"mean": 0.6}})
    update_results_table(path, "bm25", {"R@100": {"mean": 0.55}})
    table = pd.read_csv(path)
    assert table["run"].tolist() == ["dense", "bm25"]
    assert table.set_index("run").loc["bm25", "R@100"] == 0.55
