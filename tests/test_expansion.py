from pathlib import Path

import pandas as pd
import pytest
import yaml

from product_search.eval.stats import benjamini_hochberg
from product_search.expansion.append import appended_product_texts, appended_query
from product_search.expansion.generate import clean, load_expansion_config
from product_search.expansion.grid import ListBank, subsets
from product_search.expansion.latency import combination_ms
from product_search.expansion.lists import product_combinations, query_combinations
from product_search.retrieval.bm25 import BM25Config, BM25Retriever
from product_search.retrieval.fusion import rrf
from product_search.runs import save_results

CONFIGS = Path(__file__).parents[1] / "configs"


def test_clean_lines_strips_markers_and_duplicates() -> None:
    raw = "1. blue sofa\n- Blue Sofa\n* velvet couch\n\n2) small loveseat"
    assert clean(raw, "lines") == "blue sofa\nvelvet couch\nsmall loveseat"


def test_clean_comma_splits_and_drops_empty() -> None:
    assert (
        clean("couch, settee,, sofa bed.\nloveseat", "comma") == "couch, settee, sofa bed, loveseat"
    )


def test_clean_key_value_keeps_known_values() -> None:
    raw = "type: sofa\ncolour: blue\nmaterial: unknown\nstyle: N/A\nroom: living room\nno colon"
    assert clean(raw, "key_value") == "sofa, blue, living room"


def test_clean_removes_think_block() -> None:
    assert clean("<think>\nhmm\n</think>\ncouch, sofa", "comma") == "couch, sofa"


def test_generator_key_fills_model_and_renames() -> None:
    name, config = load_expansion_config(
        CONFIGS / "expand_q1.yaml", "qwen3_4b", CONFIGS / "generators.yaml"
    )
    assert name == "expand_q1_qwen3_4b"
    assert config.model == "Qwen/Qwen3-4B"


def test_chat_config_without_generator_is_rejected() -> None:
    with pytest.raises(ValueError, match="model and revision"):
        load_expansion_config(CONFIGS / "expand_q1.yaml")


def test_list_bank_matches_rrf(tmp_path: Path) -> None:
    a = {"q1": {"x": 3.0, "y": 2.0, "z": 1.0}, "q2": {"y": 1.0}}
    b = {"q1": {"y": 0.9, "w": 0.8}, "q2": {"x": 0.5, "y": 0.4}}
    for name, run in (("a", a), ("b", b)):
        save_results(run, tmp_path / f"{name}.parquet")
    bank = ListBank(["q1", "q2"], ["w", "x", "y", "z"])
    bank.add("a", tmp_path / "a.parquet")
    bank.add("b", tmp_path / "b.parquet")
    fused = bank.fuse(["a", "b"])
    expected = rrf([a, b])
    for query in ("q1", "q2"):
        assert list(fused[query]) == list(expected[query])
        assert fused[query] == pytest.approx(expected[query])


def test_list_bank_ignores_other_queries(tmp_path: Path) -> None:
    save_results({"q1": {"x": 1.0}, "other": {"y": 1.0}}, tmp_path / "a.parquet")
    bank = ListBank(["q1"], ["x", "y"])
    bank.add("a", tmp_path / "a.parquet")
    assert set(bank.fuse(["a"])) == {"q1"}


def test_subsets_count_and_order() -> None:
    result = list(subsets(["a", "b", "c"]))
    assert result[0] == () and len(result) == 8


def test_expansion_configs_load() -> None:
    for path in CONFIGS.glob("expand_*.yaml"):
        generator = None if "t5" in path.stem else "qwen3_0.6b"
        name, config = load_expansion_config(path, generator, CONFIGS / "generators.yaml")
        assert name.startswith(path.stem)
        if config.target == "queries":
            assert "{query}" in config.prompt
        elif config.kind == "chat":
            assert "{text}" in config.prompt
    assert pd.api.types.is_list_like([])


def test_appended_product_text_puts_expansions_before_description() -> None:
    products = pd.DataFrame(
        {
            "product_id": ["a", "b"],
            "name": ["oak table", "red lamp"],
            "categories": [["tables"], ["lamps"]],
            "category_path": ["", ""],
            "description": ["solid oak", ""],
        }
    )
    fields = ("name", "categories", "category_path", "description")
    texts = appended_product_texts(products, fields, [{"a": "dining table"}, {"a": "", "b": "x"}])
    assert texts == {
        "a": "oak table\ntables\ndining table\nsolid oak",
        "b": "red lamp\nlamps\nx",
    }


def test_appended_query_repeats_raw_and_skips_empty() -> None:
    assert appended_query("sofa", ["couch", "", "settee"], 2) == "sofa sofa\ncouch\nsettee"
    assert appended_query("sofa", [], 1) == "sofa"


def test_bm25_counts_repeated_query_words() -> None:
    # The x5 weighting of the raw query relies on bm25s scoring every repeat.
    retriever = BM25Retriever(BM25Config(stemming=False, stopwords=False))
    retriever.index({"r": "red sofa", "b": "blue chair", "rb": "red blue chair", "g": "green lamp"})
    once = retriever.search({"q": "red blue"}, 3)["q"]
    five = retriever.search({"q": "red red red red red blue"}, 3)["q"]
    assert next(iter(once)) == "rb"
    assert five["r"] > once["r"] * 4


def test_grid_combination_counts() -> None:
    config = yaml.safe_load((CONFIGS / "expansion_grid.yaml").read_text(encoding="utf-8"))
    products = product_combinations(config)
    assert len(products) == 16 and products[0] == ()
    assert len(query_combinations(config)) == 16


def test_list_bank_drop(tmp_path: Path) -> None:
    save_results({"q1": {"x": 1.0}}, tmp_path / "a.parquet")
    bank = ListBank(["q1"], ["x"])
    bank.add("a", tmp_path / "a.parquet")
    bank.drop("a")
    bank.drop("never added")
    assert "a" not in bank


def test_clean_strips_lead_in_lines() -> None:
    raw = "Here are 3 search queries:\n1. blue sofa\n2. velvet couch"
    assert clean(raw, "lines") == "blue sofa\nvelvet couch"
    assert clean("Sure! Keywords: couch, settee", "comma") == "couch, settee"
    assert clean("Keywords:\ncouch, settee", "comma") == "couch, settee"
    assert clean("Here are the attributes:\ntype: sofa\ncolour: blue", "key_value") == "sofa, blue"
    assert clean("Here is a product:\nBlue sofa. Material: velvet", "text") == (
        "Blue sofa. Material: velvet"
    )


def test_clean_keeps_words_that_only_start_like_a_lead_in() -> None:
    assert clean("hereford leather chair\nokapi print rug", "lines") == (
        "hereford leather chair\nokapi print rug"
    )


def test_clean_can_keep_repeats() -> None:
    raw = "blue sofa\nBlue Sofa\nvelvet couch"
    assert clean(raw, "lines", dedupe=False) == raw
    assert clean(raw, "lines") == "blue sofa\nvelvet couch"


def test_t5_configs_keep_repeats() -> None:
    for name in ("expand_p1_t5", "expand_p1_t5all"):
        _, config = load_expansion_config(CONFIGS / f"{name}.yaml")
        assert config.dedupe is False


def test_benjamini_hochberg_known_values() -> None:
    p = pd.Series([0.01, 0.04, 0.03, 0.5], index=list("abcd"))
    q = benjamini_hochberg(p)
    assert list(q.index) == list("abcd")
    assert q.to_list() == pytest.approx([0.04, 0.04 * 4 / 3, 0.04 * 4 / 3, 0.5])


def test_vllm_generator_key_sets_engine() -> None:
    name, config = load_expansion_config(
        CONFIGS / "expand_p2.yaml", "qwen3_8b_awq", CONFIGS / "generators.yaml"
    )
    assert name == "expand_p2_qwen3_8b_awq"
    assert config.engine == "vllm" and config.kind == "chat"


SEARCH = {
    "bm25": {"base": 2.0, "product:p1": 3.0, "q4@g": 4.0, "q3@g": 5.0, "append:q4@g": 6.0},
    "inf15": {"base": 20.0, "q4@g": 21.0, "q3@g": 22.0},
    "arcticm": {"base": 10.0, "q4@g": 11.0, "q3@g": 12.0},
}
LLM = {"q3": 300.0, "q4": 200.0}
HYBRID = ("bm25", "inf15", "arcticm")


def test_latency_without_grounding_runs_llm_alongside_original_searches() -> None:
    # max(LLM 300, originals max(2, 20+10)) + Q3 lists max(5, 22+12)
    assert combination_ms("separate", HYBRID, (), ("q3",), "g", LLM, SEARCH) == 300 + 34


def test_latency_with_grounding_is_search_then_llm_then_search() -> None:
    # first stage max(2, 30) + products (BM25 3 alongside) -> max(2+3, 20+10+20+10)
    # + LLM 300+200 + Q lists max(4+5, 21+11+22+12)
    expected = max(5, 60) + 500 + max(9, 66)
    got = combination_ms("separate", HYBRID, ("p1",), ("q3", "q4"), "g", LLM, SEARCH)
    assert got == expected


def test_latency_products_only_needs_no_llm() -> None:
    # originals + product lists, dense product lists reuse the base timing
    assert combination_ms("separate", HYBRID, ("p1",), (), "g", LLM, SEARCH) == max(5, 60)


def test_latency_append_bm25_grounded() -> None:
    # first stage max(2, 30) + LLM 200 + appended BM25 search 6
    assert combination_ms("append", ("bm25",), ("p1",), ("q4",), "g", LLM, SEARCH) == 236


def test_latency_append_dense_is_not_timed() -> None:
    with pytest.raises(ValueError, match="BM25 only"):
        combination_ms("append", HYBRID, (), ("q4",), "g", LLM, SEARCH)


def test_latency_uses_measured_concurrent_llm_time() -> None:
    together = combination_ms("separate", HYBRID, (), ("q3",), "g", LLM, SEARCH, llm_total=50.0)
    # LLM 50 now shorter than the originals (30): max(50, 30) + 34
    assert together == 50 + 34
