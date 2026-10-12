from pathlib import Path

from product_search.retrieval.bm25 import (
    BM25Config,
    BM25Retriever,
    load_bm25_config,
    save_bm25_config,
    tokenize,
)


def test_config_round_trip(tmp_path: Path) -> None:
    config = BM25Config(fields=("name", "categories"), stemming=False, k1=1.2, b=0.6)
    save_bm25_config(tmp_path / "bm25.yaml", "bm25_test", config)
    assert load_bm25_config(tmp_path / "bm25.yaml") == ("bm25_test", config)


def test_tokenize_min_length_stopwords_and_stemming() -> None:
    text = ["L Shaped Desk for Kids"]
    assert tokenize(text, stemming=False, stopwords=False, min_word_length=2) == [
        ["shaped", "desk", "for", "kids"]
    ]
    assert tokenize(text, stemming=False, stopwords=True, min_word_length=1) == [
        ["l", "shaped", "desk", "kids"]
    ]
    assert tokenize(text, stemming=True, stopwords=True, min_word_length=1) == [
        ["l", "shape", "desk", "kid"]
    ]


def test_bm25_ranks_the_obvious_match_first_and_drops_non_matches() -> None:
    products = {
        "1": "oak dining table",
        "2": "green velvet chaise lounge",
        "3": "velvet cushion",
        "4": "metal bar stool",
    }
    retriever = BM25Retriever(BM25Config(min_word_length=1))
    retriever.index(products)
    results = retriever.search({"q": "velvet chaise"}, k=10)

    ranking = sorted(results["q"], key=lambda p: -results["q"][p])
    assert ranking[0] == "2"
    assert set(results["q"]) == {"2", "3"}  # products without a matching word are left out


def test_search_handles_queries_with_no_known_words() -> None:
    retriever = BM25Retriever(BM25Config())
    retriever.index({"1": "oak table"})
    assert retriever.search({"q": "zzz"}, k=5) == {"q": {}}
