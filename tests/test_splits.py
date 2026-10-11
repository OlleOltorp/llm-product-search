from product_search.data.splits import (
    difficulty_bucket,
    find_duplicates,
    group_queries,
    normalise_words,
    split_groups,
)


def test_normalise_words_ignores_case_punctuation_and_plurals() -> None:
    assert normalise_words("Leather Chairs!") == normalise_words("leather chair")


def test_difficulty_bucket_edges() -> None:
    assert [difficulty_bucket(n) for n in (0, 1, 10, 11, 100, 101)] == [
        "0",
        "1-10",
        "1-10",
        "11-100",
        "11-100",
        ">100",
    ]


def test_find_duplicates_by_words_and_by_shared_exact_products() -> None:
    queries = {
        "1": "leather chairs",
        "2": "leather chair",
        "3": "l shaped desk",
        "4": "l shape desk",
    }
    exact = {"3": {"a", "b", "c"}, "4": {"a", "b", "d"}}
    assert find_duplicates(queries, exact, min_overlap=0.5) == [("1", "2"), ("3", "4")]
    assert find_duplicates(queries, exact, min_overlap=0.6) == [("1", "2")]


def test_group_queries_is_transitive() -> None:
    groups = group_queries(["1", "2", "3", "4"], [("1", "2"), ("2", "3")])
    assert groups == [["1", "2", "3"], ["4"]]


def test_split_keeps_groups_together_and_balances_buckets() -> None:
    groups = [["1", "2"]] + [[str(i)] for i in range(3, 21)]
    buckets = {str(i): ("0" if i <= 10 else ">100") for i in range(1, 21)}
    dev, test = split_groups(groups, buckets, test_share=0.5, seed=0)

    assert sorted(dev + test, key=int) == [str(i) for i in range(1, 21)]
    assert ("1" in dev) == ("2" in dev)
    for bucket in ("0", ">100"):
        in_bucket = [q for q in test if buckets[q] == bucket]
        assert len(in_bucket) == 5


def test_split_is_reproducible() -> None:
    groups = [[str(i)] for i in range(50)]
    buckets = {str(i): "0" for i in range(50)}
    assert split_groups(groups, buckets, 0.5, seed=7) == split_groups(groups, buckets, 0.5, seed=7)
    assert split_groups(groups, buckets, 0.5, seed=7) != split_groups(groups, buckets, 0.5, seed=8)
