"""Load the WANDS dataset: download, check, clean and cache it as Parquet.

WANDS (Wayfair ANnotation DataSet) has 42,994 products, 480 queries and
233,448 graded query-product labels. See docs/dataset.md.

Typical use:
    data = load_wands()
    products = product_texts(data.products)
    queries = query_texts(data.queries)
    labels = label_grades(data.labels)
"""

import logging
import shutil
import tempfile
import urllib.request
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from product_search.types import Labels, Products, Queries

logger = logging.getLogger(__name__)

# Pinned so every run uses the same data (the only release, 2022-01-18).
WANDS_COMMIT = "3b74dcf4ba29ab8ff3e6a50b5b09fc627cb882b5"
WANDS_URL = f"https://github.com/wayfair/WANDS/archive/{WANDS_COMMIT}.zip"

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[3] / "data"

EXPECTED_RAW_ROWS = {"product.csv": 42_994, "query.csv": 480, "label.csv": 233_448}
TABLES = ("products", "queries", "labels")

GRADES = {"Exact": 2, "Partial": 1, "Irrelevant": 0}

# Product fields that can be indexed, and the default choice.
TEXT_FIELDS = ("name", "categories", "category_path", "description", "features")
DEFAULT_TEXT_FIELDS = ("name", "categories", "category_path", "description")


@dataclass(frozen=True)
class WandsData:
    """Cleaned WANDS tables.

    Attributes:
        products: One row per product: product_id, name, categories (list),
            category_path, description, features, rating_count,
            average_rating, review_count. Missing text is "", missing
            ratings are NaN.
        queries: One row per query: query_id, query, query_class.
            query_class comes from Wayfair's query classifier, which also
            helped pick the labelled products. Use it for analysis only,
            never as input to a method.
        labels: One row per (query, product) pair: query_id, product_id,
            label, grade (Exact 2, Partial 1, Irrelevant 0).
    """

    products: pd.DataFrame
    queries: pd.DataFrame
    labels: pd.DataFrame


def download_wands(raw_dir: Path) -> None:
    """Download the pinned WANDS release into raw_dir, unless it is already there.

    Args:
        raw_dir: Target folder; the CSV files end up in raw_dir/dataset.
    """
    dataset_dir = raw_dir / "dataset"
    if all((dataset_dir / name).exists() for name in EXPECTED_RAW_ROWS):
        logger.info("Raw WANDS found in %s", dataset_dir)
        return

    logger.info("Downloading WANDS %s", WANDS_COMMIT[:7])
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "wands.zip"
        urllib.request.urlretrieve(WANDS_URL, archive)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(tmp)
        shutil.copytree(Path(tmp) / f"WANDS-{WANDS_COMMIT}", raw_dir, dirs_exist_ok=True)


def read_raw(dataset_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read the three raw files, with ids as strings, and check their row counts.

    Args:
        dataset_dir: Folder holding product.csv, query.csv and label.csv.

    Returns:
        Raw products, queries and labels.

    Raises:
        ValueError: If a file does not have the expected number of rows.
    """
    products = pd.read_csv(dataset_dir / "product.csv", sep="\t", dtype={"product_id": str})
    queries = pd.read_csv(dataset_dir / "query.csv", sep="\t", dtype={"query_id": str})
    labels = pd.read_csv(
        dataset_dir / "label.csv", sep="\t", dtype={"query_id": str, "product_id": str}
    )

    for name, df in zip(EXPECTED_RAW_ROWS, (products, queries, labels), strict=True):
        expected = EXPECTED_RAW_ROWS[name]
        if len(df) != expected:
            raise ValueError(
                f"{name}: expected {expected:,} rows, found {len(df):,}. "
                f"Delete {dataset_dir.parent} and run again to re-download."
            )
    return products, queries, labels


def split_categories(product_class: pd.Series) -> pd.Series:
    """Turn 'Sofas|Sectionals' into ['Sofas', 'Sectionals']; missing becomes [].

    Args:
        product_class: Raw product_class column.

    Returns:
        A list of category names per product.
    """
    return product_class.fillna("").map(
        lambda text: [c.strip() for c in text.split("|") if c.strip()]
    )


def deduplicate_labels(labels: pd.DataFrame) -> pd.DataFrame:
    """Keep one label per (query, product) pair: the newest.

    1,467 pairs are labelled more than once, 14 of them with different
    labels. Row ids follow labelling order (the label mix by id matches the
    paper's two rounds), so the highest id is the newest label, made by
    better-calibrated annotators.

    Args:
        labels: Raw labels with an id column.

    Returns:
        Labels with exactly one row per (query_id, product_id).
    """
    return (
        labels.sort_values("id")
        .drop_duplicates(["query_id", "product_id"], keep="last")
        .reset_index(drop=True)
    )


def clean_products(raw: pd.DataFrame) -> pd.DataFrame:
    """Rename columns, split categories and replace missing text with ""."""
    return pd.DataFrame(
        {
            "product_id": raw["product_id"],
            "name": raw["product_name"].fillna(""),
            "categories": split_categories(raw["product_class"]),
            "category_path": raw["category hierarchy"].fillna(""),
            "description": raw["product_description"].fillna(""),
            "features": raw["product_features"].fillna(""),
            "rating_count": raw["rating_count"],
            "average_rating": raw["average_rating"],
            "review_count": raw["review_count"],
        }
    )


def clean_queries(raw: pd.DataFrame) -> pd.DataFrame:
    """Keep query text as is (typos are part of the task); missing class becomes ""."""
    return pd.DataFrame(
        {
            "query_id": raw["query_id"],
            "query": raw["query"],
            "query_class": raw["query_class"].fillna(""),
        }
    )


def clean_labels(raw: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate labels and add a numeric grade.

    Raises:
        ValueError: If a label is not Exact, Partial or Irrelevant.
    """
    labels = deduplicate_labels(raw)
    unknown = set(labels["label"]) - set(GRADES)
    if unknown:
        raise ValueError(f"Unknown label values: {sorted(unknown)}")
    labels["grade"] = labels["label"].map(GRADES).astype(int)
    return labels[["query_id", "product_id", "label", "grade"]]


def validate(data: WandsData) -> None:
    """Check ids are unique and every label points to a known query and product.

    Raises:
        ValueError: If any check fails.
    """
    problems = []
    if data.products["product_id"].duplicated().any():
        problems.append("duplicate product ids")
    if data.queries["query_id"].duplicated().any():
        problems.append("duplicate query ids")
    if data.labels.duplicated(["query_id", "product_id"]).any():
        problems.append("more than one label per (query, product) pair")
    if not data.labels["query_id"].isin(data.queries["query_id"]).all():
        problems.append("labels with unknown query ids")
    if not data.labels["product_id"].isin(data.products["product_id"]).all():
        problems.append("labels with unknown product ids")
    if problems:
        raise ValueError("WANDS checks failed: " + "; ".join(problems))


def prepare_wands(data_dir: Path = DEFAULT_DATA_DIR, force: bool = False) -> None:
    """Download (if needed), clean and save WANDS as Parquet.

    Output goes to data_dir/processed/wands/{products,queries,labels}.parquet.
    Does nothing if those files exist, unless force is set.

    Args:
        data_dir: Root data folder (raw/ and processed/ live under it).
        force: Rebuild the processed files even if they exist.
    """
    out_dir = data_dir / "processed" / "wands"
    if not force and all((out_dir / f"{t}.parquet").exists() for t in TABLES):
        logger.info("Processed WANDS found in %s", out_dir)
        return

    raw_dir = data_dir / "raw" / "WANDS"
    download_wands(raw_dir)
    raw_products, raw_queries, raw_labels = read_raw(raw_dir / "dataset")
    data = WandsData(
        products=clean_products(raw_products),
        queries=clean_queries(raw_queries),
        labels=clean_labels(raw_labels),
    )
    validate(data)

    out_dir.mkdir(parents=True, exist_ok=True)
    for table in TABLES:
        getattr(data, table).to_parquet(out_dir / f"{table}.parquet", index=False)
    logger.info(
        "Saved %s products, %s queries, %s labels (%s repeated labels dropped) to %s",
        f"{len(data.products):,}",
        f"{len(data.queries):,}",
        f"{len(data.labels):,}",
        f"{len(raw_labels) - len(data.labels):,}",
        out_dir,
    )


def load_wands(data_dir: Path = DEFAULT_DATA_DIR) -> WandsData:
    """Load cleaned WANDS, preparing it first if the processed files are missing.

    Args:
        data_dir: Root data folder.

    Returns:
        Cleaned products, queries and labels.
    """
    prepare_wands(data_dir)
    out_dir = data_dir / "processed" / "wands"
    products, queries, labels = (pd.read_parquet(out_dir / f"{t}.parquet") for t in TABLES)
    # Parquet returns list columns as numpy arrays.
    products["categories"] = products["categories"].map(list)
    return WandsData(products=products, queries=queries, labels=labels)


def product_texts(
    products: pd.DataFrame, fields: Sequence[str] = DEFAULT_TEXT_FIELDS
) -> Products:
    """Join the chosen fields into one text per product.

    Fields are joined in the given order, one per line, skipping empty ones.
    Categories are joined with ", ".

    Args:
        products: Cleaned products table.
        fields: Fields to include, from TEXT_FIELDS.

    Returns:
        product_id -> text.

    Raises:
        ValueError: If a field is not in TEXT_FIELDS.
    """
    unknown = set(fields) - set(TEXT_FIELDS)
    if unknown:
        raise ValueError(f"Unknown fields {sorted(unknown)}; choose from {TEXT_FIELDS}")

    columns = [
        products[f].map(", ".join) if f == "categories" else products[f] for f in fields
    ]
    texts = ["\n".join(part for part in parts if part) for parts in zip(*columns, strict=True)]
    return dict(zip(products["product_id"], texts, strict=True))


def query_texts(queries: pd.DataFrame) -> Queries:
    """Return query_id -> query text."""
    return dict(zip(queries["query_id"], queries["query"], strict=True))


def label_grades(labels: pd.DataFrame) -> Labels:
    """Return query_id -> {product_id: grade}, including grade 0 (Irrelevant)."""
    grades: Labels = {}
    for query_id, product_id, grade in zip(
        labels["query_id"], labels["product_id"], labels["grade"], strict=True
    ):
        grades.setdefault(query_id, {})[product_id] = int(grade)
    return grades
