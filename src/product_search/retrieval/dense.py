"""Dense retrieval: embed queries and products, rank by cosine similarity.

Models run through sentence-transformers, pinned to a Hugging Face revision.
Prompts are written out in the config instead of taken from the model repo,
so a run records exactly what text the model saw. Search is exact: every
query is scored against every product on the GPU.
"""

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from sentence_transformers import SentenceTransformer

from product_search.data.wands import DEFAULT_TEXT_FIELDS
from product_search.types import Products, Queries, Results

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}
QUERY_BATCH_SIZE = 64


@dataclass(frozen=True)
class DenseConfig:
    """Dense model settings.

    Attributes:
        model: Hugging Face model id.
        revision: Commit of the model repo, so reruns use the same weights.
        fields: Product fields joined into the embedded text.
        query_prompt: Text put before every query, e.g. "query: ".
        document_prompt: Text put before every product text.
        max_seq_length: Longest input in tokens; longer texts are cut.
        dtype: "float32" or "bfloat16".
        batch_size: Product texts per encoding batch.
        trust_remote_code: Run model code from the model repo (pinned by revision).
        config_kwargs: Overrides for the model's config.json.
        model_kwargs: Extra loading options, e.g. device_map and max_memory to
            keep part of a model too large for the GPU in system RAM.
    """

    model: str
    revision: str
    fields: tuple[str, ...] = DEFAULT_TEXT_FIELDS
    query_prompt: str = ""
    document_prompt: str = ""
    max_seq_length: int = 512
    dtype: str = "float32"
    batch_size: int = 32
    trust_remote_code: bool = False
    config_kwargs: dict[str, Any] = field(default_factory=dict)
    model_kwargs: dict[str, Any] = field(default_factory=dict)


def load_dense_config(path: Path) -> tuple[str, DenseConfig]:
    """Read a run name and dense settings from YAML.

    Raises:
        ValueError: If 'name' is missing, a setting is unknown or dtype is unsupported.
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    name = raw.pop("name", None)
    if name is None:
        raise ValueError(f"{path}: missing 'name'")
    unknown = set(raw) - {f.name for f in fields(DenseConfig)}
    if unknown:
        raise ValueError(f"{path}: unknown dense settings {sorted(unknown)}")
    if raw.get("dtype", "float32") not in DTYPES:
        raise ValueError(f"{path}: dtype must be one of {sorted(DTYPES)}")
    if "fields" in raw:
        raw["fields"] = tuple(raw["fields"])
    return name, DenseConfig(**raw)


def save_dense_config(path: Path, name: str, config: DenseConfig) -> None:
    """Write a run name and dense settings as YAML."""
    settings = asdict(config) | {"fields": list(config.fields)}
    path.write_text(yaml.safe_dump({"name": name, **settings}, sort_keys=False), encoding="utf-8")


def load_model(config: DenseConfig, device: str) -> SentenceTransformer:
    """Load the pinned model in the configured dtype with the length cap applied."""
    model_kwargs = {"dtype": DTYPES[config.dtype]} | config.model_kwargs
    model = SentenceTransformer(
        config.model,
        revision=config.revision,
        # With a device_map, accelerate places the layers itself.
        device=None if "device_map" in model_kwargs else device,
        trust_remote_code=config.trust_remote_code,
        model_kwargs=model_kwargs,
        config_kwargs=config.config_kwargs or None,
    )
    model.max_seq_length = config.max_seq_length
    return model


def top_k(
    query_ids: list[str],
    query_embeddings: torch.Tensor,
    product_ids: list[str],
    product_embeddings: torch.Tensor,
    k: int,
) -> Results:
    """Top k products per query by dot product (cosine for unit vectors)."""
    k = min(k, len(product_ids))
    scores, indices = torch.topk(query_embeddings @ product_embeddings.T, k, dim=1)
    results: Results = {}
    for query_id, score_row, index_row in zip(
        query_ids, scores.tolist(), indices.tolist(), strict=True
    ):
        results[query_id] = {
            product_ids[i]: float(s) for i, s in zip(index_row, score_row, strict=True)
        }
    return results


class DenseRetriever:
    """Exact cosine search over product embeddings."""

    def __init__(self, config: DenseConfig, device: str | None = None) -> None:
        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_model(config, self.device)
        self._embeddings: torch.Tensor | None = None
        self._product_ids: list[str] = []

    def encode(self, texts: list[str], prompt: str, batch_size: int) -> torch.Tensor:
        """Embed texts as unit-length float32 vectors on the device."""
        embeddings = self.model.encode(
            texts,
            prompt=prompt,
            batch_size=batch_size,
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        # Normalise in float32 so bfloat16 models give exact unit vectors.
        return torch.nn.functional.normalize(embeddings.float(), dim=-1)

    def encode_queries(self, texts: list[str]) -> torch.Tensor:
        """Embed query texts with the query prompt."""
        return self.encode(texts, self.config.query_prompt, QUERY_BATCH_SIZE)

    def encode_documents(self, texts: list[str]) -> torch.Tensor:
        """Embed product texts with the document prompt."""
        return self.encode(texts, self.config.document_prompt, self.config.batch_size)

    def index(self, products: Products) -> None:
        """Embed all products from product_id -> text."""
        self._product_ids = list(products)
        self._embeddings = self.encode_documents([products[p] for p in self._product_ids])

    def save(self, path: Path) -> None:
        """Write the product embeddings as .npy (used to measure index size)."""
        if self._embeddings is None:
            raise RuntimeError("Call index() before save()")
        np.save(path / "embeddings.npy", self._embeddings.cpu().numpy())

    def load(self, path: Path, product_ids: list[str]) -> None:
        """Read embeddings written by save(), in the order of product_ids."""
        embeddings = torch.from_numpy(np.load(path / "embeddings.npy")).to(self.device)
        if len(embeddings) != len(product_ids):
            raise ValueError(f"{path}: {len(embeddings)} embeddings for {len(product_ids)} ids")
        self._product_ids = list(product_ids)
        self._embeddings = embeddings

    def search(self, queries: Queries, k: int) -> Results:
        """Return the top k products per query."""
        if self._embeddings is None:
            raise RuntimeError("Call index() before search()")
        query_ids = list(queries)
        query_embeddings = self.encode_queries([queries[q] for q in query_ids])
        return top_k(query_ids, query_embeddings, self._product_ids, self._embeddings, k)
