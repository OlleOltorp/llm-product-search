"""Reranking: score the first stage's top candidates again with a stronger model.

The first stage gives each query a ranked list. A reranker scores the top
`depth` candidates again and re-sorts them; candidates below the depth keep
their first-stage order underneath, so deeper metrics stay comparable.

Methods (config `method`):
    cross_encoder     query and product read together; a classification head scores the pair
    yes_no            a language model is asked whether the product fits; score = yes - no logit
    listwise          the model reads the query and all candidates at once
    late_interaction  one vector per token; each query token takes its best product token (ColBERT)
    bi_encoder        query and product embedded separately; cosine similarity

Scorers that store product encodings (bi_encoder, late_interaction) encode
the candidate products in index(), before any query: in a real system this
happens offline, so it is not part of query latency.
"""

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Protocol

import torch
import yaml

from product_search.data.wands import DEFAULT_TEXT_FIELDS
from product_search.eval.metrics import rank
from product_search.retrieval.dense import DTYPES, DenseRetriever, load_dense_config
from product_search.types import Products, Queries, Results

METHODS = ("cross_encoder", "yes_no", "listwise", "late_interaction", "bi_encoder")
FIRST_STAGE = ("bm25_default", "dense_inf_15b", "dense_arctic_m")


@dataclass(frozen=True)
class RerankConfig:
    """Reranker settings.

    Attributes:
        model: Hugging Face model id (unused for bi_encoder, see dense_config).
        revision: Commit of the model repo, so reruns use the same weights.
        method: One of METHODS.
        depth: Candidates reranked per query.
        first_stage: Saved runs fused with RRF to give the candidates.
        fields: Product fields joined into the product text.
        max_length: Longest input in tokens (query, product and prompt together).
        dtype: "float32" or "bfloat16".
        batch_size: Query-product pairs per forward pass.
        trust_remote_code: Run model code from the model repo (pinned by revision).
        model_kwargs: Extra loading options, e.g. an 8-bit quantization_config.
        instruction: Task description, put into the yes_no prompt or passed
            to a cross_encoder as its prompt.
        prefix: yes_no prompt text before the instruction, query and product.
        body: yes_no prompt with {instruction}, {query} and {document}; the
            only part cut when the input is too long.
        suffix: yes_no prompt text after the body; the model's next token is read.
        yes_token: Token whose logit, minus the no_token logit, is the score.
        no_token: Token it is compared with.
        dense_config: Dense YAML config used by bi_encoder.
    """

    model: str
    revision: str
    method: str
    depth: int = 100
    first_stage: tuple[str, ...] = FIRST_STAGE
    fields: tuple[str, ...] = DEFAULT_TEXT_FIELDS
    max_length: int = 512
    dtype: str = "float32"
    batch_size: int = 16
    trust_remote_code: bool = False
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    instruction: str = ""
    prefix: str = ""
    body: str = ""
    suffix: str = ""
    yes_token: str = "yes"
    no_token: str = "no"
    dense_config: str = ""


def load_rerank_config(path: Path) -> tuple[str, RerankConfig]:
    """Read a run name and reranker settings from YAML.

    Raises:
        ValueError: If 'name' is missing, a setting is unknown, or method or
            dtype is unsupported.
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    name = raw.pop("name", None)
    if name is None:
        raise ValueError(f"{path}: missing 'name'")
    unknown = set(raw) - {f.name for f in fields(RerankConfig)}
    if unknown:
        raise ValueError(f"{path}: unknown rerank settings {sorted(unknown)}")
    if raw.get("method") not in METHODS:
        raise ValueError(f"{path}: method must be one of {METHODS}")
    if raw.get("dtype", "float32") not in DTYPES:
        raise ValueError(f"{path}: dtype must be one of {sorted(DTYPES)}")
    for key in ("fields", "first_stage"):
        if key in raw:
            raw[key] = tuple(raw[key])
    return name, RerankConfig(**raw)


def save_rerank_config(path: Path, name: str, config: RerankConfig) -> None:
    """Write a run name and reranker settings as YAML."""
    settings = asdict(config) | {
        "fields": list(config.fields),
        "first_stage": list(config.first_stage),
    }
    path.write_text(yaml.safe_dump({"name": name, **settings}, sort_keys=False), encoding="utf-8")


class Scorer(Protocol):
    """Scores products for one query; higher is more relevant."""

    def index(self, products: Products) -> None:
        """Store (and, if the method allows, encode) the products that may be scored."""

    def score(self, query: str, product_ids: list[str]) -> list[float]:
        """One score per product id, in the same order."""


class CrossEncoderScorer:
    """Query and product in one input to a sentence-transformers CrossEncoder.

    Also used for LLM rerankers that sentence-transformers loads natively
    (zerank-2, mxbai v2, Lychee): their prompt template then comes from the
    model repo, pinned by revision. A non-empty instruction is passed as the prompt.
    """

    def __init__(self, config: RerankConfig, device: str) -> None:
        from sentence_transformers import CrossEncoder

        self.config = config
        self.model = CrossEncoder(
            config.model,
            revision=config.revision,
            max_length=config.max_length,
            device=device,
            trust_remote_code=config.trust_remote_code,
            model_kwargs={"dtype": DTYPES[config.dtype]} | config.model_kwargs,
        )
        self.texts: Products = {}

    def index(self, products: Products) -> None:
        self.texts = products

    def score(self, query: str, product_ids: list[str]) -> list[float]:
        pairs = [(query, self.texts[p]) for p in product_ids]
        scores = self.model.predict(
            pairs,
            batch_size=self.config.batch_size,
            prompt=self.config.instruction or None,
            show_progress_bar=False,
        )
        return [float(s) for s in scores]


class YesNoScorer:
    """A causal language model asked whether the product fits the query.

    The prompt is prefix + body + suffix. Only the body is cut when the input
    is too long, so the question at the end always stays. The score is the
    "yes" logit minus the "no" logit at the next token: the same order as
    P("yes"), but it does not round to 1.0 for very relevant products, which
    would create ties at the top.
    """

    def __init__(self, config: RerankConfig, device: str) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.config = config
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model,
            revision=config.revision,
            padding_side="left",
            trust_remote_code=config.trust_remote_code,
        )
        model_kwargs = {"dtype": DTYPES[config.dtype]} | config.model_kwargs
        quantized = "quantization_config" in model_kwargs
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model,
            revision=config.revision,
            trust_remote_code=config.trust_remote_code,
            # bitsandbytes places quantized weights itself.
            device_map=device if quantized else None,
            **model_kwargs,
        )
        if not quantized:
            self.model.to(device)  # type: ignore[arg-type]  # transformers stub types .to wrongly
        self.model.eval()
        self.yes_id = self.tokenizer.convert_tokens_to_ids(config.yes_token)
        self.no_id = self.tokenizer.convert_tokens_to_ids(config.no_token)
        self.prefix_ids = self._ids(config.prefix)
        self.suffix_ids = self._ids(config.suffix)
        self.body_length = config.max_length - len(self.prefix_ids) - len(self.suffix_ids)
        self.texts: Products = {}

    def _ids(self, text: str) -> list[int]:
        ids: list[int] = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        return ids

    def index(self, products: Products) -> None:
        self.texts = products

    def inputs(self, query: str, product_ids: list[str]) -> list[list[int]]:
        """Token ids of the full prompt for each product, body cut to fit max_length."""
        bodies = [
            self.config.body.format(
                instruction=self.config.instruction, query=query, document=self.texts[p]
            )
            for p in product_ids
        ]
        body_ids = self.tokenizer(
            bodies, add_special_tokens=False, truncation=True, max_length=self.body_length
        )["input_ids"]
        return [self.prefix_ids + ids + self.suffix_ids for ids in body_ids]

    @torch.inference_mode()
    def score(self, query: str, product_ids: list[str]) -> list[float]:
        inputs = self.inputs(query, product_ids)
        scores: list[float] = []
        for start in range(0, len(inputs), self.config.batch_size):
            batch = self.tokenizer.pad(
                {"input_ids": inputs[start : start + self.config.batch_size]},
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            # Left padding: count positions from each prompt's first real token.
            positions = (batch["attention_mask"].cumsum(dim=1) - 1).clamp(min=0)
            # Only the last position is read; full logits over the vocabulary
            # for every position would take gigabytes per batch.
            logits = self.model(**batch, position_ids=positions, logits_to_keep=1).logits[:, -1, :]
            scores.extend((logits[:, self.yes_id] - logits[:, self.no_id]).float().tolist())
        return scores


class ListwiseScorer:
    """A model that reads the query and all candidates at once (jina-reranker-v3)."""

    def __init__(self, config: RerankConfig, device: str) -> None:
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(
            config.model,
            revision=config.revision,
            dtype=DTYPES[config.dtype],
            trust_remote_code=config.trust_remote_code,
            **config.model_kwargs,
        )
        self.model.to(device).eval()
        self.texts: Products = {}

    def index(self, products: Products) -> None:
        self.texts = products

    @torch.inference_mode()
    def score(self, query: str, product_ids: list[str]) -> list[float]:
        ranked = self.model.rerank(query, [self.texts[p] for p in product_ids])
        scores = [0.0] * len(product_ids)
        for item in ranked:
            scores[item["index"]] = float(item["relevance_score"])
        return scores


class LateInteractionScorer:
    """ColBERT through sentence-transformers' MultiVectorEncoder.

    Product token vectors are made at index time; at query time only the query
    is encoded, then MaxSim against the stored vectors of its candidates.
    """

    def __init__(self, config: RerankConfig, device: str) -> None:
        from sentence_transformers import MultiVectorEncoder

        self.config = config
        self.device = device
        self.model = MultiVectorEncoder(config.model, revision=config.revision, device=device)
        # The checkpoint default (300 tokens) would cut longer product texts.
        self.model[0].document_length = config.max_length  # type: ignore[assignment]
        self.vectors: dict[str, torch.Tensor] = {}

    def index(self, products: Products) -> None:
        ids = list(products)
        encoded = self.model.encode_document(
            [products[p] for p in ids],
            batch_size=self.config.batch_size,
            show_progress_bar=False,
        )
        self.vectors = dict(zip(ids, encoded, strict=True))

    @torch.inference_mode()
    def score(self, query: str, product_ids: list[str]) -> list[float]:
        query_vectors = self.model.encode_query([query], show_progress_bar=False)
        scores = self.model.similarity(query_vectors, [self.vectors[p] for p in product_ids])
        return [float(s) for s in scores[0]]


class BiEncoderScorer:
    """A dense model re-scoring the candidates: product vectors made at index time."""

    def __init__(self, config: RerankConfig, device: str) -> None:
        _, dense = load_dense_config(Path(config.dense_config))
        self.retriever = DenseRetriever(dense, device)
        self.rows: dict[str, int] = {}
        self.vectors: torch.Tensor | None = None

    def index(self, products: Products) -> None:
        ids = list(products)
        self.rows = {p: i for i, p in enumerate(ids)}
        self.vectors = self.retriever.encode_documents([products[p] for p in ids])

    @torch.inference_mode()
    def score(self, query: str, product_ids: list[str]) -> list[float]:
        if self.vectors is None:
            raise RuntimeError("Call index() before score()")
        query_vector = self.retriever.encode_queries([query])[0]
        rows = self.vectors[[self.rows[p] for p in product_ids]]
        return (rows @ query_vector).tolist()


SCORERS: dict[str, Callable[[RerankConfig, str], Scorer]] = {
    "cross_encoder": CrossEncoderScorer,
    "yes_no": YesNoScorer,
    "listwise": ListwiseScorer,
    "late_interaction": LateInteractionScorer,
    "bi_encoder": BiEncoderScorer,
}


def load_scorer(config: RerankConfig, device: str | None = None) -> Scorer:
    """Load the model for the configured method."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return SCORERS[config.method](config, device)


def rerank(
    scorer: Scorer,
    queries: Queries,
    results: Results,
    depth: int,
    sync: Callable[[], None] | None = None,
) -> tuple[Results, list[float]]:
    """Rerank each query's top depth candidates, one query at a time.

    Args:
        scorer: Indexed scorer.
        queries: query_id -> query text.
        results: First-stage results for the same queries.
        depth: Candidates reranked per query.
        sync: Called before each clock stop, e.g. torch.cuda.synchronize.

    Returns:
        Reranked results (reranked candidates first, the rest below in
        first-stage order) and the reranking time per query in ms.
    """
    ordered = rank(results)
    reranked: Results = {}
    latencies = []
    for query_id, text in queries.items():
        head, tail = ordered[query_id][:depth], ordered[query_id][depth:]
        start = time.perf_counter_ns()
        scores = scorer.score(text, head)
        if sync is not None:
            sync()
        latencies.append((time.perf_counter_ns() - start) / 1e6)
        floor = min(scores, default=0.0) - 1.0
        reranked[query_id] = dict(zip(head, scores, strict=True)) | {
            p: floor - i for i, p in enumerate(tail)
        }
    return reranked, latencies
