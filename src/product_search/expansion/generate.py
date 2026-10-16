"""Generate expansions: extra text for products (offline) or queries.

Two kinds of generator:
    seq2seq  a model trained to write search queries for a passage (doc2query T5);
             several queries are sampled per input, as in docTTTTTquery
    chat     an instruction model (Qwen3) given a prompt; greedy decoding,
             thinking switched off

Chat models run in transformers, or (engine: vllm) on a vLLM server started
with scripts/serve_vllm.ps1. The prompts are built the same way for both;
vLLM batches the requests itself, which is far faster for long runs.

A run keeps, per input, the raw model output and the cleaned text that goes
into search. Cleaning depends on what the prompt asks for (config `parse`):
    lines      one item per line (search queries); numbering and bullets removed
    comma      comma-separated items (keywords)
    key_value  "attribute: value" lines; only the values are kept, unknowns dropped
    text       free text (a short pseudo-product)
Lead-in lines such as "Here are 20 queries:" are removed first. Repeated items
are dropped unless the config sets dedupe: false (docTTTTTquery keeps the
repeats of its sampled queries; for BM25 they weigh the words up).
"""

import json
import logging
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, cast

import torch
import yaml

from product_search.data.wands import DEFAULT_TEXT_FIELDS
from product_search.retrieval.dense import DTYPES

KINDS = ("seq2seq", "chat")
ENGINES = ("transformers", "vllm")
TARGETS = ("products", "queries")
PARSERS = ("lines", "comma", "key_value", "text")
UNKNOWN_VALUES = {"", "unknown", "none", "n/a", "na", "not specified", "not stated", "-"}
GENERATOR_SETTINGS = ("model", "revision", "dtype", "batch_size", "model_kwargs", "engine")
REQUEST_TRIES = 4
PROGRESS_STEPS = 20  # log progress about every 5%
PROGRESS_MIN = 1000  # ... for runs at least this long
THINK_BLOCK = re.compile(r"<think>.*?</think>", flags=re.DOTALL)
LIST_MARKER = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")
PREAMBLE = re.compile(
    r"^\s*(?:here(?:'s| is| are)|sure|certainly|of course|below (?:is|are)|okay|ok)\b", re.I
)


@dataclass(frozen=True)
class ExpansionConfig:
    """Expansion settings.

    Attributes:
        target: "products" or "queries".
        kind: "seq2seq" or "chat".
        parse: How the output is cleaned, one of PARSERS.
        model: Hugging Face model id (may come from the generators file).
        revision: Commit of the model repo.
        prompt: chat only: user message with {text} (products) or {query}
            and optionally {titles} (queries).
        fields: Product fields given to the model.
        max_input_tokens: Product text is cut to this many tokens.
        max_new_tokens: Longest output per sample.
        num_samples: seq2seq only: queries sampled per product.
        top_k: seq2seq only: sampling from the k most likely tokens.
        dtype: "float32" or "bfloat16".
        batch_size: Inputs per generate call (vllm: requests in flight).
        model_kwargs: Extra loading options, e.g. a quantization_config.
        grounding_k: queries only: first-stage product titles put in {titles}.
        seed: Random seed for sampling.
        dedupe: Drop repeated items when cleaning.
        engine: "transformers" or "vllm" (chat only).
        server_url: vllm only: address of the server.
    """

    target: str
    kind: str
    parse: str
    model: str = ""
    revision: str = ""
    prompt: str = ""
    fields: tuple[str, ...] = DEFAULT_TEXT_FIELDS
    max_input_tokens: int = 384
    max_new_tokens: int = 64
    num_samples: int = 1
    top_k: int = 10
    dtype: str = "bfloat16"
    batch_size: int = 16
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    grounding_k: int = 0
    seed: int = 0
    dedupe: bool = True
    engine: str = "transformers"
    server_url: str = "http://localhost:8000"


def load_expansion_config(
    path: Path, generator: str | None = None, generators: Path | None = None
) -> tuple[str, ExpansionConfig]:
    """Read a run name and expansion settings, optionally with a generator from a registry.

    Args:
        path: Expansion YAML (task and prompt).
        generator: Key in the generators file; its model settings replace the
            config's, and the run is named <name>_<generator>.
        generators: Generators YAML (key -> model, revision, dtype, batch_size,
            model_kwargs).

    Raises:
        ValueError: If 'name' is missing, a setting is unknown or out of range,
            or the generator is not in the registry.
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    name = raw.pop("name", None)
    if name is None:
        raise ValueError(f"{path}: missing 'name'")
    if generator is not None:
        if generators is None:
            raise ValueError("a generator needs a generators file")
        registry = yaml.safe_load(generators.read_text(encoding="utf-8"))
        if generator not in registry:
            raise ValueError(f"{generators}: no generator '{generator}'")
        settings = registry[generator]
        unknown = set(settings) - set(GENERATOR_SETTINGS)
        if unknown:
            raise ValueError(f"{generators}: unknown generator settings {sorted(unknown)}")
        raw |= settings
        name = f"{name}_{generator}"
    unknown = set(raw) - {f.name for f in fields(ExpansionConfig)}
    if unknown:
        raise ValueError(f"{path}: unknown expansion settings {sorted(unknown)}")
    for key, allowed in (("target", TARGETS), ("kind", KINDS), ("parse", PARSERS)):
        if raw.get(key) not in allowed:
            raise ValueError(f"{path}: {key} must be one of {allowed}")
    if raw.get("engine", "transformers") not in ENGINES:
        raise ValueError(f"{path}: engine must be one of {ENGINES}")
    if raw.get("engine") == "vllm" and raw["kind"] != "chat":
        raise ValueError(f"{path}: the vllm engine is for chat models")
    if raw.get("dtype", "bfloat16") not in DTYPES:
        raise ValueError(f"{path}: dtype must be one of {sorted(DTYPES)}")
    if not raw.get("model") or not raw.get("revision"):
        raise ValueError(f"{path}: model and revision are required (or pass a generator)")
    if "fields" in raw:
        raw["fields"] = tuple(raw["fields"])
    return name, ExpansionConfig(**raw)


def save_expansion_config(path: Path, name: str, config: ExpansionConfig) -> None:
    """Write a run name and expansion settings as YAML."""
    settings = asdict(config) | {"fields": list(config.fields)}
    path.write_text(yaml.safe_dump({"name": name, **settings}, sort_keys=False), encoding="utf-8")


def _dedupe(items: list[str], dedupe: bool = True) -> list[str]:
    seen, kept = set(), []
    for item in items:
        key = item.lower()
        if item and (not dedupe or key not in seen):
            seen.add(key)
            kept.append(item)
    return kept


def _strip_preamble(line: str) -> str:
    """Remove a lead-in: "Here are some keywords: sofa, couch" -> "sofa, couch";
    a lead-in or heading with nothing after its colon -> ""."""
    if PREAMBLE.match(line):
        return line.split(":", 1)[1].strip() if ":" in line else ""
    return "" if line.strip().endswith(":") else line


def clean(raw: str, parse: str, dedupe: bool = True) -> str:
    """Turn a raw model output into the text used for search.

    Args:
        raw: Model output.
        parse: One of PARSERS.
        dedupe: Drop repeated items (case-insensitive).

    Returns:
        lines -> one item per line; comma -> items joined by ", ";
        key_value -> known values joined by ", "; text -> stripped text.
    """
    lines = THINK_BLOCK.sub("", raw).strip().splitlines()
    if parse == "text":
        # Only a lead-in at the start; a colon later may be part of the text.
        if lines:
            lines[0] = _strip_preamble(lines[0])
        return "\n".join(lines).strip()
    if parse == "key_value":
        # "type: sofa" ends with a value, so only lead-ins are removed.
        lines = [line.split(":", 1)[1] if PREAMBLE.match(line) else line for line in lines]
    else:
        lines = [_strip_preamble(line) for line in lines]
    if parse == "lines":
        items = [LIST_MARKER.sub("", line).strip().strip('"') for line in lines]
        return "\n".join(_dedupe(items, dedupe))
    if parse == "comma":
        items = [i.strip().strip('"').strip(".") for i in re.split(r"[,\n]", "\n".join(lines))]
        return ", ".join(_dedupe([LIST_MARKER.sub("", i) for i in items], dedupe))
    values = []
    for line in lines:
        if ":" not in line:
            continue
        value = line.split(":", 1)[1].strip().strip('"').strip(".")
        if value.lower() not in UNKNOWN_VALUES:
            values.append(value)
    return ", ".join(_dedupe(values, dedupe))


class Generator:
    """Loads the model once and generates raw outputs for a list of inputs."""

    def __init__(self, config: ExpansionConfig, device: str | None = None) -> None:
        from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model,
            revision=config.revision,
            # Decoder-only models generate after the prompt, so pad on the left.
            padding_side="left" if config.kind == "chat" else "right",
        )
        if config.engine == "vllm":
            # The server holds the model; only the tokenizer is needed here.
            self.model: Any = None
            self._wait_for_server()
            return
        model_class = AutoModelForCausalLM if config.kind == "chat" else AutoModelForSeq2SeqLM
        model_kwargs = {"dtype": DTYPES[config.dtype]} | config.model_kwargs
        quantized = "quantization_config" in model_kwargs
        # Typed as Any: the transformers stubs do not cover generate() on Auto classes.
        self.model = cast(
            Any,
            model_class.from_pretrained(
                config.model,
                revision=config.revision,
                # bitsandbytes places quantized weights itself.
                device_map=self.device if quantized else None,
                **model_kwargs,
            ),
        )
        if not quantized:
            self.model.to(self.device)
        self.model.eval()

    def truncate(self, texts: list[str]) -> list[str]:
        """Cut each text to max_input_tokens tokens."""
        ids = self.tokenizer(
            texts,
            add_special_tokens=False,
            truncation=True,
            max_length=self.config.max_input_tokens,
        )["input_ids"]
        return self.tokenizer.batch_decode(ids, skip_special_tokens=True)

    def chat_prompts(self, messages: list[str]) -> list[str]:
        """Wrap user messages in the model's chat template, thinking off."""
        return [
            str(
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": message}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )
            for message in messages
        ]

    @torch.inference_mode()
    def _generate_batch(self, inputs: list[str]) -> tuple[list[str], int]:
        c = self.config
        # Chat prompts already carry the template's special tokens; T5 needs its end token.
        batch = self.tokenizer(
            inputs, padding=True, return_tensors="pt", add_special_tokens=c.kind == "seq2seq"
        ).to(self.device)
        if c.kind == "seq2seq":
            output = self.model.generate(
                **batch,
                do_sample=True,
                top_k=c.top_k,
                max_new_tokens=c.max_new_tokens,
                num_return_sequences=c.num_samples,
            )
            texts = self.tokenizer.batch_decode(output, skip_special_tokens=True)
            grouped = [
                "\n".join(texts[i * c.num_samples : (i + 1) * c.num_samples])
                for i in range(len(inputs))
            ]
            new_tokens = int((output != self.tokenizer.pad_token_id).sum())
            return grouped, new_tokens
        # Greedy: unset the sampling defaults from the model's generation config.
        output = self.model.generate(
            **batch,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            max_new_tokens=c.max_new_tokens,
        )
        generated = output[:, batch["input_ids"].shape[1] :]
        new_tokens = int((generated != self.tokenizer.pad_token_id).sum())
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True), new_tokens

    def _wait_for_server(self, timeout: float = 900) -> None:
        deadline = time.monotonic() + timeout
        while True:
            try:
                with urllib.request.urlopen(f"{self.config.server_url}/health", timeout=5):
                    return
            except OSError:
                if time.monotonic() > deadline:
                    raise RuntimeError(f"no vLLM server at {self.config.server_url}") from None
                time.sleep(5)

    def _complete(self, prompt: str) -> tuple[str, int]:
        c = self.config
        body = {
            "model": c.model,
            "prompt": prompt,
            "max_tokens": c.max_new_tokens,
            "temperature": 0.0,
            "seed": c.seed,
        }
        request = urllib.request.Request(
            f"{c.server_url}/v1/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        # A long run sends tens of thousands of requests; retry a few failures.
        for attempt in range(REQUEST_TRIES):
            try:
                with urllib.request.urlopen(request, timeout=900) as response:
                    reply = json.load(response)
                return reply["choices"][0]["text"], int(reply["usage"]["completion_tokens"])
            except OSError:
                if attempt == REQUEST_TRIES - 1:
                    raise
                time.sleep(10 * (attempt + 1))
        raise AssertionError("unreachable")

    def _generate_server(self, inputs: list[str]) -> tuple[list[str], int]:
        # The server batches whatever is in flight (continuous batching).
        replies: list[tuple[str, int]] = [("", 0)] * len(inputs)
        step = max(1, len(inputs) // PROGRESS_STEPS)
        with ThreadPoolExecutor(self.config.batch_size) as pool:
            futures = {pool.submit(self._complete, text): i for i, text in enumerate(inputs)}
            for done, future in enumerate(as_completed(futures), 1):
                replies[futures[future]] = future.result()
                if len(inputs) >= PROGRESS_MIN and done % step == 0:
                    logging.info("  %d / %d", done, len(inputs))
        return [text for text, _ in replies], sum(n for _, n in replies)

    def generate(self, inputs: list[str]) -> tuple[list[str], int]:
        """Raw outputs in input order, and the number of generated tokens.

        Inputs are batched by length so a batch carries little padding.
        """
        if self.config.engine == "vllm":
            return self._generate_server(inputs)
        torch.manual_seed(self.config.seed)
        order = sorted(range(len(inputs)), key=lambda i: len(inputs[i]))
        outputs: list[str] = [""] * len(inputs)
        new_tokens = 0
        starts = range(0, len(order), self.config.batch_size)
        step = max(1, len(starts) // PROGRESS_STEPS)
        for n, start in enumerate(starts, 1):
            chunk = order[start : start + self.config.batch_size]
            texts, count = self._generate_batch([inputs[i] for i in chunk])
            new_tokens += count
            for i, text in zip(chunk, texts, strict=True):
                outputs[i] = text
            if len(inputs) >= PROGRESS_MIN and n % step == 0:
                logging.info("  %d / %d", min(start + len(chunk), len(inputs)), len(inputs))
        return outputs, new_tokens


def product_inputs(generator: Generator, texts: list[str]) -> list[str]:
    """Model inputs for product texts: the text itself (seq2seq) or the filled prompt (chat)."""
    texts = generator.truncate(texts)
    if generator.config.kind == "seq2seq":
        return texts
    return generator.chat_prompts([generator.config.prompt.format(text=t) for t in texts])


def query_inputs(
    generator: Generator, queries: list[str], titles: list[list[str]] | None = None
) -> list[str]:
    """Chat inputs for queries, with first-stage product titles when the prompt grounds on them."""
    prompt = generator.config.prompt
    messages = [
        prompt.format(
            query=query,
            titles="\n".join(f"- {t}" for t in titles[i]) if titles is not None else "",
        )
        for i, query in enumerate(queries)
    ]
    return generator.chat_prompts(messages)
