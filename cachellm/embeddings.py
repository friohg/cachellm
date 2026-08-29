"""Embedding backends for the semantic cache.

Three interchangeable backends, selected by ``SEMANTIC_BACKEND``:

``hash``                  dependency-free local hashing embedder (default).
                          Deterministic, offline, no model download; good enough
                          to catch paraphrases with high thresholds.
``sentence_transformers`` real local model (``pip install cachellm[semantic]``).
``openai``                any OpenAI-compatible /embeddings endpoint.

Vectors are stored as float32 little-endian blobs and always L2-normalised, so
cosine similarity is a plain dot product.
"""

from __future__ import annotations

import array
import hashlib
import math
import re
from typing import Any, Protocol, Sequence, runtime_checkable

from .config import SemanticConfig
from .logging_utils import get_logger

log = get_logger("cachellm.embeddings")

_TOKEN_RE = re.compile(r"[a-z0-9']+")


@runtime_checkable
class EmbeddingBackend(Protocol):
    name: str
    dimensions: int

    async def embed(self, text: str) -> list[float]: ...
    async def close(self) -> None: ...


# ---------------------------------------------------------------------------
# vector helpers
# ---------------------------------------------------------------------------


def pack_vector(vector: Sequence[float]) -> bytes:
    return array.array("f", [float(v) for v in vector]).tobytes()


def unpack_vector(blob: bytes) -> list[float]:
    arr = array.array("f")
    arr.frombytes(blob)
    return list(arr)


def l2_normalize(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(float(v) * float(v) for v in vector))
    if norm == 0.0:
        return [0.0] * len(vector)
    return [float(v) / norm for v in vector]


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity; inputs are expected pre-normalised but we are safe anyway."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = na = nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / math.sqrt(na * nb)


# ---------------------------------------------------------------------------
# hashing backend (offline default)
# ---------------------------------------------------------------------------


class HashingEmbedder:
    """Local bag-of-ngrams hashing embedder.

    Uses word unigrams + bigrams + character 4-grams with sublinear term
    weighting, hashed into a fixed-size vector.  Purely local and deterministic;
    no model files, no network.
    """

    name = "hash"

    def __init__(self, dimensions: int = 256) -> None:
        self.dimensions = max(32, int(dimensions))

    async def embed(self, text: str) -> list[float]:
        return self.embed_sync(text)

    def embed_sync(self, text: str) -> list[float]:
        tokens = _TOKEN_RE.findall((text or "").lower())
        features: dict[str, float] = {}
        for token in tokens:
            features[f"w:{token}"] = features.get(f"w:{token}", 0.0) + 1.0
        for a, b in zip(tokens, tokens[1:]):
            key = f"b:{a}_{b}"
            features[key] = features.get(key, 0.0) + 1.0
        compact = re.sub(r"\s+", " ", (text or "").lower())
        for i in range(0, max(0, len(compact) - 3)):
            key = f"c:{compact[i : i + 4]}"
            features[key] = features.get(key, 0.0) + 0.5

        vector = [0.0] * self.dimensions
        for feature, count in features.items():
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign * (1.0 + math.log(count))
        return l2_normalize(vector)

    async def close(self) -> None:
        return None


# ---------------------------------------------------------------------------
# sentence-transformers backend (real local model)
# ---------------------------------------------------------------------------


class SentenceTransformerEmbedder:
    name = "sentence_transformers"

    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dep
            raise RuntimeError(
                "SEMANTIC_BACKEND=sentence_transformers requires "
                "'pip install \"cachellm[semantic]\"'"
            ) from exc
        self._model = SentenceTransformer(model_name)
        self.dimensions = int(self._model.get_sentence_embedding_dimension())
        self.model_name = model_name

    async def embed(self, text: str) -> list[float]:
        import asyncio

        vector = await asyncio.to_thread(
            lambda: self._model.encode(text, normalize_embeddings=True)
        )
        return l2_normalize([float(v) for v in vector])

    async def close(self) -> None:
        return None


# ---------------------------------------------------------------------------
# OpenAI-compatible embeddings endpoint
# ---------------------------------------------------------------------------


class OpenAIEmbedder:
    name = "openai"

    def __init__(self, *, base_url: str, api_key: str, model: str, dimensions: int = 0) -> None:
        import httpx

        if not base_url:
            raise RuntimeError("SEMANTIC_BACKEND=openai requires EMBED_BASE_URL")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.dimensions = int(dimensions or 0)
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(timeout=30.0, headers=headers)

    async def embed(self, text: str) -> list[float]:
        payload: dict[str, Any] = {"model": self.model, "input": text}
        response = await self._client.post(f"{self.base_url}/embeddings", json=payload)
        response.raise_for_status()
        data = response.json()
        vector = data["data"][0]["embedding"]
        self.dimensions = len(vector)
        return l2_normalize([float(v) for v in vector])

    async def close(self) -> None:
        await self._client.aclose()


def build_embedder(cfg: SemanticConfig) -> EmbeddingBackend:
    backend = (cfg.backend or "hash").lower()
    if backend in {"hash", "local", "hashing", "none"}:
        return HashingEmbedder(cfg.dimensions)
    if backend in {"sentence_transformers", "sentence-transformers", "st"}:
        return SentenceTransformerEmbedder(cfg.model)
    if backend in {"openai", "openai_compatible"}:
        return OpenAIEmbedder(
            base_url=cfg.embed_base_url,
            api_key=cfg.embed_api_key,
            model=cfg.model,
            dimensions=cfg.dimensions,
        )
    raise RuntimeError(f"unknown SEMANTIC_BACKEND: {cfg.backend!r}")
