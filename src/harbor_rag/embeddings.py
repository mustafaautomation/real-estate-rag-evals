from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from typing import Protocol, cast

import numpy as np

from harbor_rag.textutil import tokens

Vector = np.ndarray[tuple[int], np.dtype[np.float32]]
Matrix = np.ndarray[tuple[int, int], np.dtype[np.float32]]


class Embedder(Protocol):
    dim: int

    def embed_documents(self, texts: Sequence[str]) -> Matrix: ...

    def embed_query(self, text: str) -> Vector: ...


class HashedTfidfEmbedder:
    """Deterministic hashed TF-IDF.

    Offline tests and CI use this so retrieval does not download a model or
    call an API. Live chat can still swap in OpenAIEmbedder.
    """

    def __init__(self, dim: int = 384) -> None:
        if dim < 16:
            raise ValueError("dim must be at least 16")
        self.dim = dim
        self.idf: dict[str, float] = {}

    def fit(self, texts: Sequence[str]) -> None:
        count = len(texts)
        document_frequency: dict[str, int] = {}
        for text in texts:
            for token in set(tokens(text)):
                document_frequency[token] = document_frequency.get(token, 0) + 1
        self.idf = {
            token: math.log((count + 1) / (seen + 1)) + 1.0
            for token, seen in document_frequency.items()
        }

    def embed_documents(self, texts: Sequence[str]) -> Matrix:
        rows = [self._vector(text) for text in texts]
        if not rows:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack(rows).astype(np.float32, copy=False)

    def embed_query(self, text: str) -> Vector:
        return self._vector(text)

    def _vector(self, text: str) -> Vector:
        vec = np.zeros(self.dim, dtype=np.float32)
        counts: dict[str, int] = {}
        for token in tokens(text):
            counts[token] = counts.get(token, 0) + 1
        total = sum(counts.values())
        if total == 0:
            return vec
        default_idf = math.log((len(self.idf) + 1) / 1) + 1.0
        for token, count in counts.items():
            weight = (count / total) * self.idf.get(token, default_idf)
            _accumulate(vec, token, weight)
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= np.float32(norm)
        return vec


class OpenAIEmbedder:
    """OpenAI embeddings behind the same interface. Requires a real client."""

    def __init__(self, client: object, model: str = "text-embedding-3-small") -> None:
        self._client = client
        self.model = model
        self.dim = 1536 if "small" in model else 3072
        self.idf: dict[str, float] = {}

    def embed_documents(self, texts: Sequence[str]) -> Matrix:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        response = self._client.embeddings.create(model=self.model, input=list(texts))  # type: ignore[attr-defined]
        rows = [item.embedding for item in response.data]
        matrix = np.asarray(rows, dtype=np.float32)
        self.dim = int(matrix.shape[1])
        return cast(Matrix, matrix)

    def embed_query(self, text: str) -> Vector:
        return cast(Vector, self.embed_documents([text])[0])


def _accumulate(vec: Vector, token: str, weight: float) -> None:
    digest = hashlib.sha256(token.encode()).digest()
    dim = int(vec.shape[0])
    for slot in range(4):
        start = slot * 2
        bucket = int.from_bytes(digest[start : start + 2], "little") % dim
        sign = 1.0 if digest[8 + slot] % 2 == 0 else -1.0
        vec[bucket] += np.float32(sign * weight)
