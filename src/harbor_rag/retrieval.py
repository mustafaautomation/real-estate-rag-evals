from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from harbor_rag.embeddings import Embedder, Matrix
from harbor_rag.types import Chunk, ScoredChunk


class Retriever(Protocol):
    def retrieve(self, query: str, k: int) -> list[ScoredChunk]: ...


class FaissIndex:
    def __init__(self, chunks: Sequence[Chunk], embedder: Embedder) -> None:
        import faiss

        self.chunks = list(chunks)
        self.embedder = embedder
        self._matrix: Matrix = embedder.embed_documents([chunk.text for chunk in self.chunks])
        if self._matrix.shape[0] != len(self.chunks):
            raise ValueError("embedding row count does not match chunk count")
        index = faiss.IndexFlatIP(int(self._matrix.shape[1]))
        if len(self.chunks):
            index.add(np.ascontiguousarray(self._matrix, dtype=np.float32))
        self._index = index

    def rank(self, query: str) -> list[ScoredChunk]:
        if not self.chunks:
            return []
        query_vector = np.ascontiguousarray(
            self.embedder.embed_query(query).reshape(1, -1), dtype=np.float32
        )
        scores, indices = self._index.search(query_vector, len(self.chunks))
        score_row = np.asarray(scores[0], dtype=np.float32)
        index_row = np.asarray(indices[0], dtype=np.int64)
        ranked: list[ScoredChunk] = []
        for score, index in zip(score_row.tolist(), index_row.tolist(), strict=True):
            if index < 0:
                continue
            ranked.append(ScoredChunk(chunk=self.chunks[int(index)], score=float(score)))
        return ranked

    def retrieve(self, query: str, k: int) -> list[ScoredChunk]:
        return self.rank(query)[:k]


class BottomRetriever:
    """Deliberately bad retriever: the least similar chunks."""

    def __init__(self, index: FaissIndex) -> None:
        self._index = index

    def retrieve(self, query: str, k: int) -> list[ScoredChunk]:
        ranked = self._index.rank(query)
        if len(ranked) <= k:
            return list(reversed(ranked))
        return list(reversed(ranked[-k:]))


def vector_literal(values: Sequence[float]) -> str:
    return "[" + ",".join(f"{value:.8f}" for value in values) + "]"


CREATE_CHUNKS_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;
DROP TABLE IF EXISTS harbor_chunks;
CREATE TABLE harbor_chunks (
    chunk_id text PRIMARY KEY,
    doc_id text NOT NULL,
    title text NOT NULL,
    content text NOT NULL,
    embedding vector({dim})
);
"""

UPSERT_SQL = """
INSERT INTO harbor_chunks (chunk_id, doc_id, title, content, embedding)
VALUES (%s, %s, %s, %s, %s::vector)
"""

SEARCH_SQL = """
SELECT chunk_id, doc_id, title, content, 1 - (embedding <=> %s::vector) AS score
FROM harbor_chunks
ORDER BY embedding <=> %s::vector
LIMIT %s
"""

SEARCH_WORST_SQL = """
SELECT chunk_id, doc_id, title, content, 1 - (embedding <=> %s::vector) AS score
FROM harbor_chunks
ORDER BY embedding <=> %s::vector DESC
LIMIT %s
"""


class PgVectorRetriever:
    """Postgres retriever. The database is created by docker-compose."""

    def __init__(self, dsn: str, *, worst: bool = False) -> None:
        self.dsn = dsn
        self.worst = worst

    def build(self, chunks: Sequence[Chunk], embedder: Embedder) -> None:
        import psycopg

        matrix = embedder.embed_documents([chunk.text for chunk in chunks])
        dim = int(matrix.shape[1]) if matrix.size else embedder.dim
        with psycopg.connect(self.dsn) as conn:
            conn.execute(CREATE_CHUNKS_SQL.format(dim=dim))
            rows = [
                (
                    chunk.chunk_id,
                    chunk.doc_id,
                    chunk.title,
                    chunk.text,
                    vector_literal(matrix[row].tolist()),
                )
                for row, chunk in enumerate(chunks)
            ]
            with conn.cursor() as cursor:
                cursor.executemany(UPSERT_SQL, rows)

    def retrieve(self, query: str, k: int, embedder: Embedder) -> list[ScoredChunk]:
        import psycopg

        literal = vector_literal(embedder.embed_query(query).tolist())
        sql = SEARCH_WORST_SQL if self.worst else SEARCH_SQL
        with psycopg.connect(self.dsn) as conn:
            fetched = conn.execute(sql, (literal, literal, k)).fetchall()
        results: list[ScoredChunk] = []
        for chunk_id, doc_id, title, content, score in fetched:
            chunk = Chunk(
                chunk_id=str(chunk_id),
                doc_id=str(doc_id),
                title=str(title),
                text=str(content),
            )
            results.append(ScoredChunk(chunk=chunk, score=float(score)))
        return results
