from __future__ import annotations

from pathlib import Path

from harbor_rag.chunking import chunk_corpus
from harbor_rag.config import AppConfig, load_config
from harbor_rag.corpus import load_corpus
from harbor_rag.embeddings import HashedTfidfEmbedder
from harbor_rag.eval.dataset import load_questions
from harbor_rag.eval.metrics import MetricRow, PipelineRecord, score_records
from harbor_rag.paths import repo_root
from harbor_rag.pipeline import Pipeline
from harbor_rag.retrieval import FaissIndex
from harbor_rag.types import Chunk, EvalItem, Variant


def load_app(
    root: Path | None = None,
) -> tuple[Path, AppConfig, list[EvalItem], list[Chunk], HashedTfidfEmbedder]:
    resolved = root or repo_root()
    config = load_config(resolved / "evals" / "config.yaml")
    documents = load_corpus(resolved / "data" / "corpus")
    questions = load_questions(resolved / "data" / "eval" / "questions.json")
    known = {document.doc_id for document in documents}
    for item in questions:
        missing = [doc_id for doc_id in item.reference_doc_ids if doc_id not in known]
        if missing:
            raise ValueError(f"{item.item_id} references unknown documents {missing}")
    chunks = chunk_corpus(documents, config.chunk_max_chars, config.chunk_overlap)
    embedder = HashedTfidfEmbedder(dim=config.embedding_dim)
    embedder.fit([chunk.text for chunk in chunks])
    return resolved, config, questions, chunks, embedder


def run_variant(
    variant: Variant, root: Path | None = None
) -> tuple[AppConfig, list[PipelineRecord]]:
    resolved, config, questions, chunks, embedder = load_app(root)
    index = FaissIndex(chunks, embedder)
    pipeline = Pipeline(index, embedder, k=config.retrieval_k, variant=variant)
    by_doc: dict[str, list[str]] = {}
    for chunk in chunks:
        by_doc.setdefault(chunk.doc_id, []).append(chunk.text)
    records: list[PipelineRecord] = []
    for item in questions:
        result = pipeline.ask(item.user_input)
        reference_contexts: list[str] = []
        for doc_id in item.reference_doc_ids:
            reference_contexts.extend(by_doc.get(doc_id, []))
        records.append(
            PipelineRecord(
                item_id=item.item_id,
                user_input=item.user_input,
                response=result.response,
                expected_behavior=item.expected_behavior,
                reference_phrase=item.reference_phrase,
                retrieved_doc_ids=result.retrieved_doc_ids,
                reference_doc_ids=item.reference_doc_ids,
                retrieved_contexts=result.retrieved_contexts,
                reference_contexts=tuple(reference_contexts),
            )
        )
    return config, records


def score_variant(
    variant: Variant, root: Path | None = None
) -> tuple[AppConfig, list[PipelineRecord], list[MetricRow]]:
    config, records = run_variant(variant, root)
    return config, records, score_records(records)
