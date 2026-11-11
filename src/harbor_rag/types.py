from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Variant(StrEnum):
    BASELINE = "baseline"
    BAD_RETRIEVER = "bad_retriever"
    NO_GUARDRAIL = "no_guardrail"


class ExpectedBehavior(StrEnum):
    ANSWER = "answer"
    REFUSE_FAIR_HOUSING = "refuse_fair_housing"
    REFUSE_LEGAL = "refuse_legal"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    body: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class EvalItem:
    item_id: str
    user_input: str
    reference: str
    reference_phrase: str
    reference_doc_ids: tuple[str, ...]
    expected_behavior: ExpectedBehavior
    tags: tuple[str, ...]


@dataclass(frozen=True)
class AskResult:
    question: str
    response: str
    retrieved_contexts: tuple[str, ...]
    retrieved_doc_ids: tuple[str, ...]
    system_prompt: str
    guardrail: str | None
