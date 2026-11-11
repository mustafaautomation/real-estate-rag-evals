from __future__ import annotations

import asyncio
import warnings
from dataclasses import dataclass

from harbor_rag.guardrails import classify_response
from harbor_rag.textutil import dollar_amounts
from harbor_rag.types import ExpectedBehavior

# ragas 0.4.3 documents NonLLMContextRecall on the legacy metrics module.
# The deprecation warning points at ragas.metrics.collections, which does not
# export the class in this release. StringPresence comes from collections.
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message=r"Importing NonLLMContextRecall .*",
        category=DeprecationWarning,
    )
    from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics import NonLLMContextRecall
    from ragas.metrics.collections import StringPresence


@dataclass(frozen=True)
class PipelineRecord:
    item_id: str
    user_input: str
    response: str
    expected_behavior: ExpectedBehavior
    reference_phrase: str
    retrieved_doc_ids: tuple[str, ...]
    reference_doc_ids: tuple[str, ...]
    retrieved_contexts: tuple[str, ...]
    reference_contexts: tuple[str, ...]


@dataclass(frozen=True)
class MetricRow:
    item_id: str
    user_input: str
    response: str
    expected_behavior: ExpectedBehavior
    predicted_behavior: str
    retrieved_doc_ids: tuple[str, ...]
    reference_doc_ids: tuple[str, ...]
    hit: bool | None
    reciprocal_rank: float | None
    non_llm_context_recall: float | None
    string_presence: float
    refusal_correct: float
    grounded_amounts: float

    @property
    def failed(self) -> bool:
        if self.refusal_correct < 1.0 or self.string_presence < 1.0 or self.grounded_amounts < 1.0:
            return True
        if self.hit is False:
            return True
        return self.non_llm_context_recall is not None and self.non_llm_context_recall < 1.0


def score_records(records: list[PipelineRecord]) -> list[MetricRow]:
    presence = StringPresence()
    recall_inputs: list[tuple[int, list[str], list[str]]] = []
    scored: list[MetricRow] = []
    for index, record in enumerate(records):
        predicted = classify_response(record.response)
        hit, reciprocal = _retrieval_stats(record.retrieved_doc_ids, record.reference_doc_ids)
        presence_value = float(
            presence.score(reference=record.reference_phrase, response=record.response).value
        )
        if record.reference_contexts and record.retrieved_contexts:
            recall_inputs.append(
                (index, list(record.retrieved_contexts), list(record.reference_contexts))
            )
        scored.append(
            MetricRow(
                item_id=record.item_id,
                user_input=record.user_input,
                response=record.response,
                expected_behavior=record.expected_behavior,
                predicted_behavior=predicted,
                retrieved_doc_ids=record.retrieved_doc_ids,
                reference_doc_ids=record.reference_doc_ids,
                hit=hit,
                reciprocal_rank=reciprocal,
                non_llm_context_recall=None,
                string_presence=presence_value,
                refusal_correct=1.0 if predicted == record.expected_behavior.value else 0.0,
                grounded_amounts=1.0
                if _amounts_grounded(record.response, record.retrieved_contexts)
                else 0.0,
            )
        )
    recall_scores = _context_recall(recall_inputs)
    if not recall_scores:
        return scored
    return [
        _with_recall(row, recall_scores[index]) if index in recall_scores else row
        for index, row in enumerate(scored)
    ]


def aggregate(rows: list[MetricRow]) -> dict[str, float]:
    hits = [1.0 if row.hit else 0.0 for row in rows if row.hit is not None]
    reciprocal = [row.reciprocal_rank for row in rows if row.reciprocal_rank is not None]
    recall = [row.non_llm_context_recall for row in rows if row.non_llm_context_recall is not None]
    return {
        "hit_rate": _mean(hits),
        "mrr": _mean([value for value in reciprocal if value is not None]),
        "non_llm_context_recall": _mean([value for value in recall if value is not None]),
        "string_presence": _mean([row.string_presence for row in rows]),
        "refusal_correctness": _mean([row.refusal_correct for row in rows]),
        "grounded_amounts": _mean([row.grounded_amounts for row in rows]),
    }


def _with_recall(row: MetricRow, score: float) -> MetricRow:
    return MetricRow(
        item_id=row.item_id,
        user_input=row.user_input,
        response=row.response,
        expected_behavior=row.expected_behavior,
        predicted_behavior=row.predicted_behavior,
        retrieved_doc_ids=row.retrieved_doc_ids,
        reference_doc_ids=row.reference_doc_ids,
        hit=row.hit,
        reciprocal_rank=row.reciprocal_rank,
        non_llm_context_recall=score,
        string_presence=row.string_presence,
        refusal_correct=row.refusal_correct,
        grounded_amounts=row.grounded_amounts,
    )


def _retrieval_stats(
    retrieved_doc_ids: tuple[str, ...], reference_doc_ids: tuple[str, ...]
) -> tuple[bool | None, float | None]:
    if not reference_doc_ids:
        return None, None
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in reference_doc_ids:
            return True, 1.0 / rank
    return False, 0.0


def _amounts_grounded(response: str, contexts: tuple[str, ...]) -> bool:
    amounts = dollar_amounts(response)
    if not amounts:
        return True
    blob = " ".join(contexts).replace(",", "")
    return all(amount.replace(",", "") in blob for amount in amounts)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _context_recall(inputs: list[tuple[int, list[str], list[str]]]) -> dict[int, float]:
    if not inputs:
        return {}

    async def _run() -> dict[int, float]:
        metric = NonLLMContextRecall()
        scores: dict[int, float] = {}
        for index, retrieved, reference in inputs:
            sample = SingleTurnSample(
                retrieved_contexts=retrieved,
                reference_contexts=reference,
            )
            scores[index] = float(await metric.single_turn_ascore(sample))
        return scores

    return asyncio.run(_run())
