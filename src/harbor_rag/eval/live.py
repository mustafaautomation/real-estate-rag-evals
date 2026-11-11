from __future__ import annotations

import asyncio
import os

from harbor_rag.eval.metrics import PipelineRecord


class LiveEvalError(RuntimeError):
    pass


def require_openai_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key.strip():
        raise LiveEvalError(
            "OPENAI_API_KEY is not set. Live RAGAS metrics were not run "
            "and no judge scores were invented."
        )
    return key


def score_live(
    records: list[PipelineRecord],
    *,
    model: str,
    embedding_model: str,
) -> dict[str, float]:
    """LLM-judged RAGAS metrics on answerable rows. Requires OPENAI_API_KEY."""

    require_openai_key()
    answerable = [record for record in records if record.expected_behavior.value == "answer"]
    if not answerable:
        raise LiveEvalError("no answerable rows to score")

    async def _run() -> dict[str, float]:
        from openai import AsyncOpenAI
        from ragas.embeddings.base import BaseRagasEmbedding, embedding_factory
        from ragas.llms import llm_factory
        from ragas.metrics.collections import (
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
            Faithfulness,
        )

        client = AsyncOpenAI()
        llm = llm_factory(model, client=client)
        embeddings = embedding_factory("openai", model=embedding_model, client=client)
        if not isinstance(embeddings, BaseRagasEmbedding):
            raise LiveEvalError("embedding factory returned an unsupported embeddings object")
        faithfulness = Faithfulness(llm=llm)
        relevancy = AnswerRelevancy(llm=llm, embeddings=embeddings)
        precision = ContextPrecision(llm=llm)
        recall = ContextRecall(llm=llm)
        totals = {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
        }
        for record in answerable:
            contexts = list(record.retrieved_contexts)
            totals["faithfulness"] += float(
                (
                    await faithfulness.ascore(
                        user_input=record.user_input,
                        response=record.response,
                        retrieved_contexts=contexts,
                    )
                ).value
            )
            totals["answer_relevancy"] += float(
                (
                    await relevancy.ascore(
                        user_input=record.user_input,
                        response=record.response,
                    )
                ).value
            )
            totals["context_precision"] += float(
                (
                    await precision.ascore(
                        user_input=record.user_input,
                        reference=record.reference_phrase,
                        retrieved_contexts=contexts,
                    )
                ).value
            )
            totals["context_recall"] += float(
                (
                    await recall.ascore(
                        user_input=record.user_input,
                        reference=record.reference_phrase,
                        retrieved_contexts=contexts,
                    )
                ).value
            )
        count = float(len(answerable))
        return {name: value / count for name, value in totals.items()}

    return asyncio.run(_run())
