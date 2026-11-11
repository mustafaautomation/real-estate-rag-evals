from __future__ import annotations

from harbor_rag.embeddings import HashedTfidfEmbedder
from harbor_rag.guardrails import UNSAFE_RESPONSE, canned_response, classify_question
from harbor_rag.prompts import system_prompt
from harbor_rag.providers import ChatProvider, ExtractiveProvider, missing_corpus_tokens
from harbor_rag.retrieval import BottomRetriever, FaissIndex, Retriever
from harbor_rag.types import AskResult, Variant


class Pipeline:
    def __init__(
        self,
        index: FaissIndex,
        embedder: HashedTfidfEmbedder,
        *,
        k: int,
        variant: Variant = Variant.BASELINE,
        provider: ChatProvider | None = None,
        retriever: Retriever | None = None,
    ) -> None:
        self.index = index
        self.embedder = embedder
        self.k = k
        self.variant = variant
        self.provider = provider or ExtractiveProvider(embedder.idf)
        if retriever is not None:
            self._retriever = retriever
        elif variant is Variant.BAD_RETRIEVER:
            self._retriever = BottomRetriever(index)
        else:
            self._retriever = index
        self._vocabulary = set(embedder.idf)

    def ask(self, question: str) -> AskResult:
        prompt = system_prompt(self.variant)
        decision = classify_question(question)
        if decision is not None and self.variant is Variant.NO_GUARDRAIL:
            return self._finish(question, UNSAFE_RESPONSE, (), prompt, "no_guardrail")
        if decision is not None:
            return self._finish(question, canned_response(decision), (), prompt, decision)
        missing = missing_corpus_tokens(question, self._vocabulary)
        if missing:
            return self._finish(question, canned_response("unknown"), (), prompt, "unknown")
        retrieved = self._retriever.retrieve(question, self.k)
        contexts = [item.chunk.text for item in retrieved]
        doc_ids = tuple(item.chunk.doc_id for item in retrieved)
        response = self.provider.generate(system=prompt, question=question, contexts=contexts)
        return self._finish(question, response, doc_ids, prompt, None, contexts)

    def _finish(
        self,
        question: str,
        response: str,
        doc_ids: tuple[str, ...],
        prompt: str,
        guardrail: str | None,
        contexts: list[str] | None = None,
    ) -> AskResult:
        stored = tuple(contexts or ())
        return AskResult(
            question=question,
            response=response,
            retrieved_contexts=stored,
            retrieved_doc_ids=doc_ids,
            system_prompt=prompt,
            guardrail=guardrail,
        )
