from __future__ import annotations

from typing import Protocol

from harbor_rag.guardrails import UNKNOWN_RESPONSE
from harbor_rag.prompts import user_message
from harbor_rag.textutil import (
    MONEY_WORDS,
    address_keys,
    content_token_set,
    split_sentences,
    tokens,
)


class ChatProvider(Protocol):
    def generate(self, *, system: str, question: str, contexts: list[str]) -> str: ...


class ExtractiveProvider:
    """Quote the best matching sentences. Used for offline runs and tests."""

    def __init__(self, idf: dict[str, float] | None = None) -> None:
        self.idf = dict(idf or {})

    def generate(self, *, system: str, question: str, contexts: list[str]) -> str:
        del system
        keys = address_keys(question)
        usable = list(contexts)
        if keys:
            usable = [context for context in contexts if keys <= content_token_set(context)]
            if not usable:
                return UNKNOWN_RESPONSE
        question_tokens = content_token_set(question)
        min_shared = 1 if keys else 2
        scored: list[tuple[float, int, str]] = []
        for index, context in enumerate(usable):
            for sentence in split_sentences(context):
                shared = question_tokens & content_token_set(sentence)
                if len(shared) < min_shared:
                    continue
                weight = sum(self.idf.get(token, 1.0) for token in shared)
                scored.append((weight, index, sentence))
        if not scored:
            return UNKNOWN_RESPONSE
        scored.sort(key=lambda item: item[0], reverse=True)
        best_index = scored[0][1]
        chosen: list[str] = []
        for _weight, index, sentence in scored:
            if index != best_index or sentence in chosen:
                continue
            chosen.append(sentence)
            if len(chosen) == 4:
                break
        if question_tokens & MONEY_WORDS:
            for sentence in split_sentences(usable[best_index]):
                if "$" in sentence and sentence not in chosen:
                    chosen.append(sentence)
        return " ".join(chosen)


class OpenAIProvider:
    def __init__(self, client: object, model: str) -> None:
        self._client = client
        self.model = model

    def generate(self, *, system: str, question: str, contexts: list[str]) -> str:
        response = self._client.chat.completions.create(  # type: ignore[attr-defined]
            model=self.model,
            temperature=0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_message(question, contexts)},
            ],
        )
        content = response.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            return UNKNOWN_RESPONSE
        return content.strip()


class AnthropicProvider:
    def __init__(self, client: object, model: str) -> None:
        self._client = client
        self.model = model

    def generate(self, *, system: str, question: str, contexts: list[str]) -> str:
        response = self._client.messages.create(  # type: ignore[attr-defined]
            model=self.model,
            max_tokens=500,
            temperature=0,
            system=system,
            messages=[{"role": "user", "content": user_message(question, contexts)}],
        )
        blocks = getattr(response, "content", None)
        if not blocks:
            return UNKNOWN_RESPONSE
        text = getattr(blocks[0], "text", None)
        if not isinstance(text, str) or not text.strip():
            return UNKNOWN_RESPONSE
        return text.strip()


def missing_corpus_tokens(question: str, vocabulary: set[str]) -> set[str]:
    return {token for token in tokens(question) if token not in vocabulary}
