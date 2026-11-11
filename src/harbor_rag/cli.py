from __future__ import annotations

import argparse
import os

from harbor_rag.embeddings import Embedder
from harbor_rag.eval.runner import load_app
from harbor_rag.pipeline import Pipeline
from harbor_rag.providers import AnthropicProvider, ChatProvider, ExtractiveProvider, OpenAIProvider
from harbor_rag.retrieval import FaissIndex, PgVectorRetriever, Retriever
from harbor_rag.types import ScoredChunk, Variant


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="harbor-ask")
    parser.add_argument("question")
    parser.add_argument("--provider", choices=("fake", "openai", "anthropic"), default="fake")
    parser.add_argument(
        "--variant", choices=tuple(item.value for item in Variant), default="baseline"
    )
    parser.add_argument("--backend", choices=("faiss", "pgvector"), default="faiss")
    parser.add_argument("--dsn", default=os.environ.get("HARBOR_PG_DSN", ""))
    args = parser.parse_args(argv)
    _root, config, _questions, chunks, embedder = load_app()
    variant = Variant(args.variant)
    index = FaissIndex(chunks, embedder)
    provider = _provider(args.provider, config.live_model, embedder.idf)
    retriever: Retriever | None = None
    if args.backend == "pgvector":
        if not args.dsn:
            raise SystemExit("HARBOR_PG_DSN or --dsn is required for the pgvector backend")
        store = PgVectorRetriever(args.dsn, worst=variant is Variant.BAD_RETRIEVER)
        store.build(chunks, embedder)
        retriever = _PgRetriever(store, embedder)
    pipeline = Pipeline(
        index,
        embedder,
        k=config.retrieval_k,
        variant=variant,
        provider=provider,
        retriever=retriever,
    )
    result = pipeline.ask(args.question)
    print(result.response)


class _PgRetriever:
    def __init__(self, store: PgVectorRetriever, embedder: Embedder) -> None:
        self._store = store
        self._embedder = embedder

    def retrieve(self, query: str, k: int) -> list[ScoredChunk]:
        return self._store.retrieve(query, k, self._embedder)


def _provider(name: str, live_model: str, idf: dict[str, float]) -> ChatProvider:
    if name == "fake":
        return ExtractiveProvider(idf)
    if name == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY is not set")
        from openai import OpenAI

        return OpenAIProvider(OpenAI(), model=live_model)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set")
    from anthropic import Anthropic

    model = os.environ.get("HARBOR_ANTHROPIC_MODEL", "claude-3-5-haiku-latest")
    return AnthropicProvider(Anthropic(), model=model)


if __name__ == "__main__":
    main()
