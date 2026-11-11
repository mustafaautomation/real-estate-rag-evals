from __future__ import annotations

from harbor_rag.types import Chunk, Document


def chunk_document(document: Document, max_chars: int, overlap: int) -> list[Chunk]:
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    if overlap < 0 or overlap >= max_chars:
        raise ValueError("overlap must be >= 0 and smaller than max_chars")
    paragraphs = [part.strip() for part in document.body.split("\n\n") if part.strip()]
    pieces: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            if current:
                pieces.append(current)
                current = ""
            pieces.extend(_window(paragraph, max_chars, overlap))
            continue
        candidate = paragraph if not current else f"{current}\n\n{paragraph}"
        if len(candidate) <= max_chars:
            current = candidate
            continue
        pieces.append(current)
        tail = current[-overlap:].strip()
        current = f"{tail}\n\n{paragraph}".strip() if tail else paragraph
    if current:
        pieces.append(current)
    chunks: list[Chunk] = []
    for index, piece in enumerate(pieces):
        text = f"{document.title}.\n\n{piece}"
        chunks.append(
            Chunk(
                chunk_id=f"{document.doc_id}-{index}",
                doc_id=document.doc_id,
                title=document.title,
                text=text,
            )
        )
    return chunks


def chunk_corpus(documents: list[Document], max_chars: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(chunk_document(document, max_chars, overlap))
    return chunks


def _window(text: str, max_chars: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        next_start = end - overlap
        start = end if next_start <= start else next_start
    return chunks
