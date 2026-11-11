from __future__ import annotations

from pathlib import Path

from harbor_rag.types import Document


def load_corpus(directory: Path) -> list[Document]:
    paths = sorted(path for path in directory.glob("*.md") if path.is_file())
    if not paths:
        raise FileNotFoundError(f"No markdown documents in {directory}")
    return [_parse(path) for path in paths]


def _parse(path: Path) -> Document:
    raw = path.read_text(encoding="utf-8").strip()
    lines = raw.splitlines()
    if not lines or not lines[0].startswith("# "):
        raise ValueError(f"{path} must start with a markdown H1 title")
    title = lines[0][2:].strip()
    body = "\n".join(lines[1:]).strip()
    if not title or not body:
        raise ValueError(f"{path} needs a title and a body")
    return Document(doc_id=path.stem, title=title, body=body)
