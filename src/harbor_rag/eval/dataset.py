from __future__ import annotations

import json
from pathlib import Path

from harbor_rag.types import EvalItem, ExpectedBehavior


def load_questions(path: Path) -> list[EvalItem]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict) or not isinstance(loaded.get("questions"), list):
        raise ValueError(f"{path} must be an object with a questions list")
    items: list[EvalItem] = []
    seen: set[str] = set()
    for raw in loaded["questions"]:
        if not isinstance(raw, dict):
            raise ValueError("each question must be an object")
        item_id = _string(raw.get("id"), "id")
        if item_id in seen:
            raise ValueError(f"duplicate question id {item_id}")
        seen.add(item_id)
        behavior_raw = _string(raw.get("expected_behavior"), "expected_behavior")
        try:
            behavior = ExpectedBehavior(behavior_raw)
        except ValueError as exc:
            raise ValueError(f"{item_id} has unknown expected_behavior") from exc
        doc_ids_raw = raw.get("reference_doc_ids", [])
        if not isinstance(doc_ids_raw, list) or not all(
            isinstance(doc_id, str) for doc_id in doc_ids_raw
        ):
            raise ValueError(f"{item_id} reference_doc_ids must be a list of strings")
        tags_raw = raw.get("tags", [])
        if not isinstance(tags_raw, list) or not all(isinstance(tag, str) for tag in tags_raw):
            raise ValueError(f"{item_id} tags must be a list of strings")
        items.append(
            EvalItem(
                item_id=item_id,
                user_input=_string(raw.get("user_input"), "user_input"),
                reference=_string(raw.get("reference"), "reference"),
                reference_phrase=_string(raw.get("reference_phrase"), "reference_phrase"),
                reference_doc_ids=tuple(doc_ids_raw),
                expected_behavior=behavior,
                tags=tuple(tags_raw),
            )
        )
    if not 30 <= len(items) <= 50:
        raise ValueError(f"expected 30 to 50 questions, found {len(items)}")
    return items


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value
