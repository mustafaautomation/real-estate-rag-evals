from __future__ import annotations

import json
from pathlib import Path

from harbor_rag.eval.metrics import PipelineRecord
from harbor_rag.types import ExpectedBehavior

SNAPSHOT_LABEL = "pipeline_outputs_only_judge_scores_not_recorded"


def snapshot_paths(root: Path) -> tuple[Path, Path]:
    directory = root / "data" / "snapshots"
    return directory / "offline_responses.jsonl", directory / "snapshot_meta.json"


def write_snapshot(root: Path, records: list[PipelineRecord], ragas_version: str) -> None:
    jsonl_path, meta_path = snapshot_paths(root)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(_record_payload(record), sort_keys=True) for record in records]
    jsonl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    meta = {
        "label": SNAPSHOT_LABEL,
        "judge_scores": None,
        "judge_scores_note": (
            "No OPENAI_API_KEY was available when this snapshot was written. "
            "These rows are outputs of the deterministic extractive provider. "
            "They are not LLM-judge scores. Do not treat any field as a RAGAS "
            "faithfulness, relevancy, context precision, or context recall score."
        ),
        "provider": "extractive",
        "embedder": "hashed_tfidf",
        "ragas_version": ragas_version,
        "rows": len(records),
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_snapshot(root: Path) -> list[dict[str, object]]:
    jsonl_path, _meta_path = snapshot_paths(root)
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(
        jsonl_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        loaded = json.loads(line)
        if not isinstance(loaded, dict):
            raise ValueError(f"{jsonl_path}:{line_number} is not an object")
        rows.append(loaded)
    return rows


def snapshot_mismatches(root: Path, records: list[PipelineRecord]) -> list[str]:
    saved = load_snapshot(root)
    current = [_record_payload(record) for record in records]
    if len(saved) != len(current):
        return [f"snapshot has {len(saved)} rows and the rerun has {len(current)}"]
    mismatches: list[str] = []
    for saved_row, current_row in zip(saved, current, strict=True):
        if saved_row != current_row:
            item_id = current_row.get("id")
            mismatches.append(f"{item_id} does not match the committed snapshot")
    return mismatches


def _record_payload(record: PipelineRecord) -> dict[str, object]:
    return {
        "expected_behavior": record.expected_behavior.value
        if isinstance(record.expected_behavior, ExpectedBehavior)
        else str(record.expected_behavior),
        "id": record.item_id,
        "judge_scores": None,
        "judge_scores_label": SNAPSHOT_LABEL,
        "reference_contexts": list(record.reference_contexts),
        "reference_doc_ids": list(record.reference_doc_ids),
        "reference_phrase": record.reference_phrase,
        "response": record.response,
        "retrieved_contexts": list(record.retrieved_contexts),
        "retrieved_doc_ids": list(record.retrieved_doc_ids),
        "user_input": record.user_input,
    }
