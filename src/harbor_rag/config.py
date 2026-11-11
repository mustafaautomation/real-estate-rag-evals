from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

OFFLINE_METRICS: tuple[str, ...] = (
    "hit_rate",
    "mrr",
    "non_llm_context_recall",
    "string_presence",
    "refusal_correctness",
    "grounded_amounts",
)
LIVE_METRICS: tuple[str, ...] = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)


@dataclass(frozen=True)
class AppConfig:
    retrieval_k: int
    chunk_max_chars: int
    chunk_overlap: int
    embedding_dim: int
    live_model: str
    live_embedding_model: str
    offline_thresholds: dict[str, float]
    live_thresholds: dict[str, float]


def load_config(path: Path) -> AppConfig:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must be a mapping")
    thresholds = loaded.get("thresholds")
    if not isinstance(thresholds, dict):
        raise ValueError("thresholds must be a mapping")
    offline = _threshold_block(thresholds.get("offline"), OFFLINE_METRICS, "offline")
    live = _threshold_block(thresholds.get("live"), LIVE_METRICS, "live")
    retrieval_k = _positive_int(loaded.get("retrieval_k"), "retrieval_k")
    chunk_max_chars = _positive_int(loaded.get("chunk_max_chars"), "chunk_max_chars")
    chunk_overlap = loaded.get("chunk_overlap")
    if not isinstance(chunk_overlap, int) or isinstance(chunk_overlap, bool):
        raise ValueError("chunk_overlap must be an integer")
    if chunk_overlap < 0 or chunk_overlap >= chunk_max_chars:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_max_chars")
    embedding_dim = _positive_int(loaded.get("embedding_dim"), "embedding_dim")
    live_model = loaded.get("live_model")
    live_embedding_model = loaded.get("live_embedding_model")
    if not isinstance(live_model, str) or not live_model:
        raise ValueError("live_model must be a non-empty string")
    if not isinstance(live_embedding_model, str) or not live_embedding_model:
        raise ValueError("live_embedding_model must be a non-empty string")
    return AppConfig(
        retrieval_k=retrieval_k,
        chunk_max_chars=chunk_max_chars,
        chunk_overlap=chunk_overlap,
        embedding_dim=embedding_dim,
        live_model=live_model,
        live_embedding_model=live_embedding_model,
        offline_thresholds=offline,
        live_thresholds=live,
    )


def _threshold_block(raw: object, expected: tuple[str, ...], label: str) -> dict[str, float]:
    if not isinstance(raw, dict):
        raise ValueError(f"thresholds.{label} must be a mapping")
    missing = [name for name in expected if name not in raw]
    extra = [str(name) for name in raw if str(name) not in expected]
    if missing or extra:
        raise ValueError(f"thresholds.{label} keys must be {list(expected)}")
    parsed: dict[str, float] = {}
    for name in expected:
        value = raw[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"thresholds.{label}.{name} must be a number")
        number = float(value)
        if not 0.0 <= number <= 1.0:
            raise ValueError(f"thresholds.{label}.{name} must be between 0 and 1")
        parsed[name] = number
    return parsed


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value
