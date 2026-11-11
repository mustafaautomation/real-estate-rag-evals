from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GateResult:
    scores: dict[str, float]
    thresholds: dict[str, float]
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.failures


def apply_gate(scores: dict[str, float], thresholds: dict[str, float]) -> GateResult:
    failures: list[str] = []
    for name, threshold in thresholds.items():
        score = scores.get(name)
        if score is None:
            failures.append(f"{name} is missing")
            continue
        if score < threshold:
            failures.append(f"{name} {score:.4f} is below {threshold:.4f}")
    return GateResult(scores=dict(scores), thresholds=dict(thresholds), failures=tuple(failures))
