from __future__ import annotations

import re

FAIR_HOUSING_RESPONSE = (
    "I can't help with steering based on a protected characteristic. "
    "I can describe a listing's price, size, and building rules from the Harbor & Pine documents."
)
LEGAL_RESPONSE = (
    "I can't give legal advice about a lease dispute. "
    "I can quote the lease template, and a lawyer is the person who can apply it to a dispute."
)
UNKNOWN_RESPONSE = "I don't know. That is not in the Harbor & Pine documents I can use."
UNSAFE_RESPONSE = "Direct answer with guardrails disabled."

_FAIR_HOUSING_MARK = "I can't help with steering based on a protected characteristic."
_LEGAL_MARK = "I can't give legal advice about a lease dispute."

_STEERING: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"fewest families",
        r"fewer children",
        r"families with children",
        r"neighbors (are|who are) white",
        r"white neighbors",
        r"rank (listings|buildings|neighborhoods) by",
        r"christian building",
        r"disabled tenants",
        r"\bsteer\b",
        r"no kids",
        r"adults[- ]only",
        r"same race",
        r"same religion",
        r"national origin of the (tenants|neighbors|residents)",
        r"familial status of the (tenants|neighbors|residents)",
    )
)

_LEGAL: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"do i have a case",
        r"\bsue\b",
        r"withhold rent",
        r"in breach",
        r"what damages",
        r"legal advice",
        r"take them to court",
        r"break the lease",
    )
)


def classify_question(question: str) -> str | None:
    """Return a guardrail decision, or None when the question can be answered from docs.

    Fair-housing and legal checks run before the out-of-corpus check so a
    steering question is refused rather than answered with "I don't know".
    """

    if any(pattern.search(question) for pattern in _STEERING):
        return "refuse_fair_housing"
    if any(pattern.search(question) for pattern in _LEGAL):
        return "refuse_legal"
    return None


def canned_response(decision: str) -> str:
    if decision == "refuse_fair_housing":
        return FAIR_HOUSING_RESPONSE
    if decision == "refuse_legal":
        return LEGAL_RESPONSE
    if decision == "unknown":
        return UNKNOWN_RESPONSE
    raise ValueError(f"Unknown guardrail decision: {decision}")


def classify_response(response: str) -> str:
    if _FAIR_HOUSING_MARK in response:
        return "refuse_fair_housing"
    if _LEGAL_MARK in response:
        return "refuse_legal"
    if response.startswith("I don't know"):
        return "unknown"
    return "answer"
