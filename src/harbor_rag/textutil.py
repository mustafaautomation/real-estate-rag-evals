from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9$]+")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_ADDRESS = re.compile(r"\b(\d+)\s+([a-z]+(?:\s+[a-z]+){0,3})\b")

STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "of",
        "to",
        "for",
        "in",
        "on",
        "at",
        "by",
        "with",
        "from",
        "is",
        "are",
        "was",
        "were",
        "be",
        "this",
        "that",
        "it",
        "its",
        "as",
        "if",
        "do",
        "does",
        "did",
        "what",
        "which",
        "who",
        "whom",
        "how",
        "when",
        "where",
        "why",
        "can",
        "i",
        "me",
        "my",
        "we",
        "our",
        "you",
        "your",
        "they",
        "their",
        "not",
        "no",
        "yes",
        "about",
        "into",
        "over",
        "under",
        "than",
        "then",
        "there",
        "here",
        "still",
        "any",
        "each",
        "please",
        "tell",
        "should",
        "would",
        "could",
        "have",
        "has",
        "had",
        "because",
        "while",
        "also",
        "only",
        "just",
        "up",
        "down",
        "out",
        "so",
        "them",
    }
)

GENERIC_SUFFIXES: frozenset[str] = frozenset(
    {"street", "avenue", "court", "row", "walk", "boulevard", "road", "lane"}
)
DIRECTION_WORDS: frozenset[str] = frozenset({"north", "south", "east", "west"})
PLACE_WORDS: frozenset[str] = frozenset({"pier", "annex", "house", "flats", "lofts"})
MONEY_WORDS: frozenset[str] = frozenset({"rent", "price", "fee", "deposit", "cost"})


def normalize(text: str) -> str:
    lowered = text.lower()
    return re.sub(r"(\d),(\d)", r"\1\2", lowered)


def stem(token: str) -> str:
    if any(character.isdigit() for character in token) or "$" in token:
        return token
    for suffix in ("ing", "ers", "er", "ed", "es", "s"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def tokens(text: str) -> list[str]:
    out: list[str] = []
    for raw in _TOKEN.findall(normalize(text)):
        if raw in STOPWORDS:
            continue
        out.append(stem(raw))
    return out


def content_token_set(text: str) -> set[str]:
    return set(tokens(text))


def split_sentences(text: str) -> list[str]:
    parts = _SENTENCE.split(text.strip())
    return [part.strip() for part in parts if part.strip()]


_NON_NAME = frozenset(
    {"hours", "hour", "days", "day", "months", "month", "am", "pm", "percent", "dollars"}
)


def address_keys(question: str) -> set[str]:
    """Street number plus name tokens, when the question names a street address."""

    keys: set[str] = set()
    for match in _ADDRESS.finditer(normalize(question)):
        number, rest = match.group(1), match.group(2).split()
        suffix_at = next(
            (
                index
                for index, word in enumerate(rest)
                if word in GENERIC_SUFFIXES or word in PLACE_WORDS
            ),
            None,
        )
        if suffix_at is None or rest[0] in _NON_NAME:
            continue
        keys.add(number)
        for word in rest[: suffix_at + 1]:
            if word in GENERIC_SUFFIXES or word in DIRECTION_WORDS or word in STOPWORDS:
                continue
            keys.add(stem(word))
    return keys


def dollar_amounts(text: str) -> set[str]:
    return set(re.findall(r"\$\d[\d,]*", text))
