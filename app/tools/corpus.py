"""Curated BNCC-aligned Brazilian History corpus loader and retrieval tool."""

from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any

CORPUS_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "brazil_history_corpus.json"
)

STOPWORDS = frozenset(
    {
        "de",
        "da",
        "do",
        "das",
        "dos",
        "em",
        "no",
        "na",
        "nos",
        "nas",
        "os",
        "as",
        "um",
        "uma",
        "uns",
        "umas",
        "por",
        "para",
        "com",
        "sem",
        "que",
        "qual",
        "quais",
        "como",
        "sobre",
        "entre",
        "pelo",
        "pela",
        "the",
        "and",
        "for",
        "from",
        "with",
        "about",
    }
)


def normalize_text(text: str) -> str:
    """Lowercase and strip Portuguese diacritics for accent-insensitive search."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def tokenize_query(text: str) -> list[str]:
    """Split normalized text into meaningful alphanumeric search tokens."""
    normalized = normalize_text(text)
    raw_tokens = re.findall(r"[a-z0-9-]+", normalized)
    return [tok for tok in raw_tokens if len(tok) >= 3 and tok not in STOPWORDS]


@lru_cache(maxsize=1)
def load_corpus() -> tuple[dict[str, Any], ...]:
    """Load the immutable curated Brazilian History corpus from disk."""
    raw = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    return tuple(raw)


def _score_entry(entry: dict[str, Any], tokens: list[str]) -> int:
    title = normalize_text(entry["title"])
    period = normalize_text(entry["period"])
    body = normalize_text(
        " ".join(
            [
                entry["summary"],
                " ".join(entry["key_facts"]),
                " ".join(entry["primary_actors"]),
                " ".join(entry["bncc_codes"]),
            ]
        )
    )

    score = 0
    matched_tokens = 0
    title_matches = 0
    for token in tokens:
        matched = False
        if token in title:
            score += 5
            title_matches += 1
            matched = True
        if token in period:
            score += 3
            matched = True
        if token in body:
            score += 2
            matched = True
        if matched:
            matched_tokens += 1

    # Avoid single-token false positives on longer queries (e.g., fictional events
    # sharing one generic word such as "tratado" with an unrelated real entry).
    if len(tokens) >= 4:
        min_required = (
            max(2, len(tokens) // 3)
            if title_matches > 0
            else max(2, (len(tokens) + 1) // 2)
        )
        if matched_tokens < min_required:
            return 0

    return score


def search_historical_corpus(query: str, max_results: int = 3) -> dict[str, Any]:
    """Search the curated BNCC-aligned Brazilian History knowledge base.

    Args:
        query: Historical topic, event, actor, year, or BNCC code in Portuguese or English.
        max_results: Maximum number of matching entries to return (1 to 5).

    Returns:
        A JSON-serializable dict with `status`, `query`, and matching historical `entries`.
    """
    cleaned_query = query.strip()
    if not cleaned_query:
        return {"status": "error", "message": "Query must not be empty.", "entries": []}

    limit = max(1, min(max_results, 5))
    tokens = tokenize_query(cleaned_query)
    if not tokens:
        return {
            "status": "not_found",
            "query": cleaned_query,
            "message": "No meaningful historical search terms were found in the query.",
            "entries": [],
        }

    scored: list[tuple[int, dict[str, Any]]] = []
    for entry in load_corpus():
        score = _score_entry(entry, tokens)
        if score > 0:
            scored.append((score, entry))

    scored.sort(key=lambda item: item[0], reverse=True)
    matches = [entry for _, entry in scored[:limit]]

    if not matches:
        return {
            "status": "not_found",
            "query": cleaned_query,
            "message": "No curated Brazilian History entries matched the query.",
            "entries": [],
        }

    return {
        "status": "ok",
        "query": cleaned_query,
        "count": len(matches),
        "entries": matches,
    }
