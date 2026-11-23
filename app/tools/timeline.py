"""Chronological timeline lookup tool over the curated Brazilian History corpus."""

from __future__ import annotations

from typing import Any

from app.tools.corpus import load_corpus, normalize_text


def _coerce_optional_int(value: int | str | float | None) -> int | None:
    """Coerce optional stringified year arguments emitted by smaller local LLMs."""
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def lookup_timeline(
    start_year: int | None = None,
    end_year: int | None = None,
    period: str | None = None,
) -> dict[str, Any]:
    """Retrieve Brazilian History events ordered chronologically by year range or period.

    Args:
        start_year: Optional lower bound year (inclusive, e.g., 1800).
        end_year: Optional upper bound year (inclusive, e.g., 1900).
        period: Optional historical period filter (e.g., 'Brasil Colônia', 'Brasil Império',
            'Brasil República', 'Brasil Contemporâneo').

    Returns:
        A JSON-serializable dict with `status` and chronologically sorted `events`.
    """
    resolved_start = _coerce_optional_int(start_year)
    resolved_end = _coerce_optional_int(end_year)
    if (
        resolved_start is not None
        and resolved_end is not None
        and resolved_start > resolved_end
    ):
        return {
            "status": "error",
            "message": "start_year must be less than or equal to end_year.",
            "events": [],
        }

    normalized_period = normalize_text(period.strip()) if period else None
    results: list[dict[str, Any]] = []

    for entry in load_corpus():
        entry_start = int(entry["start_year"])
        entry_end = int(entry["end_year"])

        if resolved_start is not None and entry_end < resolved_start:
            continue
        if resolved_end is not None and entry_start > resolved_end:
            continue
        if normalized_period and normalized_period not in normalize_text(
            entry["period"]
        ):
            continue

        results.append(
            {
                "id": entry["id"],
                "title": entry["title"],
                "period": entry["period"],
                "start_year": entry_start,
                "end_year": entry_end,
                "bncc_codes": entry["bncc_codes"],
                "summary": entry["summary"],
            }
        )

    results.sort(key=lambda item: (item["start_year"], item["end_year"]))
    if not results:
        return {
            "status": "not_found",
            "message": "No historical events matched the requested timeline filters.",
            "events": [],
        }

    return {"status": "ok", "count": len(results), "events": results}
