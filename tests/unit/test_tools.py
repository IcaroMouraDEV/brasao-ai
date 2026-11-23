"""Offline deterministic unit tests for curated Brazilian History tools."""

from __future__ import annotations

from app.tools import (
    build_quiz_blueprint,
    lookup_timeline,
    search_historical_corpus,
)
from app.tools.corpus import load_corpus

REQUIRED_CORPUS_FIELDS = {
    "id",
    "title",
    "period",
    "start_year",
    "end_year",
    "bncc_codes",
    "summary",
    "key_facts",
    "primary_actors",
    "critical_perspectives",
    "common_misconceptions",
    "recommended_sources",
}


def test_curated_corpus_entries_match_required_schema() -> None:
    corpus = load_corpus()
    assert len(corpus) >= 8
    ids = set()
    for entry in corpus:
        assert REQUIRED_CORPUS_FIELDS.issubset(entry.keys())
        assert entry["id"] not in ids
        ids.add(entry["id"])
        assert entry["start_year"] <= entry["end_year"]
        assert len(entry["bncc_codes"]) >= 1
        assert len(entry["key_facts"]) >= 3


def test_search_historical_corpus_is_accent_insensitive() -> None:
    result = search_historical_corpus("abolicao lei aurea")
    assert result["status"] == "ok"
    assert result["entries"][0]["id"] == "abolicao_escravidao_1888"


def test_search_historical_corpus_returns_not_found_for_fictional_event() -> None:
    result = search_historical_corpus("Tratado de Guanabara 1742 Imperador do Japao")
    assert result["status"] == "not_found"
    assert result["entries"] == []


def test_lookup_timeline_orders_chronologically_and_filters_years() -> None:
    result = lookup_timeline(start_year=1800, end_year=1900)
    assert result["status"] == "ok"
    years = [event["start_year"] for event in result["events"]]
    assert years == sorted(years)
    assert all(
        event["end_year"] >= 1800 and event["start_year"] <= 1900
        for event in result["events"]
    )


def test_lookup_timeline_rejects_inverted_year_range() -> None:
    result = lookup_timeline(start_year=1950, end_year=1800)
    assert result["status"] == "error"


def test_build_quiz_blueprint_returns_bncc_aligned_payload() -> None:
    blueprint = build_quiz_blueprint(
        "Era Vargas", level="ensino_medio", num_questions=3
    )
    assert blueprint["status"] == "ok"
    assert blueprint["topic_id"] == "era_vargas_1930_1945"
    assert blueprint["level"] == "ensino_medio"
    assert blueprint["num_questions"] == 3
    assert len(blueprint["misconceptions_to_test"]) >= 1


def test_build_quiz_blueprint_rejects_unknown_level() -> None:
    result = build_quiz_blueprint("Era Vargas", level="pos_doutorado")
    assert result["status"] == "error"
