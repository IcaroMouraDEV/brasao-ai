"""Offline deterministic unit tests for curated Brazilian History tools."""

from __future__ import annotations

import re

import pytest

from app.tools import (
    build_quiz_blueprint,
    lookup_timeline,
    search_historical_corpus,
)
from app.tools.corpus import load_corpus
from app.tools.rag import clear_rag_index_cache


@pytest.fixture(autouse=True)
def _force_offline_lexical_for_tool_unit_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_ENABLE_DENSE", "false")
    clear_rag_index_cache()


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

BNCC_CODE_PATTERN = re.compile(r"^(EF0[6-9]HI\d{2}|EM13CHS\d{3})$")


def test_curated_corpus_entries_match_required_schema() -> None:
    corpus = load_corpus()
    assert len(corpus) >= 20
    ids = set()
    for entry in corpus:
        assert REQUIRED_CORPUS_FIELDS.issubset(entry.keys())
        assert entry["id"] not in ids
        ids.add(entry["id"])
        assert isinstance(entry["start_year"], int)
        assert isinstance(entry["end_year"], int)
        assert entry["start_year"] <= entry["end_year"]
        assert entry["title"].strip()
        assert entry["summary"].strip()
        assert len(entry["bncc_codes"]) >= 1
        for code in entry["bncc_codes"]:
            assert BNCC_CODE_PATTERN.match(code), (
                f"Invalid BNCC code {code!r} in {entry['id']}"
            )
        assert len(entry["key_facts"]) >= 3
        assert len(entry["primary_actors"]) >= 3
        assert len(entry["critical_perspectives"]) >= 2
        assert len(entry["common_misconceptions"]) >= 2
        assert len(entry["recommended_sources"]) >= 2


def test_curated_corpus_covers_full_bncc_curriculum_periods() -> None:
    corpus = load_corpus()
    ids = {entry["id"] for entry in corpus}
    expected_core_topics = {
        "povos_originarios_pre_coloniais",
        "brasil_colonial_povos_indigenas_1500",
        "ciclo_do_acucar_sociedade_colonial",
        "bandeirantes_e_expansao_territorial",
        "quilombo_dos_palmares_resistencia",
        "ciclo_do_ouro_mineracao_colonial",
        "inconfidencia_mineira_1789",
        "conjuracao_baiana_1798",
        "revolucao_pernambucana_1817",
        "independencia_do_brasil_1822",
        "primeiro_reinado_1822_1831",
        "periodo_regencial_e_revoltas_1831_1840",
        "segundo_reinado_politica_economia_1840_1889",
        "guerra_do_paraguai_1864_1870",
        "ciclo_do_cafe_e_imigracao_1850_1930",
        "abolicao_escravidao_1888",
        "proclamacao_da_republica_1889",
        "semana_de_arte_moderna_1922",
        "tenentismo_e_crise_primeira_republica_1922_1930",
        "era_vargas_1930_1945",
        "republica_populista_1946_1964",
        "ditadura_militar_e_redemocratizacao_1964_1988",
        "nova_republica_constituicao_1988_plano_real",
    }
    assert expected_core_topics.issubset(ids)


@pytest.mark.parametrize(
    ("query", "expected_id"),
    [
        ("abolicao lei aurea", "abolicao_escravidao_1888"),
        ("troncos linguisticos tupi sambaquis", "povos_originarios_pre_coloniais"),
        ("ciclo do acucar plantation engenho", "ciclo_do_acucar_sociedade_colonial"),
        ("bandeirantes tratado de madri", "bandeirantes_e_expansao_territorial"),
        ("ciclo do ouro barroco mineiro derrama", "ciclo_do_ouro_mineracao_colonial"),
        ("conjuracao baiana revolta dos alfaiates", "conjuracao_baiana_1798"),
        ("revolucao pernambucana 1817", "revolucao_pernambucana_1817"),
        ("primeiro reinado poder moderador 1824", "primeiro_reinado_1822_1831"),
        (
            "revolta dos males cabanagem balaiada",
            "periodo_regencial_e_revoltas_1831_1840",
        ),
        (
            "segundo reinado parlamentarismo lei de terras",
            "segundo_reinado_politica_economia_1840_1889",
        ),
        ("guerra do paraguai triplice alianca", "guerra_do_paraguai_1864_1870"),
        (
            "ciclo do cafe imigracao branqueamento",
            "ciclo_do_cafe_e_imigracao_1850_1930",
        ),
        ("semana de arte moderna 1922 antropofagico", "semana_de_arte_moderna_1922"),
        (
            "tenentismo coluna prestes 18 do forte",
            "tenentismo_e_crise_primeira_republica_1922_1930",
        ),
        (
            "republica populista brasilia reformas de base",
            "republica_populista_1946_1964",
        ),
        (
            "nova republica constituicao 1988 plano real",
            "nova_republica_constituicao_1988_plano_real",
        ),
    ],
)
def test_search_historical_corpus_is_accent_insensitive_across_expanded_topics(
    query: str, expected_id: str
) -> None:
    result = search_historical_corpus(query)
    assert result["status"] == "ok"
    assert result["entries"][0]["id"] == expected_id
    assert result["entries"][0]["source_id"]
    assert result["entries"][0]["document"]
    assert len(result["chunks"]) >= 1
    assert result["chunks"][0]["source_id"]
    assert result["chunks"][0]["document"]
    assert isinstance(result["chunks"][0]["bncc_codes"], list)


def test_search_historical_corpus_returns_not_found_for_fictional_event() -> None:
    result = search_historical_corpus("Tratado de Guanabara 1742 Imperador do Japao")
    assert result["status"] == "not_found"
    assert result["entries"] == []


def test_lookup_timeline_orders_chronologically_and_filters_years() -> None:
    result = lookup_timeline(start_year=1800, end_year=1900)
    assert result["status"] == "ok"
    assert result["count"] >= 7
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


def test_build_quiz_blueprint_supports_new_topics() -> None:
    blueprint = build_quiz_blueprint(
        "Conjuracao Baiana", level="fundamental_2", num_questions=2
    )
    assert blueprint["status"] == "ok"
    assert blueprint["topic_id"] == "conjuracao_baiana_1798"
    assert blueprint["level"] == "fundamental_2"
    assert blueprint["num_questions"] == 2
    assert "EF08HI05" in blueprint["bncc_codes"]


def test_build_quiz_blueprint_rejects_unknown_level() -> None:
    result = build_quiz_blueprint("Era Vargas", level="pos_doutorado")
    assert result["status"] == "error"


def test_tools_coerce_stringified_integer_arguments() -> None:
    search_result = search_historical_corpus("Abolição 1888", max_results="2")  # type: ignore[arg-type]
    assert search_result["status"] == "ok"
    assert search_result["count"] <= 2

    timeline_result = lookup_timeline(start_year="1800", end_year="1900")  # type: ignore[arg-type]
    assert timeline_result["status"] == "ok"
    assert timeline_result["count"] >= 7

    quiz_result = build_quiz_blueprint(
        "Era Vargas",
        level="ensino_medio",
        num_questions="2",  # type: ignore[arg-type]
    )
    assert quiz_result["status"] == "ok"
    assert quiz_result["num_questions"] == 2


def test_build_quiz_blueprint_handles_reference_chunk_without_key_facts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.tools.quiz as quiz_module

    monkeypatch.setattr(
        quiz_module,
        "search_historical_corpus",
        lambda topic, max_results=1: {
            "status": "ok",
            "entries": [
                {
                    "id": "bncc:EF07HI10",
                    "title": "BNCC EF07HI10",
                    "period": "Brasil Colônia",
                    "bncc_codes": ["EF07HI10"],
                    "summary": "Analisar aspectos sociais e culturais dos povos indígenas.",
                }
            ],
        },
    )
    result = build_quiz_blueprint(
        "BNCC EF07HI10", level="fundamental_2", num_questions=2
    )
    assert result["status"] == "ok"
    assert result["grounded_facts"] == [
        "Analisar aspectos sociais e culturais dos povos indígenas."
    ]
    assert result["critical_perspectives"] == []
    assert result["misconceptions_to_test"] == []
