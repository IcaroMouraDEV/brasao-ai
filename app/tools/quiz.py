"""Pedagogical quiz blueprint generator aligned with UN SDG 4 and BNCC."""

from __future__ import annotations

from typing import Any

from app.tools.corpus import search_historical_corpus

SUPPORTED_LEVELS = {
    "fundamental_2": {
        "label": "Ensino Fundamental — Anos Finais (6º ao 9º ano)",
        "focus": "Compreensão de causa e consequência, vocabulário acessível e identificação de fatos centrais.",
    },
    "ensino_medio": {
        "label": "Ensino Médio / Preparação ENEM",
        "focus": "Análise historiográfica, leitura crítica de múltiplas perspectivas e conexões estruturais.",
    },
    "geral": {
        "label": "Público Geral",
        "focus": "Contextualização clara, desconstrução de mitos históricos e reflexão cidadã.",
    },
}


def build_quiz_blueprint(
    topic: str,
    level: str = "ensino_medio",
    num_questions: int = 3,
) -> dict[str, Any]:
    """Build a grounded pedagogical quiz blueprint from the curated Brazilian History corpus.

    Args:
        topic: Historical topic or period to assess (e.g., 'Abolição', 'Era Vargas').
        level: Target education level: 'fundamental_2', 'ensino_medio', or 'geral'.
        num_questions: Number of questions to plan (1 to 5).

    Returns:
        A JSON-serializable dict containing BNCC metadata, grounded facts, misconception traps,
        and rubric instructions for the tutor to draft formative questions in pt-BR.
    """
    normalized_level = level.strip().lower()
    if normalized_level not in SUPPORTED_LEVELS:
        return {
            "status": "error",
            "message": (
                f"Unsupported level '{level}'. Choose from: {', '.join(sorted(SUPPORTED_LEVELS))}."
            ),
        }

    search_result = search_historical_corpus(topic, max_results=1)
    if search_result["status"] != "ok":
        return {
            "status": "not_found",
            "topic": topic,
            "message": "Topic not found in the curated corpus; cannot build a grounded quiz.",
        }

    entry = search_result["entries"][0]
    clamped_questions = max(1, min(num_questions, 5))
    level_meta = SUPPORTED_LEVELS[normalized_level]

    return {
        "status": "ok",
        "topic_id": entry["id"],
        "topic_title": entry["title"],
        "period": entry["period"],
        "bncc_codes": entry["bncc_codes"],
        "level": normalized_level,
        "level_label": level_meta["label"],
        "pedagogical_focus": level_meta["focus"],
        "num_questions": clamped_questions,
        "grounded_facts": entry["key_facts"],
        "critical_perspectives": entry["critical_perspectives"],
        "misconceptions_to_test": entry["common_misconceptions"],
        "format_guidelines": [
            "Elabore questões em português do Brasil com enunciados contextualizados.",
            "Inclua pelo menos uma questão que desafie um mito histórico comum listado em misconceptions_to_test.",
            "Forneça gabarito comentado explicando por que a alternativa correta está certa e o erro das demais.",
        ],
    }
