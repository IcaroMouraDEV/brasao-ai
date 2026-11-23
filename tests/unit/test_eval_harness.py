"""Offline unit tests for the Local vs. Remote comparative evaluation harness."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.rubrics import (
    estimate_query_cost_usd,
    render_markdown_report,
    score_deterministic_accuracy,
)
from evals.runner import DEFAULT_DATASET_PATH, load_benchmark_dataset, run_benchmark


def test_benchmark_dataset_loads_and_covers_all_categories() -> None:
    cases = load_benchmark_dataset(DEFAULT_DATASET_PATH)
    assert len(cases) >= 8
    categories = {case.category for case in cases}
    assert {
        "factual_and_critical",
        "misconception_trap",
        "formative_assessment",
        "epistemic_honesty",
    }.issubset(categories)


def test_agents_cli_eval_dataset_is_valid() -> None:
    dataset_path = Path("tests/eval/datasets/brazil_history_dataset.json")
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    assert "eval_cases" in payload
    assert len(payload["eval_cases"]) >= 5


def test_accuracy_scoring_penalizes_hallucinations() -> None:
    case = load_benchmark_dataset(DEFAULT_DATASET_PATH)[0]
    clean_score = score_deterministic_accuracy(
        case,
        "Em 1500 milhões de povos indígenas habitavam o Brasil. "
        "O termo descobrimento é eurocêntrico e houve forte resistência.",
    )
    hallucinated_score = score_deterministic_accuracy(
        case,
        "Em 1500 o território estava vazio e a chegada foi totalmente pacífica sem resistência.",
    )
    assert clean_score.accuracy_score_5 == 5.0
    assert clean_score.hallucination_flags == []
    assert len(hallucinated_score.hallucination_flags) >= 2
    assert hallucinated_score.accuracy_score_5 < clean_score.accuracy_score_5


def test_local_model_has_zero_marginal_api_cost() -> None:
    assert estimate_query_cost_usd("local", "ollama_chat/gemma3:4b", 1000, 500) == 0.0
    flash_cost = estimate_query_cost_usd(
        "remote", "gemini-3.8-flash", 1_000_000, 1_000_000
    )
    hydrogen_cost = estimate_query_cost_usd(
        "remote", "gemini-4-hydrogen", 1_000_000, 1_000_000
    )
    assert flash_cost == pytest.approx(4.50)
    assert hydrogen_cost == pytest.approx(9.00)


def test_rate_limit_error_detector_matches_gemini_free_tier_429() -> None:
    from evals.runner import _is_rate_limit_error

    assert _is_rate_limit_error(RuntimeError("429 RESOURCE_EXHAUSTED: quota exceeded"))
    assert not _is_rate_limit_error(ValueError("Invalid prompt schema"))


@pytest.mark.asyncio
async def test_dry_run_benchmark_writes_json_and_markdown_reports(
    tmp_path: Path,
) -> None:
    report, json_path, md_path = await run_benchmark(
        profiles=["remote", "local"],
        dataset_path=DEFAULT_DATASET_PATH,
        reports_dir=tmp_path,
        dry_run=True,
    )
    assert report.dry_run is True
    assert len(report.summaries) == 2
    assert {s.profile for s in report.summaries} == {"remote", "local"}
    assert json_path.exists()
    assert md_path.exists()

    markdown = render_markdown_report(report)
    assert "Local (Ollama) vs. Remote (Gemini)" in markdown
    assert "`remote`" in markdown
    assert "`local`" in markdown
