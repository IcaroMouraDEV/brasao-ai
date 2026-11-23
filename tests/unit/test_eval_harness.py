"""Offline unit tests for the Local vs. Remote comparative evaluation harness."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pytest

from app.tools.corpus import normalize_text
from evals.rubrics import (
    estimate_query_cost_usd,
    render_markdown_report,
    score_deterministic_accuracy,
)
from evals.runner import DEFAULT_DATASET_PATH, load_benchmark_dataset, run_benchmark

BNCC_CODE_PATTERN = re.compile(r"^(EF0[6-9]HI\d{2}|EM13CHS\d{3})$")


def test_benchmark_dataset_loads_and_covers_all_categories_and_levels() -> None:
    cases = load_benchmark_dataset(DEFAULT_DATASET_PATH)
    assert 20 <= len(cases) <= 30

    case_ids = [case.id for case in cases]
    assert len(case_ids) == len(set(case_ids))

    category_counts = Counter(case.category for case in cases)
    assert set(category_counts.keys()) == {
        "factual_and_critical",
        "misconception_trap",
        "formative_assessment",
        "epistemic_honesty",
    }
    for category, count in category_counts.items():
        assert count >= 4, f"Category {category!r} has too few cases ({count})"

    level_counts = Counter(case.target_level for case in cases)
    assert set(level_counts.keys()) == {"fundamental_2", "ensino_medio", "geral"}
    for level, count in level_counts.items():
        assert count >= 6, f"Level {level!r} has too few cases ({count})"

    for case in cases:
        assert len(case.bncc_codes) >= 1
        for code in case.bncc_codes:
            assert BNCC_CODE_PATTERN.match(code), (
                f"Invalid BNCC code {code!r} in case {case.id}"
            )
        assert len(case.expected_facts) >= 2
        assert len(case.forbidden_hallucinations) >= 2
        # Reference summary must never trigger its own forbidden hallucination traps.
        normalized_ref = normalize_text(case.reference_summary)
        for trap in case.forbidden_hallucinations:
            assert normalize_text(trap) not in normalized_ref, (
                f"Trap {trap!r} matched reference_summary in case {case.id}"
            )


def test_agents_cli_eval_dataset_is_valid() -> None:
    dataset_path = Path("tests/eval/datasets/brazil_history_dataset.json")
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    assert "eval_cases" in payload
    cases = payload["eval_cases"]
    assert len(cases) >= 10

    seen_ids: set[str] = set()
    single_turn_count = 0
    multi_turn_count = 0

    for case in cases:
        case_id = case["eval_case_id"]
        assert case_id not in seen_ids
        seen_ids.add(case_id)
        assert case["reference"]["response"]["role"] == "model"
        assert case["reference"]["response"]["parts"][0]["text"].strip()

        if "prompt" in case:
            single_turn_count += 1
            assert case["prompt"]["role"] == "user"
            assert case["prompt"]["parts"][0]["text"].strip()
        elif "agent_data" in case:
            multi_turn_count += 1
            turns = case["agent_data"]["turns"]
            assert len(turns) >= 1
            events = turns[-1]["events"]
            assert len(events) >= 3
            assert events[-1]["author"] == "user"
            assert events[-1]["content"]["role"] == "user"
        else:
            pytest.fail(f"Eval case {case_id!r} missing both 'prompt' and 'agent_data'")

    assert single_turn_count >= 8
    assert multi_turn_count >= 2


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
    assert all(s.total_cases >= 20 for s in report.summaries)
    assert all(s.hallucination_case_count == 0 for s in report.summaries)
    assert json_path.exists()
    assert md_path.exists()

    markdown = render_markdown_report(report)
    assert "Local (Ollama) vs. Remote (Gemini)" in markdown
    assert "`remote`" in markdown
    assert "`local`" in markdown
