"""Offline unit tests for the Local vs. Remote comparative evaluation harness."""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import pytest

from app.config import Settings
from app.tools.corpus import normalize_text
from evals import charts as eval_charts
from evals import runner as eval_runner
from evals.judge import (
    build_blind_judge_prompt,
    build_judge_score_from_verdict,
    grade_case_with_judge,
    parse_judge_verdict,
    sanitize_response_for_blind_judging,
)
from evals.rubrics import (
    ComparisonReport,
    JudgeVerdict,
    estimate_query_cost_usd,
    render_markdown_report,
    score_deterministic_accuracy,
    score_heuristic_pedagogy,
)
from evals.runner import (
    DEFAULT_DATASET_PATH,
    _parse_args,
    load_benchmark_dataset,
    run_benchmark,
)

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
    assert _is_rate_limit_error(
        RuntimeError("503 UNAVAILABLE: model experiencing high demand")
    )
    assert _is_rate_limit_error(RuntimeError("504 DEADLINE_EXCEEDED"))
    assert _is_rate_limit_error(TimeoutError())
    assert not _is_rate_limit_error(
        RuntimeError(
            "429 RESOURCE_EXHAUSTED: GenerateRequestsPerDayPerProjectPerModel-FreeTier requests per day limit exceeded"
        )
    )
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


def test_parse_args_supports_use_judge_and_judge_model_flags() -> None:
    default_args = _parse_args([])
    assert default_args.use_judge is False
    assert default_args.judge_model is None

    custom_args = _parse_args(
        [
            "--profiles",
            "remote",
            "local",
            "--use-judge",
            "--judge-model",
            "gemini-2.5-pro",
        ]
    )
    assert custom_args.profiles == ["remote", "local"]
    assert custom_args.use_judge is True
    assert custom_args.judge_model == "gemini-2.5-pro"


def test_blind_judge_prompt_strips_model_identity_and_includes_checklist() -> None:
    case = load_benchmark_dataset(DEFAULT_DATASET_PATH)[0]
    leaky_response = (
        "[DRY-RUN:remote] Como modelo gemini-3.8-flash (e diferente do ollama_chat/gemma3:4b), "
        "explico que em 1500 já havia milhões de povos indígenas no território."
    )

    sanitized = sanitize_response_for_blind_judging(
        leaky_response,
        extra_identifiers=("gemini-3.8-flash", "ollama_chat/gemma3:4b", "gemma3:4b"),
    )
    assert "[DRY-RUN:remote]" not in sanitized
    assert "gemini-3.8-flash" not in sanitized.lower()
    assert "ollama_chat" not in sanitized.lower()
    assert "gemma3:4b" not in sanitized.lower()
    assert "[MODEL_REDACTED]" in sanitized

    prompt = build_blind_judge_prompt(
        case,
        leaky_response,
        extra_identifiers=("gemini-3.8-flash", "ollama_chat/gemma3:4b", "gemma3:4b"),
    )
    assert "gemini-3.8-flash" not in prompt.lower()
    assert "ollama" not in prompt.lower()
    assert "gemma3" not in prompt.lower()
    assert "[DRY-RUN:remote]" not in prompt
    assert case.prompt in prompt
    assert case.reference_summary in prompt
    assert case.target_level in prompt
    for fact in case.expected_facts:
        assert fact in prompt
    for trap in case.forbidden_hallucinations:
        assert trap in prompt


def test_parse_judge_verdict_and_score_clamping() -> None:
    raw_dict = {
        "fact_coverage_score_5": 5.4,
        "hallucination_resistance_score_5": 4.5,
        "grounding_score_5": 4.0,
        "clarity_score_5": 4.5,
        "level_adaptation_score_5": 5.0,
        "critical_thinking_score_5": -0.5,
        "accuracy_rationale": "Covered key facts without anachronisms.",
        "pedagogy_rationale": "Clear pt-BR but lacked a reflective closing question.",
        "detected_issues": ["  ", "Missing active reflection prompt"],
    }

    from_dict = parse_judge_verdict(raw_dict)
    assert isinstance(from_dict, JudgeVerdict)

    fenced_json = f"Here is the evaluation:\n```json\n{json.dumps(raw_dict)}\n```"
    from_fenced = parse_judge_verdict(fenced_json)
    assert from_fenced == from_dict

    from_model = parse_judge_verdict(from_dict)
    assert from_model is from_dict

    score = build_judge_score_from_verdict(from_fenced, judge_model="gemini-3.8-flash")
    assert score.status == "ok"
    assert score.fact_coverage_score_5 == 5.0
    assert score.critical_thinking_score_5 == 0.0
    assert score.accuracy_score_5 == pytest.approx(round((5.0 + 4.5 + 4.0) / 3.0, 2))
    assert score.pedagogy_score_5 == pytest.approx(round((4.5 + 5.0 + 0.0) / 3.0, 2))
    assert score.overall_score_5 == pytest.approx(
        round((score.accuracy_score_5 + score.pedagogy_score_5) / 2.0, 2)
    )
    assert score.detected_issues == ["Missing active reflection prompt"]
    assert "Accuracy:" in score.rationale
    assert "Pedagogy:" in score.rationale


@pytest.mark.asyncio
async def test_grade_case_with_judge_fallback_behavior() -> None:
    case = load_benchmark_dataset(DEFAULT_DATASET_PATH)[0]
    response = (
        "Em 1500 milhões de povos indígenas habitavam o Brasil. "
        "O termo descobrimento é eurocêntrico e houve forte resistência."
    )
    det_accuracy = score_deterministic_accuracy(case, response)
    det_pedagogy = score_heuristic_pedagogy(case, response)

    async def malformed_invoker(prompt: str, model: str) -> str:
        return "not valid json at all"

    fallback_malformed = await grade_case_with_judge(
        case=case,
        response_text=response,
        accuracy=det_accuracy,
        pedagogy=det_pedagogy,
        judge_model="gemini-3.8-flash",
        judge_invoker=malformed_invoker,
    )
    assert fallback_malformed.status == "fallback"
    assert fallback_malformed.error is not None
    assert fallback_malformed.accuracy_score_5 == det_accuracy.accuracy_score_5
    assert fallback_malformed.pedagogy_score_5 == det_pedagogy.pedagogy_avg_5

    async def missing_fields_invoker(prompt: str, model: str) -> str:
        return json.dumps({"fact_coverage_score_5": 4.0})

    fallback_schema = await grade_case_with_judge(
        case=case,
        response_text=response,
        accuracy=det_accuracy,
        pedagogy=det_pedagogy,
        judge_model="gemini-3.8-flash",
        judge_invoker=missing_fields_invoker,
    )
    assert fallback_schema.status == "fallback"
    assert "ValidationError" in (fallback_schema.error or "")

    async def failing_invoker(prompt: str, model: str) -> str:
        raise RuntimeError("Judge backend unavailable")

    fallback_error = await grade_case_with_judge(
        case=case,
        response_text=response,
        accuracy=det_accuracy,
        pedagogy=det_pedagogy,
        judge_model="gemini-3.8-flash",
        judge_invoker=failing_invoker,
        max_rate_limit_retries=0,
    )
    assert fallback_error.status == "fallback"
    assert "Judge backend unavailable" in (fallback_error.error or "")

    invoker_called = False

    async def unexpected_invoker(prompt: str, model: str) -> str:
        nonlocal invoker_called
        invoker_called = True
        return "{}"

    fallback_empty = await grade_case_with_judge(
        case=case,
        response_text="   ",
        accuracy=score_deterministic_accuracy(case, ""),
        pedagogy=score_heuristic_pedagogy(case, ""),
        judge_model="gemini-3.8-flash",
        judge_invoker=unexpected_invoker,
    )
    assert invoker_called is False
    assert fallback_empty.status == "fallback"
    assert fallback_empty.error == "Empty candidate response"


@pytest.mark.asyncio
async def test_run_benchmark_with_mocked_judge_and_partial_fallback(
    tmp_path: Path,
) -> None:
    full_dataset = json.loads(DEFAULT_DATASET_PATH.read_text(encoding="utf-8"))
    mini_dataset_path = tmp_path / "mini_benchmark.json"
    mini_dataset_path.write_text(json.dumps(full_dataset[:2]), encoding="utf-8")

    captured_prompts: list[str] = []
    captured_models: list[str] = []

    async def mock_judge_invoker(prompt: str, judge_model: str) -> str:
        captured_prompts.append(prompt)
        captured_models.append(judge_model)
        # Fail the second case for each profile to verify fallback accounting.
        if full_dataset[1]["id"] in prompt:
            raise ValueError("Simulated malformed judge response")
        return json.dumps(
            {
                "fact_coverage_score_5": 5.0,
                "hallucination_resistance_score_5": 4.5,
                "grounding_score_5": 4.5,
                "clarity_score_5": 4.5,
                "level_adaptation_score_5": 4.0,
                "critical_thinking_score_5": 5.0,
                "accuracy_rationale": "Accurate and free of myths.",
                "pedagogy_rationale": "Strong SDG 4 critical framing.",
                "detected_issues": [],
            }
        )

    settings = Settings()
    report, json_path, md_path = await run_benchmark(
        profiles=["remote", "local"],
        dataset_path=mini_dataset_path,
        reports_dir=tmp_path / "reports",
        dry_run=True,
        settings=settings,
        use_judge=True,
        judge_invoker=mock_judge_invoker,
    )

    assert report.use_judge is True
    assert report.judge_model == settings.eval_judge_model
    assert len(captured_prompts) == 4
    assert set(captured_models) == {settings.eval_judge_model}
    for prompt in captured_prompts:
        assert "gemini-3.8-flash" not in prompt.lower()
        assert "ollama" not in prompt.lower()
        assert "[DRY-RUN:remote]" not in prompt
        assert "[DRY-RUN:local]" not in prompt

    assert len(report.summaries) == 2
    for summary in report.summaries:
        assert summary.judge_evaluated_cases == 1
        assert summary.judge_fallback_cases == 1
        assert summary.avg_judge_accuracy_5 == pytest.approx(4.67)
        assert summary.avg_judge_pedagogy_5 == pytest.approx(4.5)
        assert summary.avg_judge_overall_5 == pytest.approx(4.58)

    json_payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert json_payload["use_judge"] is True
    assert json_payload["judge_model"] == settings.eval_judge_model
    assert all(item["judge"] is not None for item in json_payload["results"])

    markdown = md_path.read_text(encoding="utf-8")
    assert "LLM-as-a-judge:" in markdown
    assert "Judge Accuracy (/5)" in markdown
    assert "Judge Pedagogy (/5)" in markdown
    assert "Judge Overall (/5)" in markdown
    assert "Det. Accuracy (/5)" in markdown
    assert "## LLM-as-a-Judge Rationales" in markdown
    assert "`ok`" in markdown
    assert "`fallback`" in markdown
    assert "(1 fallback)" in markdown


@pytest.mark.asyncio
async def test_dry_run_with_use_judge_stays_fully_offline(tmp_path: Path) -> None:
    settings = Settings()
    report, json_path, md_path = await run_benchmark(
        profiles=["remote", "local"],
        dataset_path=DEFAULT_DATASET_PATH,
        reports_dir=tmp_path,
        dry_run=True,
        settings=settings,
        use_judge=True,
    )
    assert report.use_judge is True
    assert report.judge_model == settings.eval_judge_model
    assert all(
        result.judge is not None and result.judge.status == "ok"
        for result in report.results
    )
    assert all(
        s.judge_evaluated_cases == s.total_cases and s.judge_fallback_cases == 0
        for s in report.summaries
    )
    assert json_path.exists()
    assert md_path.exists()

    custom_report, _, _ = await run_benchmark(
        profiles=["remote"],
        dataset_path=DEFAULT_DATASET_PATH,
        reports_dir=tmp_path / "custom",
        dry_run=True,
        settings=settings,
        use_judge=True,
        judge_model="ollama_chat/gemma3:4b",
    )
    assert custom_report.judge_model == "ollama_chat/gemma3:4b"
    assert all(
        r.judge is not None and r.judge.judge_model == "ollama_chat/gemma3:4b"
        for r in custom_report.results
    )


def test_parse_args_supports_disable_tools_and_case_ids() -> None:
    default_args = _parse_args([])
    assert default_args.disable_tools is False
    assert default_args.case_ids is None

    ablation_args = _parse_args(
        [
            "--profiles",
            "local",
            "--disable-tools",
            "--case-ids",
            "colonial_indigenas_eurocentrismo",
            "epistemic_honesty_fake_treaty",
        ]
    )
    assert ablation_args.profiles == ["local"]
    assert ablation_args.disable_tools is True
    assert ablation_args.case_ids == [
        "colonial_indigenas_eurocentrismo",
        "epistemic_honesty_fake_treaty",
    ]


@pytest.mark.asyncio
async def test_dry_run_disable_tools_ablation_and_case_id_filtering(
    tmp_path: Path,
) -> None:
    selected_ids = [
        "colonial_indigenas_eurocentrismo",
        "epistemic_honesty_fake_treaty",
    ]
    report, json_path, md_path = await run_benchmark(
        profiles=["local"],
        dataset_path=DEFAULT_DATASET_PATH,
        reports_dir=tmp_path,
        dry_run=True,
        use_judge=True,
        disable_tools=True,
        case_ids=selected_ids,
    )

    assert report.tools_enabled is False
    assert len(report.summaries) == 1
    assert report.summaries[0].tools_enabled is False
    assert report.summaries[0].total_cases == 2
    assert [r.case_id for r in report.results] == selected_ids
    assert all(r.tools_enabled is False for r in report.results)
    assert json_path.name.endswith("_notools.json")
    assert md_path.name.endswith("_notools.md")

    markdown = md_path.read_text(encoding="utf-8")
    assert "Tools / Hybrid RAG:" in markdown
    assert "pure model memory ablation" in markdown
    assert "| `local` |" in markdown

    sanitized = sanitize_response_for_blind_judging(
        "[DRY-RUN:local:no-tools] Explicação sobre 1500."
    )
    assert "[DRY-RUN:local:no-tools]" not in sanitized

    with pytest.raises(ValueError, match="Unknown benchmark case IDs"):
        await run_benchmark(
            profiles=["local"],
            dataset_path=DEFAULT_DATASET_PATH,
            reports_dir=tmp_path,
            dry_run=True,
            case_ids=["nonexistent_case_id"],
        )


def test_parse_args_supports_checkpoint_resume_and_charts_flags(tmp_path: Path) -> None:
    default_args = _parse_args([])
    assert default_args.checkpoint is None
    assert default_args.resume is None
    assert default_args.charts is False
    assert default_args.charts_dir is None

    ckpt = tmp_path / "partial.json"
    charts_out = tmp_path / "svg_out"
    custom_args = _parse_args(
        [
            "--checkpoint",
            str(ckpt),
            "--resume",
            str(ckpt),
            "--charts",
            "--charts-dir",
            str(charts_out),
        ]
    )
    assert custom_args.checkpoint == ckpt
    assert custom_args.resume == ckpt
    assert custom_args.charts is True
    assert custom_args.charts_dir == charts_out


@pytest.mark.asyncio
async def test_incremental_checkpoint_persists_completed_cases_and_resumes_cleanly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases = load_benchmark_dataset(DEFAULT_DATASET_PATH)[:2]
    selected_ids = [case.id for case in cases]
    checkpoint_file = tmp_path / "live_checkpoint.json"
    settings = Settings()

    run1_calls: list[tuple[str, str]] = []

    async def failing_run1_invoker(
        _settings: Settings,
        profile: str,
        _prompt: str,
        case_id: str,
        **_kwargs: object,
    ) -> tuple[str, int, int, float]:
        run1_calls.append((profile, case_id))
        if profile == "remote" and case_id == selected_ids[1]:
            raise RuntimeError("503 UNAVAILABLE: upstream spike")
        if profile == "local" and case_id == selected_ids[1]:
            raise KeyboardInterrupt("Simulated user interruption")
        return (
            f"Resposta grounded para {case_id} com contexto histórico e reflexão crítica?",
            120,
            180,
            450.0 if profile == "remote" else 900.0,
        )

    monkeypatch.setattr(
        eval_runner, "_invoke_with_rate_limit_retry", failing_run1_invoker
    )

    with pytest.raises(KeyboardInterrupt):
        await run_benchmark(
            profiles=["remote", "local"],
            dataset_path=DEFAULT_DATASET_PATH,
            reports_dir=tmp_path,
            dry_run=False,
            settings=settings,
            remote_request_delay_s=0.0,
            case_ids=selected_ids,
            checkpoint_path=checkpoint_file,
        )

    assert checkpoint_file.exists()
    partial_report = ComparisonReport.model_validate_json(
        checkpoint_file.read_text(encoding="utf-8")
    )
    assert len(partial_report.results) == 3
    by_key = {(r.profile, r.case_id): r for r in partial_report.results}
    assert by_key[("remote", selected_ids[0])].error is None
    assert "503 UNAVAILABLE" in (by_key[("remote", selected_ids[1])].error or "")
    assert by_key[("local", selected_ids[0])].error is None
    assert ("local", selected_ids[1]) not in by_key

    # Resume from the checkpoint: completed non-errored pairs must be skipped,
    # while the errored remote case and missing local case must be executed.
    run2_calls: list[tuple[str, str]] = []
    sleep_calls: list[float] = []

    async def recovering_run2_invoker(
        _settings: Settings,
        profile: str,
        _prompt: str,
        case_id: str,
        **_kwargs: object,
    ) -> tuple[str, int, int, float]:
        run2_calls.append((profile, case_id))
        return (
            f"Resposta recuperada para {case_id} com perspectiva crítica e análise?",
            110,
            190,
            390.0 if profile == "remote" else 840.0,
        )

    async def fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    monkeypatch.setattr(
        eval_runner, "_invoke_with_rate_limit_retry", recovering_run2_invoker
    )
    monkeypatch.setattr(eval_runner.asyncio, "sleep", fake_sleep)

    resumed_report, json_path, md_path = await run_benchmark(
        profiles=["remote", "local"],
        dataset_path=DEFAULT_DATASET_PATH,
        reports_dir=tmp_path,
        dry_run=False,
        settings=settings,
        remote_request_delay_s=5.0,
        case_ids=selected_ids,
        checkpoint_path=checkpoint_file,
        resume_from=checkpoint_file,
    )

    assert run2_calls == [
        ("remote", selected_ids[1]),
        ("local", selected_ids[1]),
    ]
    # Only one live remote call was needed on resume, so no inter-request delay sleep occurred.
    assert sleep_calls == []
    assert json_path == checkpoint_file
    assert md_path.exists()
    assert len(resumed_report.results) == 4
    assert all(r.error is None for r in resumed_report.results)
    assert [(r.profile, r.case_id) for r in resumed_report.results] == [
        ("remote", selected_ids[0]),
        ("remote", selected_ids[1]),
        ("local", selected_ids[0]),
        ("local", selected_ids[1]),
    ]
    assert all(
        s.error_count == 0 and s.total_cases == 2 for s in resumed_report.summaries
    )

    with pytest.raises(FileNotFoundError, match="Resume checkpoint not found"):
        await run_benchmark(
            profiles=["remote"],
            dataset_path=DEFAULT_DATASET_PATH,
            reports_dir=tmp_path,
            dry_run=True,
            resume_from=tmp_path / "does_not_exist.json",
        )


@pytest.mark.asyncio
async def test_resume_reuses_candidate_output_when_adding_judge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected_ids = [load_benchmark_dataset(DEFAULT_DATASET_PATH)[0].id]
    ckpt_path = tmp_path / "without_judge.json"

    await run_benchmark(
        profiles=["remote"],
        dataset_path=DEFAULT_DATASET_PATH,
        reports_dir=tmp_path,
        dry_run=True,
        case_ids=selected_ids,
        checkpoint_path=ckpt_path,
    )

    candidate_called = False

    def fail_if_candidate_runs(
        *_args: object, **_kwargs: object
    ) -> tuple[str, int, int, float]:
        nonlocal candidate_called
        candidate_called = True
        raise AssertionError("Cached candidate response should not be re-executed")

    monkeypatch.setattr(
        eval_runner, "_simulate_dry_run_response", fail_if_candidate_runs
    )

    judge_calls = 0

    async def mock_judge(_prompt: str, _model: str) -> str:
        nonlocal judge_calls
        judge_calls += 1
        return json.dumps(
            {
                "fact_coverage_score_5": 5.0,
                "hallucination_resistance_score_5": 5.0,
                "grounding_score_5": 4.5,
                "clarity_score_5": 4.5,
                "level_adaptation_score_5": 4.5,
                "critical_thinking_score_5": 5.0,
                "accuracy_rationale": "Grounded.",
                "pedagogy_rationale": "Clear and reflective.",
                "detected_issues": [],
            }
        )

    judged_report, judged_json, _ = await run_benchmark(
        profiles=["remote"],
        dataset_path=DEFAULT_DATASET_PATH,
        reports_dir=tmp_path / "judged",
        dry_run=True,
        use_judge=True,
        judge_invoker=mock_judge,
        case_ids=selected_ids,
        resume_from=ckpt_path,
    )
    assert candidate_called is False
    assert judge_calls == 1
    assert judged_report.results[0].judge is not None
    assert judged_report.results[0].judge.status == "ok"

    # A second resume with the already-judged report skips both candidate and judge.
    await run_benchmark(
        profiles=["remote"],
        dataset_path=DEFAULT_DATASET_PATH,
        reports_dir=tmp_path / "judged_again",
        dry_run=True,
        use_judge=True,
        judge_invoker=mock_judge,
        case_ids=selected_ids,
        resume_from=judged_json,
    )
    assert judge_calls == 1


@pytest.mark.asyncio
async def test_offline_svg_chart_generation_from_runner_and_cli(
    tmp_path: Path,
) -> None:
    selected_ids = [
        case.id for case in load_benchmark_dataset(DEFAULT_DATASET_PATH)[:3]
    ]
    charts_dir = tmp_path / "runner_charts"

    report, json_path, _ = await run_benchmark(
        profiles=["remote", "local"],
        dataset_path=DEFAULT_DATASET_PATH,
        reports_dir=tmp_path,
        dry_run=True,
        use_judge=True,
        case_ids=selected_ids,
        generate_charts=True,
        charts_dir=charts_dir,
    )

    expected_svgs = [
        charts_dir / "accuracy_pedagogy.svg",
        charts_dir / "latency_vs_accuracy.svg",
        charts_dir / "cost_and_offline.svg",
    ]
    for svg_path in expected_svgs:
        assert svg_path.exists()
        raw_svg = svg_path.read_text(encoding="utf-8")
        root = ET.fromstring(raw_svg)
        assert root.tag == "{http://www.w3.org/2000/svg}svg"
        assert root.attrib.get("viewBox")

    acc_svg = (charts_dir / "accuracy_pedagogy.svg").read_text(encoding="utf-8")
    assert "Judge Accuracy (/5)" in acc_svg
    assert "remote:" in acc_svg
    assert "local:" in acc_svg

    lat_svg = (charts_dir / "latency_vs_accuracy.svg").read_text(encoding="utf-8")
    assert "p95 latency" in lat_svg
    assert "Mean latency" in lat_svg

    cost_svg = (charts_dir / "cost_and_offline.svg").read_text(encoding="utf-8")
    assert "$0.00 (Free/Local)" in cost_svg
    assert "Offline-ready" in cost_svg
    assert "Cloud only" in cost_svg

    # Standalone CLI (`python -m evals.charts <report.json>`) works offline.
    cli_out = tmp_path / "cli_charts"
    cli_result = eval_charts.main([str(json_path), "--output-dir", str(cli_out)])
    assert set(cli_result.keys()) == {
        "accuracy_pedagogy",
        "latency_vs_accuracy",
        "cost_and_offline",
    }
    for path_str in cli_result.values():
        parsed = ET.fromstring(Path(path_str).read_text(encoding="utf-8"))
        assert parsed.tag == "{http://www.w3.org/2000/svg}svg"

    # Empty summaries report still renders well-formed SVG placeholders.
    empty_report = report.model_copy(update={"summaries": [], "results": []})
    for renderer in (
        eval_charts.render_accuracy_pedagogy_svg,
        eval_charts.render_latency_accuracy_svg,
        eval_charts.render_cost_offline_svg,
    ):
        empty_svg = renderer(empty_report)
        root = ET.fromstring(empty_svg)
        assert root.tag == "{http://www.w3.org/2000/svg}svg"
        assert "No profile summaries available" in empty_svg

    # Special XML characters in metadata/model_id are safely escaped.
    tricky_summary = report.summaries[0].model_copy(
        update={"model_id": "gemini<3.8>&\"flash'"}
    )
    tricky_report = report.model_copy(update={"summaries": [tricky_summary]})
    for renderer in (
        eval_charts.render_accuracy_pedagogy_svg,
        eval_charts.render_latency_accuracy_svg,
        eval_charts.render_cost_offline_svg,
    ):
        tricky_svg = renderer(tricky_report)
        root = ET.fromstring(tricky_svg)
        assert root.tag == "{http://www.w3.org/2000/svg}svg"

    with pytest.raises(FileNotFoundError, match="Benchmark report JSON not found"):
        eval_charts.generate_charts_from_file(tmp_path / "missing_report.json")
