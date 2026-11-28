"""CLI benchmark runner comparing local (Ollama) and remote (Gemini) ADK 2.0 agents."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.app_utils.sanitization import sanitize_student_response
from app.config import ModelProfile, Settings, load_settings
from app.prompts import PROMPT_VERSION
from evals.judge import JudgeInvoker, grade_case_with_judge, is_rate_limit_error
from evals.rubrics import (
    BenchmarkCase,
    CaseEvaluationResult,
    ComparisonReport,
    JudgeScore,
    OperationalMetrics,
    aggregate_profile_summary,
    estimate_query_cost_usd,
    render_markdown_report,
    score_deterministic_accuracy,
    score_heuristic_pedagogy,
)

DEFAULT_DATASET_PATH = (
    Path(__file__).resolve().parent / "datasets" / "brazil_history_benchmark.json"
)
DEFAULT_REPORTS_DIR = Path(__file__).resolve().parent / "reports"


def load_benchmark_dataset(dataset_path: Path) -> list[BenchmarkCase]:
    """Load and validate a frozen benchmark dataset JSON file."""
    raw = json.loads(dataset_path.read_text(encoding="utf-8"))
    return [BenchmarkCase.model_validate(item) for item in raw]


def _extract_workflow_output(
    events: list[Any],
    terminal_author: str | None = None,
) -> tuple[str, int, int]:
    """Extract the final student-facing response and cumulative token counts from ADK events.

    In multi-node workflows (`TopicRouter -> HistoricalResearcher -> PedagogicalAdapter ->
    FactVerifier`), intermediate nodes emit events containing routing summaries, evidence
    dossiers, or unverified drafts. When `terminal_author` is present among the emitted
    text events, only that node's text is returned; otherwise the last text-emitting author
    is used so single-node agents remain compatible.
    """
    text_by_author: dict[str, list[str]] = {}
    author_order: list[str] = []
    input_tokens = 0
    output_tokens = 0

    for event in events:
        is_partial = bool(getattr(event, "partial", False))
        usage = getattr(event, "usage_metadata", None)
        if usage is not None and not is_partial:
            input_tokens += int(getattr(usage, "prompt_token_count", 0) or 0)
            output_tokens += int(getattr(usage, "candidates_token_count", 0) or 0)

        if is_partial:
            continue

        event_content = getattr(event, "content", None)
        if event_content and getattr(event_content, "parts", None):
            author = str(getattr(event, "author", "") or "__default__")
            for part in event_content.parts:
                text = getattr(part, "text", None)
                if text:
                    if author not in text_by_author:
                        text_by_author[author] = []
                        author_order.append(author)
                    text_by_author[author].append(text)

    if terminal_author and terminal_author in text_by_author:
        selected_chunks = text_by_author[terminal_author]
    elif author_order:
        selected_chunks = text_by_author[author_order[-1]]
    else:
        selected_chunks = []

    return (
        sanitize_student_response("\n".join(selected_chunks)),
        input_tokens,
        output_tokens,
    )


async def _invoke_live_adk_agent(
    settings: Settings,
    profile: ModelProfile,
    prompt: str,
    case_id: str,
    *,
    enable_tools: bool = True,
) -> tuple[str, int, int, float]:
    """Execute a single prompt against the ADK 2.0 workflow via `InMemoryRunner`."""
    from google.adk.runners import InMemoryRunner
    from google.genai import types

    from app.agent import TERMINAL_NODE_NAME, create_history_tutor_agent

    agent = create_history_tutor_agent(
        settings=settings,
        profile=profile,
        enable_tools=enable_tools,
    )
    mode_tag = "rag" if enable_tools else "notools"
    app_name = f"brasao_ai_eval_{profile}_{mode_tag}"
    user_id = "benchmark_evaluator"

    runner = InMemoryRunner(agent=agent, app_name=app_name)
    session = await runner.session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=f"{profile}_{mode_tag}_{case_id}",
    )

    content = types.Content(role="user", parts=[types.Part.from_text(text=prompt)])
    events: list[Any] = []

    started_at = time.perf_counter()
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session.id,
        new_message=content,
    ):
        events.append(event)

    latency_ms = (time.perf_counter() - started_at) * 1000.0
    full_response, input_tokens, output_tokens = _extract_workflow_output(
        events, terminal_author=TERMINAL_NODE_NAME
    )
    if output_tokens == 0 and full_response:
        # Fallback estimate (~1.3 tokens per word) when a local provider omits token metadata.
        output_tokens = int(len(full_response.split()) * 1.3)
    if input_tokens == 0:
        input_tokens = int(len(prompt.split()) * 1.3)

    return full_response, input_tokens, output_tokens, latency_ms


def _simulate_dry_run_response(
    case: BenchmarkCase,
    profile: ModelProfile,
    *,
    enable_tools: bool = True,
) -> tuple[str, int, int, float]:
    """Produce a deterministic offline dry-run response to validate the benchmark pipeline."""
    facts_line = ", ".join(case.expected_facts)
    mode_suffix = "" if enable_tools else ":no-tools"
    context_label = (
        "Análise histórica contextualizada"
        if enable_tools
        else "Análise histórica sem ferramentas (memória paramétrica)"
    )
    response = (
        f"[DRY-RUN:{profile}{mode_suffix}] {context_label} ({case.target_level}): "
        f"{case.reference_summary}\n"
        f"- Elementos verificados: {facts_line}.\n"
        "- Perspectiva crítica: como esse processo histórico se conecta às desigualdades e à "
        "cidadania no Brasil contemporâneo?"
    )
    if enable_tools:
        simulated_latency = 420.0 if profile == "remote" else 980.0
    else:
        simulated_latency = 280.0 if profile == "remote" else 620.0
    input_tokens = max(20, int(len(case.prompt.split()) * 1.3))
    output_tokens = max(40, int(len(response.split()) * 1.3))
    return response, input_tokens, output_tokens, simulated_latency


def _is_rate_limit_error(exc: Exception) -> bool:
    """Return True when an exception indicates an HTTP 429/503 rate-limit or demand response."""
    return is_rate_limit_error(exc)


async def _invoke_with_rate_limit_retry(
    settings: Settings,
    profile: ModelProfile,
    prompt: str,
    case_id: str,
    max_retries: int = 4,
    base_backoff_s: float = 15.0,
    remote_case_timeout_s: float = 180.0,
    *,
    enable_tools: bool = True,
) -> tuple[str, int, int, float]:
    """Invoke the live ADK agent with exponential backoff for Gemini 429/503/504 limits."""
    attempt = 0
    while True:
        try:
            coro = _invoke_live_adk_agent(
                settings,
                profile,
                prompt,
                case_id,
                enable_tools=enable_tools,
            )
            if profile == "remote" and remote_case_timeout_s > 0:
                return await asyncio.wait_for(coro, timeout=remote_case_timeout_s)
            return await coro
        except Exception as exc:
            if (
                profile != "remote"
                or not _is_rate_limit_error(exc)
                or attempt >= max_retries
            ):
                raise
            delay = base_backoff_s * (2**attempt)
            attempt += 1
            print(
                f"[{profile}] transient error on {case_id} ({type(exc).__name__}); "
                f"retrying in {delay:.0f}s (attempt {attempt}/{max_retries})...",
                flush=True,
            )
            await asyncio.sleep(delay)


def _select_benchmark_cases(
    cases: list[BenchmarkCase],
    case_ids: list[str] | None,
) -> list[BenchmarkCase]:
    """Optionally filter a frozen benchmark dataset to an explicit ordered list of case IDs."""
    if not case_ids:
        return cases
    by_id = {case.id: case for case in cases}
    missing = [cid for cid in case_ids if cid not in by_id]
    if missing:
        raise ValueError(f"Unknown benchmark case IDs: {', '.join(missing)}")
    return [by_id[cid] for cid in case_ids]


async def run_benchmark(
    profiles: list[ModelProfile],
    dataset_path: Path = DEFAULT_DATASET_PATH,
    reports_dir: Path = DEFAULT_REPORTS_DIR,
    dry_run: bool = False,
    settings: Settings | None = None,
    remote_request_delay_s: float | None = None,
    max_rate_limit_retries: int = 4,
    use_judge: bool = False,
    judge_model: str | None = None,
    judge_invoker: JudgeInvoker | None = None,
    disable_tools: bool = False,
    case_ids: list[str] | None = None,
) -> tuple[ComparisonReport, Path, Path]:
    """Run the comparative benchmark and write JSON and Markdown report artifacts."""
    resolved_settings = settings or load_settings()
    enable_tools = not disable_tools
    resolved_judge_model = (
        (judge_model or resolved_settings.eval_judge_model).strip()
        if use_judge
        else None
    )
    delay_s = (
        remote_request_delay_s
        if remote_request_delay_s is not None
        else float(os.getenv("REMOTE_REQUEST_DELAY_S", "0.0"))
    )
    cases = _select_benchmark_cases(load_benchmark_dataset(dataset_path), case_ids)
    all_results: list[CaseEvaluationResult] = []
    summaries = []

    redacted_identifiers = (
        resolved_settings.gemini_model,
        resolved_settings.ollama_model,
        f"ollama_chat/{resolved_settings.ollama_model}",
    )

    for profile in profiles:
        model_id = resolved_settings.resolved_model_label(profile)
        offline_capable = resolved_settings.is_offline_capable(profile)
        profile_results: list[CaseEvaluationResult] = []

        for idx, case in enumerate(cases):
            if not dry_run and profile == "remote" and delay_s > 0 and idx > 0:
                await asyncio.sleep(delay_s)

            error_msg: str | None = None
            try:
                if dry_run:
                    response_text, in_tok, out_tok, latency_ms = (
                        _simulate_dry_run_response(
                            case,
                            profile,
                            enable_tools=enable_tools,
                        )
                    )
                else:
                    (
                        response_text,
                        in_tok,
                        out_tok,
                        latency_ms,
                    ) = await _invoke_with_rate_limit_retry(
                        resolved_settings,
                        profile,
                        case.prompt,
                        case.id,
                        max_retries=max_rate_limit_retries,
                        enable_tools=enable_tools,
                    )
            except Exception as exc:
                response_text = ""
                in_tok, out_tok, latency_ms = 0, 0, 0.0
                error_msg = f"{type(exc).__name__}: {exc}"

            tps = round(out_tok / (latency_ms / 1000.0), 2) if latency_ms > 0 else 0.0
            cost = estimate_query_cost_usd(profile, model_id, in_tok, out_tok)
            accuracy = score_deterministic_accuracy(case, response_text)
            pedagogy = score_heuristic_pedagogy(case, response_text)

            judge_score: JudgeScore | None = None
            judge_notes = ""
            if use_judge and resolved_judge_model is not None:
                if (
                    not dry_run
                    and judge_invoker is None
                    and delay_s > 0
                    and not resolved_judge_model.startswith(("ollama_chat/", "ollama/"))
                ):
                    await asyncio.sleep(delay_s)
                judge_score = await grade_case_with_judge(
                    case=case,
                    response_text=response_text,
                    accuracy=accuracy,
                    pedagogy=pedagogy,
                    judge_model=resolved_judge_model,
                    extra_identifiers=(*redacted_identifiers, model_id),
                    candidate_error=error_msg,
                    dry_run=dry_run,
                    judge_invoker=judge_invoker,
                    max_rate_limit_retries=max_rate_limit_retries,
                )
                judge_notes = judge_score.rationale

            result = CaseEvaluationResult(
                case_id=case.id,
                category=case.category,
                profile=profile,
                model_id=model_id,
                tools_enabled=enable_tools,
                response_text=response_text,
                accuracy=accuracy,
                pedagogy=pedagogy,
                operational=OperationalMetrics(
                    latency_ms=round(latency_ms, 2),
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    tokens_per_second=tps,
                    estimated_cost_usd=round(cost, 6),
                    offline_capable=offline_capable,
                ),
                judge=judge_score,
                judge_notes=judge_notes,
                error=error_msg,
            )
            profile_results.append(result)
            all_results.append(result)
            if not dry_run:
                judge_str = (
                    f" judge={judge_score.overall_score_5:.2f}({judge_score.status})"
                    if judge_score is not None
                    else ""
                )
                err_str = f" ERROR={error_msg}" if error_msg else ""
                tools_tag = "rag" if enable_tools else "no-tools"
                print(
                    f"[{profile}:{model_id}:{tools_tag}] {idx + 1}/{len(cases)} {case.id} "
                    f"acc={accuracy.accuracy_score_5:.2f} ped={pedagogy.pedagogy_avg_5:.2f}"
                    f"{judge_str} latency={latency_ms:.0f}ms{err_str}",
                    flush=True,
                )

        summaries.append(
            aggregate_profile_summary(
                profile,
                model_id,
                offline_capable,
                profile_results,
                tools_enabled=enable_tools,
            )
        )

    now = datetime.now(UTC)
    report = ComparisonReport(
        generated_at=now.isoformat(),
        prompt_version=PROMPT_VERSION,
        dataset_path=str(dataset_path),
        dry_run=dry_run,
        tools_enabled=enable_tools,
        use_judge=use_judge,
        judge_model=resolved_judge_model,
        summaries=summaries,
        results=all_results,
    )

    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    suffix = "" if enable_tools else "_notools"
    json_path = reports_dir / f"comparison_{stamp}{suffix}.json"
    md_path = reports_dir / f"comparison_{stamp}{suffix}.md"

    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    md_path.write_text(render_markdown_report(report), encoding="utf-8")

    return report, json_path, md_path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare local (Ollama) vs. remote (Gemini) Brazilian History ADK 2.0 agents."
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=["remote", "local"],
        default=["remote", "local"],
        help="Model profiles to evaluate in order.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to the frozen benchmark JSON dataset.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=DEFAULT_REPORTS_DIR,
        help="Output directory for JSON and Markdown comparison reports.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate model outputs offline to verify the benchmark harness end-to-end.",
    )
    parser.add_argument(
        "--disable-tools",
        action="store_true",
        help="Disable curated corpus/RAG, timeline, and quiz tools (pure model memory ablation).",
    )
    parser.add_argument(
        "--case-ids",
        nargs="+",
        default=None,
        help="Optional subset of benchmark case IDs to execute from the frozen dataset.",
    )
    parser.add_argument(
        "--remote-delay",
        type=float,
        default=None,
        help="Seconds to pause between remote Gemini requests (e.g., 6.0 for Google AI Studio Free Tier).",
    )
    parser.add_argument(
        "--max-rate-limit-retries",
        type=int,
        default=4,
        help="Max exponential-backoff retries on HTTP 429 / RESOURCE_EXHAUSTED responses.",
    )
    parser.add_argument(
        "--use-judge",
        action="store_true",
        help="Enable blind LLM-as-a-judge grading alongside deterministic checks.",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="Model identifier used for LLM-as-a-judge grading (defaults to Settings.eval_judge_model).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    """CLI entrypoint for `python -m evals.runner`."""
    args = _parse_args(argv)
    report, json_path, md_path = asyncio.run(
        run_benchmark(
            profiles=args.profiles,
            dataset_path=args.dataset,
            reports_dir=args.reports_dir,
            dry_run=args.dry_run,
            remote_request_delay_s=args.remote_delay,
            max_rate_limit_retries=args.max_rate_limit_retries,
            use_judge=args.use_judge,
            judge_model=args.judge_model,
            disable_tools=args.disable_tools,
            case_ids=args.case_ids,
        )
    )
    print(f"Benchmark completed ({len(report.results)} evaluations).")
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {md_path}")
    return {"json_path": str(json_path), "md_path": str(md_path)}


if __name__ == "__main__":
    main()
