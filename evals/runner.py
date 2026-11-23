"""CLI benchmark runner comparing local (Ollama) and remote (Gemini) ADK 2.0 agents."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.config import ModelProfile, Settings, load_settings
from app.prompts import PROMPT_VERSION
from evals.rubrics import (
    BenchmarkCase,
    CaseEvaluationResult,
    ComparisonReport,
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


async def _invoke_live_adk_agent(
    settings: Settings,
    profile: ModelProfile,
    prompt: str,
    case_id: str,
) -> tuple[str, int, int, float]:
    """Execute a single prompt against the ADK 2.0 agent via `InMemoryRunner`."""
    from google.adk.runners import InMemoryRunner
    from google.genai import types

    from app.agent import create_history_tutor_agent

    agent = create_history_tutor_agent(settings=settings, profile=profile)
    app_name = f"brasao_ai_eval_{profile}"
    user_id = "benchmark_evaluator"

    runner = InMemoryRunner(agent=agent, app_name=app_name)
    session = await runner.session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=f"{profile}_{case_id}",
    )

    content = types.Content(role="user", parts=[types.Part.from_text(text=prompt)])
    response_chunks: list[str] = []
    input_tokens = 0
    output_tokens = 0

    started_at = time.perf_counter()
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session.id,
        new_message=content,
    ):
        usage = getattr(event, "usage_metadata", None)
        if usage is not None:
            input_tokens = max(
                input_tokens, int(getattr(usage, "prompt_token_count", 0) or 0)
            )
            output_tokens = max(
                output_tokens, int(getattr(usage, "candidates_token_count", 0) or 0)
            )

        event_content = getattr(event, "content", None)
        if event_content and getattr(event_content, "parts", None):
            for part in event_content.parts:
                text = getattr(part, "text", None)
                if text:
                    response_chunks.append(text)

    latency_ms = (time.perf_counter() - started_at) * 1000.0
    full_response = "\n".join(response_chunks).strip()
    if output_tokens == 0 and full_response:
        # Fallback estimate (~1.3 tokens per word) when a local provider omits token metadata.
        output_tokens = int(len(full_response.split()) * 1.3)
    if input_tokens == 0:
        input_tokens = int(len(prompt.split()) * 1.3)

    return full_response, input_tokens, output_tokens, latency_ms


def _simulate_dry_run_response(
    case: BenchmarkCase, profile: ModelProfile
) -> tuple[str, int, int, float]:
    """Produce a deterministic offline dry-run response to validate the benchmark pipeline."""
    facts_line = ", ".join(case.expected_facts)
    response = (
        f"[DRY-RUN:{profile}] Análise histórica contextualizada ({case.target_level}): "
        f"{case.reference_summary}\n"
        f"- Elementos verificados: {facts_line}.\n"
        "- Perspectiva crítica: como esse processo histórico se conecta às desigualdades e à "
        "cidadania no Brasil contemporâneo?"
    )
    simulated_latency = 420.0 if profile == "remote" else 980.0
    input_tokens = max(20, int(len(case.prompt.split()) * 1.3))
    output_tokens = max(40, int(len(response.split()) * 1.3))
    return response, input_tokens, output_tokens, simulated_latency


async def run_benchmark(
    profiles: list[ModelProfile],
    dataset_path: Path = DEFAULT_DATASET_PATH,
    reports_dir: Path = DEFAULT_REPORTS_DIR,
    dry_run: bool = False,
    settings: Settings | None = None,
) -> tuple[ComparisonReport, Path, Path]:
    """Run the comparative benchmark and write JSON and Markdown report artifacts."""
    resolved_settings = settings or load_settings()
    cases = load_benchmark_dataset(dataset_path)
    all_results: list[CaseEvaluationResult] = []
    summaries = []

    for profile in profiles:
        model_id = resolved_settings.resolved_model_label(profile)
        offline_capable = resolved_settings.is_offline_capable(profile)
        profile_results: list[CaseEvaluationResult] = []

        for case in cases:
            error_msg: str | None = None
            try:
                if dry_run:
                    response_text, in_tok, out_tok, latency_ms = (
                        _simulate_dry_run_response(case, profile)
                    )
                else:
                    (
                        response_text,
                        in_tok,
                        out_tok,
                        latency_ms,
                    ) = await _invoke_live_adk_agent(
                        resolved_settings, profile, case.prompt, case.id
                    )
            except Exception as exc:
                response_text = ""
                in_tok, out_tok, latency_ms = 0, 0, 0.0
                error_msg = f"{type(exc).__name__}: {exc}"

            tps = round(out_tok / (latency_ms / 1000.0), 2) if latency_ms > 0 else 0.0
            cost = estimate_query_cost_usd(profile, model_id, in_tok, out_tok)

            result = CaseEvaluationResult(
                case_id=case.id,
                category=case.category,
                profile=profile,
                model_id=model_id,
                response_text=response_text,
                accuracy=score_deterministic_accuracy(case, response_text),
                pedagogy=score_heuristic_pedagogy(case, response_text),
                operational=OperationalMetrics(
                    latency_ms=round(latency_ms, 2),
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    tokens_per_second=tps,
                    estimated_cost_usd=round(cost, 6),
                    offline_capable=offline_capable,
                ),
                error=error_msg,
            )
            profile_results.append(result)
            all_results.append(result)

        summaries.append(
            aggregate_profile_summary(
                profile, model_id, offline_capable, profile_results
            )
        )

    now = datetime.now(UTC)
    report = ComparisonReport(
        generated_at=now.isoformat(),
        prompt_version=PROMPT_VERSION,
        dataset_path=str(dataset_path),
        dry_run=dry_run,
        summaries=summaries,
        results=all_results,
    )

    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    json_path = reports_dir / f"comparison_{stamp}.json"
    md_path = reports_dir / f"comparison_{stamp}.md"

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
        )
    )
    print(f"Benchmark completed ({len(report.results)} evaluations).")
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {md_path}")
    return {"json_path": str(json_path), "md_path": str(md_path)}


if __name__ == "__main__":
    main()
