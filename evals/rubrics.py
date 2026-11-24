"""Typed rubrics, scoring helpers, and report schemas for Local vs. Remote comparison."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.config import ModelProfile
from app.tools.corpus import normalize_text

# Reference pricing estimates (USD per 1M tokens, Nov 2026 global paid tier) used for
# comparative order-of-magnitude cost projections (1k queries). Local Ollama inference
# always has $0.00 marginal API cost.
MODEL_COST_PER_1M_TOKENS_USD: dict[str, tuple[float, float]] = {
    "gemini-3.8-flash": (0.75, 3.75),
    "gemini-4-hydrogen": (1.50, 7.50),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-2.0-flash": (0.10, 0.40),
}


class BenchmarkCase(BaseModel):
    """Single frozen evaluation case used identically across local and remote profiles."""

    id: str
    category: Literal[
        "factual_and_critical",
        "misconception_trap",
        "formative_assessment",
        "epistemic_honesty",
    ]
    target_level: Literal["fundamental_2", "ensino_medio", "geral"]
    bncc_codes: list[str]
    prompt: str
    expected_facts: list[str]
    forbidden_hallucinations: list[str]
    reference_summary: str


class HistoricalAccuracyScore(BaseModel):
    """Dimension 1: Historical accuracy and hallucination resistance."""

    expected_facts_matched: int
    expected_facts_total: int
    fact_recall: float = Field(ge=0.0, le=1.0)
    hallucination_flags: list[str] = Field(default_factory=list)
    accuracy_score_5: float = Field(ge=0.0, le=5.0)


class PedagogicalQualityScore(BaseModel):
    """Dimension 2: UN SDG 4 pedagogical quality."""

    clarity_score_5: float = Field(ge=0.0, le=5.0)
    level_adaptation_score_5: float = Field(ge=0.0, le=5.0)
    critical_thinking_score_5: float = Field(ge=0.0, le=5.0)
    pedagogy_avg_5: float = Field(ge=0.0, le=5.0)


class OperationalMetrics(BaseModel):
    """Dimension 3: Latency, throughput, estimated cost, and offline viability."""

    latency_ms: float = Field(ge=0.0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    tokens_per_second: float = Field(default=0.0, ge=0.0)
    estimated_cost_usd: float = Field(default=0.0, ge=0.0)
    offline_capable: bool


class JudgeVerdict(BaseModel):
    """Structured LLM-as-a-judge output schema (0-5 per sub-criterion + rationale)."""

    fact_coverage_score_5: float = Field(
        description="0-5 score for factual coverage of expected_facts."
    )
    hallucination_resistance_score_5: float = Field(
        description="0-5 score for absence of anachronisms and forbidden_hallucinations."
    )
    grounding_score_5: float = Field(
        description="0-5 score for grounding against reference_summary and epistemic honesty."
    )
    clarity_score_5: float = Field(
        description="0-5 score for clarity and organization in Brazilian Portuguese (pt-BR)."
    )
    level_adaptation_score_5: float = Field(
        description="0-5 score for adaptation to target_level (fundamental_2, ensino_medio, geral)."
    )
    critical_thinking_score_5: float = Field(
        description="0-5 score for critical thinking and multiple historical perspectives (SDG 4.7)."
    )
    accuracy_rationale: str = Field(
        description="Concise justification for the historical accuracy and hallucination scores."
    )
    pedagogy_rationale: str = Field(
        description="Concise justification for the UN SDG 4 pedagogical quality scores."
    )
    detected_issues: list[str] = Field(
        default_factory=list,
        description="Specific factual errors, anachronisms, or hallucination traps detected.",
    )


class JudgeScore(BaseModel):
    """Validated LLM-as-a-judge score record stored in JSON and Markdown reports."""

    judge_model: str
    status: Literal["ok", "fallback"] = "ok"
    fact_coverage_score_5: float = Field(ge=0.0, le=5.0)
    hallucination_resistance_score_5: float = Field(ge=0.0, le=5.0)
    grounding_score_5: float = Field(ge=0.0, le=5.0)
    accuracy_score_5: float = Field(ge=0.0, le=5.0)
    clarity_score_5: float = Field(ge=0.0, le=5.0)
    level_adaptation_score_5: float = Field(ge=0.0, le=5.0)
    critical_thinking_score_5: float = Field(ge=0.0, le=5.0)
    pedagogy_score_5: float = Field(ge=0.0, le=5.0)
    overall_score_5: float = Field(ge=0.0, le=5.0)
    accuracy_rationale: str = ""
    pedagogy_rationale: str = ""
    rationale: str = ""
    detected_issues: list[str] = Field(default_factory=list)
    error: str | None = None


class CaseEvaluationResult(BaseModel):
    """Full evaluation record for one case under one model profile."""

    case_id: str
    category: str
    profile: ModelProfile
    model_id: str
    response_text: str
    accuracy: HistoricalAccuracyScore
    pedagogy: PedagogicalQualityScore
    operational: OperationalMetrics
    judge: JudgeScore | None = None
    judge_notes: str = ""
    error: str | None = None


class ProfileSummary(BaseModel):
    """Aggregated benchmark summary for a single model profile."""

    profile: ModelProfile
    model_id: str
    offline_capable: bool
    total_cases: int
    error_count: int
    avg_accuracy_5: float
    avg_fact_recall: float
    hallucination_case_count: int
    avg_pedagogy_5: float
    judge_evaluated_cases: int = 0
    judge_fallback_cases: int = 0
    avg_judge_accuracy_5: float | None = None
    avg_judge_pedagogy_5: float | None = None
    avg_judge_overall_5: float | None = None
    mean_latency_ms: float
    p95_latency_ms: float
    mean_tokens_per_second: float
    estimated_cost_per_1k_queries_usd: float


class ComparisonReport(BaseModel):
    """Complete comparative report written to JSON and Markdown."""

    generated_at: str
    prompt_version: str
    dataset_path: str
    dry_run: bool
    use_judge: bool = False
    judge_model: str | None = None
    summaries: list[ProfileSummary]
    results: list[CaseEvaluationResult]


def estimate_query_cost_usd(
    profile: ModelProfile,
    model_id: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """Estimate marginal API cost in USD for a single interaction."""
    if profile == "local":
        return 0.0
    in_price, out_price = MODEL_COST_PER_1M_TOKENS_USD.get(model_id, (0.15, 0.60))
    return (input_tokens / 1_000_000) * in_price + (
        output_tokens / 1_000_000
    ) * out_price


def score_deterministic_accuracy(
    case: BenchmarkCase,
    response_text: str,
) -> HistoricalAccuracyScore:
    """Score factual coverage and forbidden hallucination traps deterministically."""
    normalized_response = normalize_text(response_text)
    matched = sum(
        1 for fact in case.expected_facts if normalize_text(fact) in normalized_response
    )
    total = len(case.expected_facts)
    recall = (matched / total) if total > 0 else 1.0

    hallucination_flags = [
        phrase
        for phrase in case.forbidden_hallucinations
        if normalize_text(phrase) in normalized_response
    ]

    # Map recall (0..1) onto 0..5 scale and deduct 2.0 per hallucination trigger.
    raw_score = recall * 5.0 - (2.0 * len(hallucination_flags))
    clamped_score = round(max(0.0, min(5.0, raw_score)), 2)

    return HistoricalAccuracyScore(
        expected_facts_matched=matched,
        expected_facts_total=total,
        fact_recall=round(recall, 3),
        hallucination_flags=hallucination_flags,
        accuracy_score_5=clamped_score,
    )


def score_heuristic_pedagogy(
    case: BenchmarkCase,
    response_text: str,
) -> PedagogicalQualityScore:
    """Compute baseline deterministic SDG 4 pedagogical indicators (0-5 scale).

    Used when `--use-judge` is disabled or as a floor alongside LLM-as-a-judge scoring.
    """
    text = response_text.strip()
    word_count = len(text.split())
    normalized = normalize_text(text)

    # Clarity: structured, neither empty nor excessively terse.
    if word_count == 0:
        clarity = 0.0
    elif word_count < 25:
        clarity = 2.0
    elif "\n" in text or "-" in text or "*" in text:
        clarity = 4.5
    else:
        clarity = 3.5

    # Level adaptation heuristic based on length and analytical cues.
    if case.target_level == "fundamental_2":
        adaptation = 4.0 if 40 <= word_count <= 350 else 3.0
    elif case.target_level == "ensino_medio":
        has_analytical_cues = any(
            cue in normalized
            for cue in (
                "contexto",
                "processo",
                "estrutur",
                "perspectiv",
                "historiograf",
                "consequenc",
            )
        )
        adaptation = 4.5 if has_analytical_cues and word_count >= 60 else 3.0
    else:
        adaptation = 4.0 if word_count >= 40 else 2.5

    # Critical thinking / active learning: reflective question or multi-perspective cues.
    has_question = "?" in text
    has_nuance = any(
        cue in normalized
        for cue in (
            "perspectiv",
            "resistenc",
            "contradic",
            "mito",
            "desigualdad",
            "protagonism",
        )
    )
    critical = 2.5 + (1.25 if has_question else 0.0) + (1.25 if has_nuance else 0.0)
    avg = round((clarity + adaptation + critical) / 3.0, 2)

    return PedagogicalQualityScore(
        clarity_score_5=round(clarity, 2),
        level_adaptation_score_5=round(adaptation, 2),
        critical_thinking_score_5=round(min(5.0, critical), 2),
        pedagogy_avg_5=avg,
    )


def compute_p95(values: list[float]) -> float:
    """Return the 95th percentile of a non-empty list of floats."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(0.95 * (len(ordered) - 1))))
    return round(ordered[index], 2)


def aggregate_profile_summary(
    profile: ModelProfile,
    model_id: str,
    offline_capable: bool,
    results: list[CaseEvaluationResult],
) -> ProfileSummary:
    """Aggregate per-case results into a single profile summary row."""
    total = len(results)
    if total == 0:
        return ProfileSummary(
            profile=profile,
            model_id=model_id,
            offline_capable=offline_capable,
            total_cases=0,
            error_count=0,
            avg_accuracy_5=0.0,
            avg_fact_recall=0.0,
            hallucination_case_count=0,
            avg_pedagogy_5=0.0,
            judge_evaluated_cases=0,
            judge_fallback_cases=0,
            avg_judge_accuracy_5=None,
            avg_judge_pedagogy_5=None,
            avg_judge_overall_5=None,
            mean_latency_ms=0.0,
            p95_latency_ms=0.0,
            mean_tokens_per_second=0.0,
            estimated_cost_per_1k_queries_usd=0.0,
        )

    latencies = [r.operational.latency_ms for r in results]
    avg_cost = sum(r.operational.estimated_cost_usd for r in results) / total

    judge_ok = [
        r.judge for r in results if r.judge is not None and r.judge.status == "ok"
    ]
    judge_fallback_count = sum(
        1 for r in results if r.judge is not None and r.judge.status == "fallback"
    )
    avg_judge_acc: float | None = None
    avg_judge_ped: float | None = None
    avg_judge_overall: float | None = None
    if judge_ok:
        n_ok = len(judge_ok)
        avg_judge_acc = round(sum(j.accuracy_score_5 for j in judge_ok) / n_ok, 2)
        avg_judge_ped = round(sum(j.pedagogy_score_5 for j in judge_ok) / n_ok, 2)
        avg_judge_overall = round(sum(j.overall_score_5 for j in judge_ok) / n_ok, 2)

    return ProfileSummary(
        profile=profile,
        model_id=model_id,
        offline_capable=offline_capable,
        total_cases=total,
        error_count=sum(1 for r in results if r.error is not None),
        avg_accuracy_5=round(
            sum(r.accuracy.accuracy_score_5 for r in results) / total, 2
        ),
        avg_fact_recall=round(sum(r.accuracy.fact_recall for r in results) / total, 3),
        hallucination_case_count=sum(
            1 for r in results if r.accuracy.hallucination_flags
        ),
        avg_pedagogy_5=round(
            sum(r.pedagogy.pedagogy_avg_5 for r in results) / total, 2
        ),
        judge_evaluated_cases=len(judge_ok),
        judge_fallback_cases=judge_fallback_count,
        avg_judge_accuracy_5=avg_judge_acc,
        avg_judge_pedagogy_5=avg_judge_ped,
        avg_judge_overall_5=avg_judge_overall,
        mean_latency_ms=round(sum(latencies) / total, 2),
        p95_latency_ms=compute_p95(latencies),
        mean_tokens_per_second=round(
            sum(r.operational.tokens_per_second for r in results) / total, 2
        ),
        estimated_cost_per_1k_queries_usd=round(avg_cost * 1000.0, 4),
    )


def _escape_md_cell(value: str) -> str:
    """Normalize a string for safe single-line rendering inside a Markdown table cell."""
    return " ".join(value.replace("|", "\\|").split())


def _fmt_optional_score(value: float | None) -> str:
    """Format an optional 0-5 score for Markdown tables."""
    return f"{value:.2f}" if value is not None else "N/A"


def render_markdown_report(report: ComparisonReport) -> str:
    """Render a side-by-side Markdown comparison report."""
    has_judge = report.use_judge or any(
        item.judge is not None for item in report.results
    )
    judge_line = (
        f"`enabled` (model: `{report.judge_model or 'unspecified'}`, blind checklist grading)"
        if has_judge
        else "`disabled` (deterministic + heuristic checks only)"
    )

    lines = [
        "# Brasão AI — Local (Ollama) vs. Remote (Gemini) ADK 2.0 Benchmark",
        "",
        f"- **Generated at:** `{report.generated_at}`",
        f"- **Prompt version:** `{report.prompt_version}`",
        f"- **Dataset:** `{report.dataset_path}`",
        f"- **Mode:** `{'dry-run (simulated)' if report.dry_run else 'live'}`",
        f"- **LLM-as-a-judge:** {judge_line}",
        "",
        "## Executive Summary",
        "",
    ]

    if has_judge:
        lines.extend(
            [
                "| Profile | Model | Offline | Det. Accuracy (/5) | Fact Recall | Hallucination Cases | Det. Pedagogy (/5) | Judge Accuracy (/5) | Judge Pedagogy (/5) | Judge Overall (/5) | Judge Graded | Mean Latency (ms) | p95 Latency (ms) | Tokens/s | Est. Cost / 1k Queries (USD) | Errors |",
                "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
            ]
        )
        for summary in report.summaries:
            graded_label = f"{summary.judge_evaluated_cases}/{summary.total_cases}"
            if summary.judge_fallback_cases > 0:
                graded_label += f" ({summary.judge_fallback_cases} fallback)"
            lines.append(
                f"| `{summary.profile}` | `{summary.model_id}` | "
                f"{'Yes' if summary.offline_capable else 'No'} | "
                f"{summary.avg_accuracy_5:.2f} | {summary.avg_fact_recall:.1%} | "
                f"{summary.hallucination_case_count}/{summary.total_cases} | "
                f"{summary.avg_pedagogy_5:.2f} | "
                f"{_fmt_optional_score(summary.avg_judge_accuracy_5)} | "
                f"{_fmt_optional_score(summary.avg_judge_pedagogy_5)} | "
                f"{_fmt_optional_score(summary.avg_judge_overall_5)} | "
                f"{graded_label} | {summary.mean_latency_ms:.1f} | "
                f"{summary.p95_latency_ms:.1f} | {summary.mean_tokens_per_second:.1f} | "
                f"${summary.estimated_cost_per_1k_queries_usd:.4f} | {summary.error_count} |"
            )
    else:
        lines.extend(
            [
                "| Profile | Model | Offline | Accuracy (/5) | Fact Recall | Hallucination Cases | SDG 4 Pedagogy (/5) | Mean Latency (ms) | p95 Latency (ms) | Tokens/s | Est. Cost / 1k Queries (USD) | Errors |",
                "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
            ]
        )
        for summary in report.summaries:
            lines.append(
                f"| `{summary.profile}` | `{summary.model_id}` | "
                f"{'Yes' if summary.offline_capable else 'No'} | "
                f"{summary.avg_accuracy_5:.2f} | {summary.avg_fact_recall:.1%} | "
                f"{summary.hallucination_case_count}/{summary.total_cases} | "
                f"{summary.avg_pedagogy_5:.2f} | {summary.mean_latency_ms:.1f} | "
                f"{summary.p95_latency_ms:.1f} | {summary.mean_tokens_per_second:.1f} | "
                f"${summary.estimated_cost_per_1k_queries_usd:.4f} | {summary.error_count} |"
            )

    lines.extend(["", "## Per-Case Breakdown", ""])
    if has_judge:
        lines.append(
            "| Case ID | Category | Profile | Det. Accuracy (/5) | Fact Recall | Hallucinations | Det. Pedagogy (/5) | Judge Accuracy (/5) | Judge Pedagogy (/5) | Judge Overall (/5) | Judge Status | Latency (ms) |"
        )
        lines.append(
            "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
        )
        for item in report.results:
            hallucinations = _escape_md_cell(
                ", ".join(item.accuracy.hallucination_flags) or "None"
            )
            if item.judge is not None:
                judge_acc = f"{item.judge.accuracy_score_5:.2f}"
                judge_ped = f"{item.judge.pedagogy_score_5:.2f}"
                judge_overall = f"{item.judge.overall_score_5:.2f}"
                judge_status = f"`{item.judge.status}`"
            else:
                judge_acc = "N/A"
                judge_ped = "N/A"
                judge_overall = "N/A"
                judge_status = "`skipped`"
            lines.append(
                f"| `{item.case_id}` | `{item.category}` | `{item.profile}` | "
                f"{item.accuracy.accuracy_score_5:.2f} | {item.accuracy.fact_recall:.1%} | "
                f"{hallucinations} | {item.pedagogy.pedagogy_avg_5:.2f} | "
                f"{judge_acc} | {judge_ped} | {judge_overall} | {judge_status} | "
                f"{item.operational.latency_ms:.1f} |"
            )

        lines.extend(["", "## LLM-as-a-Judge Rationales", ""])
        lines.append(
            "| Case ID | Profile | Status | Judge Overall (/5) | Detected Issues | Rationale |"
        )
        lines.append("| :--- | :--- | :---: | :---: | :--- | :--- |")
        for item in report.results:
            if item.judge is None:
                continue
            issues = _escape_md_cell(", ".join(item.judge.detected_issues) or "None")
            rationale = _escape_md_cell(item.judge.rationale or item.judge_notes or "—")
            lines.append(
                f"| `{item.case_id}` | `{item.profile}` | `{item.judge.status}` | "
                f"{item.judge.overall_score_5:.2f} | {issues} | {rationale} |"
            )
    else:
        lines.append(
            "| Case ID | Category | Profile | Accuracy (/5) | Fact Recall | Hallucinations | Pedagogy (/5) | Latency (ms) |"
        )
        lines.append("| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |")

        for item in report.results:
            hallucinations = _escape_md_cell(
                ", ".join(item.accuracy.hallucination_flags) or "None"
            )
            lines.append(
                f"| `{item.case_id}` | `{item.category}` | `{item.profile}` | "
                f"{item.accuracy.accuracy_score_5:.2f} | {item.accuracy.fact_recall:.1%} | "
                f"{hallucinations} | {item.pedagogy.pedagogy_avg_5:.2f} | "
                f"{item.operational.latency_ms:.1f} |"
            )

    lines.append("")
    return "\n".join(lines)
