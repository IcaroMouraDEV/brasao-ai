"""Blind LLM-as-a-judge grading engine for Local vs. Remote benchmark evaluation."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from app.models import format_ollama_litellm_model
from evals.rubrics import (
    BenchmarkCase,
    HistoricalAccuracyScore,
    JudgeScore,
    JudgeVerdict,
    PedagogicalQualityScore,
)

JudgeInvoker = Callable[[str, str], Awaitable[JudgeVerdict | dict[str, Any] | str]]

_DRY_RUN_TAG_RE = re.compile(r"\[DRY-RUN:(?:remote|local)\]\s*", re.IGNORECASE)
_CODE_FENCE_RE = re.compile(
    r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.IGNORECASE | re.DOTALL
)
_GENERIC_MODEL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bollama_chat/[a-zA-Z0-9._:-]+", re.IGNORECASE),
    re.compile(r"\bollama/[a-zA-Z0-9._:-]+", re.IGNORECASE),
    re.compile(r"\bgemini-[a-zA-Z0-9._-]+", re.IGNORECASE),
    re.compile(r"\bgemma\d*(?::[a-zA-Z0-9._-]+)?\b", re.IGNORECASE),
    re.compile(r"\bqwen\d*(?::[a-zA-Z0-9._-]+)?\b", re.IGNORECASE),
    re.compile(r"\bllama\d*(?:\.\d+)?(?::[a-zA-Z0-9._-]+)?\b", re.IGNORECASE),
    re.compile(r"\b(?:gemini|ollama|litellm)\b", re.IGNORECASE),
)

_LEVEL_RUBRIC_GUIDANCE: dict[str, str] = {
    "fundamental_2": (
        "Ensino Fundamental II (6th-9th grade): accessible vocabulary, concrete examples, "
        "clear cause-and-effect explanations without excessive academic jargon."
    ),
    "ensino_medio": (
        "Ensino Medio / ENEM (High School): analytical depth, structural connections, "
        "historiographical nuance, and primary-source or exam-oriented critical reasoning."
    ),
    "geral": (
        "General public: clear, engaging, historically rigorous, and balanced overview."
    ),
}

_genai_client: Any = None


def is_rate_limit_error(exc: Exception) -> bool:
    """Return True when an exception indicates an HTTP 429 / quota rate-limit response."""
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(
        token in text for token in ("429", "resource_exhausted", "rate limit", "quota")
    )


def _clamp_score(value: float) -> float:
    """Clamp a raw numeric score into [0.0, 5.0] rounded to two decimals."""
    return round(max(0.0, min(5.0, float(value))), 2)


def sanitize_response_for_blind_judging(
    response_text: str,
    extra_identifiers: Sequence[str] = (),
) -> str:
    """Strip model/profile identifiers from a candidate response to ensure blind grading."""
    sanitized = _DRY_RUN_TAG_RE.sub("", response_text)

    # Redact longest identifiers first so composite tags (e.g. ollama_chat/gemma3:4b)
    # are replaced cleanly as a single token.
    ordered_ids = sorted(
        {ident.strip() for ident in extra_identifiers if ident and ident.strip()},
        key=len,
        reverse=True,
    )
    for ident in ordered_ids:
        sanitized = re.sub(
            re.escape(ident), "[MODEL_REDACTED]", sanitized, flags=re.IGNORECASE
        )

    for pattern in _GENERIC_MODEL_PATTERNS:
        sanitized = pattern.sub("[MODEL_REDACTED]", sanitized)

    return sanitized.strip()


def build_blind_judge_prompt(
    case: BenchmarkCase,
    candidate_response: str,
    *,
    extra_identifiers: Sequence[str] = (),
) -> str:
    """Build a blind, checklist-grounded LLM-as-a-judge prompt without model metadata.

    Note: By design this function never accepts `profile` or `model_id` for inclusion in
    the prompt. `extra_identifiers` is used strictly to redact accidental self-mentions
    from `candidate_response`, mitigating judge self-preference bias.
    """
    sanitized_response = sanitize_response_for_blind_judging(
        candidate_response,
        extra_identifiers=extra_identifiers,
    )
    expected_facts_block = "\n".join(
        f"  {idx}. {fact}" for idx, fact in enumerate(case.expected_facts, start=1)
    )
    forbidden_block = "\n".join(
        f"  {idx}. {trap}"
        for idx, trap in enumerate(case.forbidden_hallucinations, start=1)
    )
    bncc_list = ", ".join(case.bncc_codes) if case.bncc_codes else "N/A"
    level_guidance = _LEVEL_RUBRIC_GUIDANCE.get(
        case.target_level, _LEVEL_RUBRIC_GUIDANCE["geral"]
    )

    return (
        "You are an independent, rigorous Brazilian History educator and UN SDG 4 "
        "(Quality Education) evaluator grading an ANONYMOUS tutor response.\n\n"
        "BLIND & ANTI-BIAS PROTOCOL:\n"
        "- The author model identity is intentionally hidden. Do NOT infer or favor any "
        "model family, stylistic signature, or sheer response length.\n"
        "- Grade strictly against the explicit per-case checklist below on a 0.0 to 5.0 "
        "scale (0 = completely failed/fabricated, 3 = partial/mixed, 5 = exemplary).\n"
        "- In misconception_trap and epistemic_honesty cases, quoting a myth or fictional "
        "premise ONLY to refute or debunk it is GOOD; endorsing or repeating it as fact "
        "must severely penalize hallucination_resistance_score_5 and grounding_score_5.\n\n"
        "PER-CASE CHECKLIST:\n"
        f"- Case ID: {case.id}\n"
        f"- Category: {case.category}\n"
        f"- Target Grade Level: {case.target_level} ({level_guidance})\n"
        f"- BNCC Codes: {bncc_list}\n"
        f"- Student Prompt (pt-BR): {case.prompt}\n"
        f"- Reference Historical Summary (ground truth): {case.reference_summary}\n"
        f"- Expected Facts Checklist:\n{expected_facts_block}\n"
        f"- Forbidden Hallucinations / Historical Myths Checklist:\n{forbidden_block}\n\n"
        "RUBRIC DIMENSIONS (0.0 to 5.0 each):\n"
        "1. Historical Accuracy & Hallucination Resistance:\n"
        "   - fact_coverage_score_5: Coverage and accuracy of expected_facts.\n"
        "   - hallucination_resistance_score_5: Absence of anachronisms, fabricated "
        "dates/events, and endorsement of forbidden_hallucinations.\n"
        "   - grounding_score_5: Alignment with reference_summary and epistemic honesty.\n"
        "2. UN SDG 4 Pedagogical Quality:\n"
        "   - clarity_score_5: Clarity, structure, and natural Brazilian Portuguese (pt-BR).\n"
        "   - level_adaptation_score_5: Calibration of language and analytical depth to "
        f"target_level ({case.target_level}).\n"
        "   - critical_thinking_score_5: Encouragement of critical thinking, active "
        "reflection, and inclusion of multiple historical perspectives (Indigenous, "
        "Afro-Brazilian, women, regional, institutional — SDG 4.7).\n\n"
        "ANONYMOUS TUTOR RESPONSE TO GRADE:\n"
        "<anonymous_tutor_response>\n"
        f"{sanitized_response}\n"
        "</anonymous_tutor_response>\n\n"
        "OUTPUT FORMAT:\n"
        "Return ONLY a valid JSON object matching this exact schema (no markdown prose "
        "outside the JSON):\n"
        "{\n"
        '  "fact_coverage_score_5": <float 0-5>,\n'
        '  "hallucination_resistance_score_5": <float 0-5>,\n'
        '  "grounding_score_5": <float 0-5>,\n'
        '  "clarity_score_5": <float 0-5>,\n'
        '  "level_adaptation_score_5": <float 0-5>,\n'
        '  "critical_thinking_score_5": <float 0-5>,\n'
        '  "accuracy_rationale": "<concise rationale in English>",\n'
        '  "pedagogy_rationale": "<concise rationale in English>",\n'
        '  "detected_issues": ["<specific error/anachronism/trap if any>"]\n'
        "}\n"
    )


def _extract_json_payload(raw_text: str) -> str:
    """Extract a JSON object string from raw text or a markdown fenced block."""
    cleaned = raw_text.strip()
    if not cleaned:
        raise ValueError("Judge returned an empty response.")

    fence_match = _CODE_FENCE_RE.match(cleaned)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    if cleaned.startswith("{") and cleaned.endswith("}"):
        return cleaned

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        return cleaned[start : end + 1]

    raise ValueError("Judge response did not contain a valid JSON object.")


def parse_judge_verdict(
    raw_output: JudgeVerdict | dict[str, Any] | str,
) -> JudgeVerdict:
    """Parse and validate a structured `JudgeVerdict` from a judge model output."""
    if isinstance(raw_output, JudgeVerdict):
        return raw_output
    if isinstance(raw_output, dict):
        return JudgeVerdict.model_validate(raw_output)
    if isinstance(raw_output, str):
        payload = json.loads(_extract_json_payload(raw_output))
        if not isinstance(payload, dict):
            raise ValueError("Judge JSON payload must be an object.")
        return JudgeVerdict.model_validate(payload)
    raise TypeError(f"Unsupported judge output type: {type(raw_output).__name__}")


def build_judge_score_from_verdict(
    verdict: JudgeVerdict,
    judge_model: str,
) -> JudgeScore:
    """Convert a validated `JudgeVerdict` into a bounded, aggregated `JudgeScore`."""
    fact_cov = _clamp_score(verdict.fact_coverage_score_5)
    hallucination_res = _clamp_score(verdict.hallucination_resistance_score_5)
    grounding = _clamp_score(verdict.grounding_score_5)
    accuracy_avg = round((fact_cov + hallucination_res + grounding) / 3.0, 2)

    clarity = _clamp_score(verdict.clarity_score_5)
    adaptation = _clamp_score(verdict.level_adaptation_score_5)
    critical = _clamp_score(verdict.critical_thinking_score_5)
    pedagogy_avg = round((clarity + adaptation + critical) / 3.0, 2)

    overall = round((accuracy_avg + pedagogy_avg) / 2.0, 2)

    acc_rationale = verdict.accuracy_rationale.strip()
    ped_rationale = verdict.pedagogy_rationale.strip()
    rationale_parts: list[str] = []
    if acc_rationale:
        rationale_parts.append(f"Accuracy: {acc_rationale}")
    if ped_rationale:
        rationale_parts.append(f"Pedagogy: {ped_rationale}")
    combined_rationale = " | ".join(rationale_parts)

    return JudgeScore(
        judge_model=judge_model,
        status="ok",
        fact_coverage_score_5=fact_cov,
        hallucination_resistance_score_5=hallucination_res,
        grounding_score_5=grounding,
        accuracy_score_5=accuracy_avg,
        clarity_score_5=clarity,
        level_adaptation_score_5=adaptation,
        critical_thinking_score_5=critical,
        pedagogy_score_5=pedagogy_avg,
        overall_score_5=overall,
        accuracy_rationale=acc_rationale,
        pedagogy_rationale=ped_rationale,
        rationale=combined_rationale,
        detected_issues=[
            issue.strip() for issue in verdict.detected_issues if issue.strip()
        ],
        error=None,
    )


def build_fallback_judge_score(
    accuracy: HistoricalAccuracyScore,
    pedagogy: PedagogicalQualityScore,
    judge_model: str,
    reason: str,
) -> JudgeScore:
    """Create a transparent fallback `JudgeScore` from deterministic rubric scores."""
    fact_cov = _clamp_score(accuracy.fact_recall * 5.0)
    hallucination_res = _clamp_score(5.0 - 2.0 * len(accuracy.hallucination_flags))
    grounding = _clamp_score(accuracy.accuracy_score_5)
    accuracy_avg = _clamp_score(accuracy.accuracy_score_5)

    clarity = _clamp_score(pedagogy.clarity_score_5)
    adaptation = _clamp_score(pedagogy.level_adaptation_score_5)
    critical = _clamp_score(pedagogy.critical_thinking_score_5)
    pedagogy_avg = _clamp_score(pedagogy.pedagogy_avg_5)
    overall = round((accuracy_avg + pedagogy_avg) / 2.0, 2)

    return JudgeScore(
        judge_model=judge_model,
        status="fallback",
        fact_coverage_score_5=fact_cov,
        hallucination_resistance_score_5=hallucination_res,
        grounding_score_5=grounding,
        accuracy_score_5=accuracy_avg,
        clarity_score_5=clarity,
        level_adaptation_score_5=adaptation,
        critical_thinking_score_5=critical,
        pedagogy_score_5=pedagogy_avg,
        overall_score_5=overall,
        accuracy_rationale=f"Deterministic accuracy fallback ({reason}).",
        pedagogy_rationale=f"Heuristic pedagogy fallback ({reason}).",
        rationale=f"Fallback to deterministic scores due to judge failure: {reason}",
        detected_issues=list(accuracy.hallucination_flags),
        error=reason,
    )


def build_dry_run_judge_score(
    case: BenchmarkCase,
    accuracy: HistoricalAccuracyScore,
    pedagogy: PedagogicalQualityScore,
    judge_model: str,
) -> JudgeScore:
    """Produce a deterministic offline simulated `JudgeScore` for `--dry-run --use-judge`."""
    verdict = JudgeVerdict(
        fact_coverage_score_5=accuracy.fact_recall * 5.0,
        hallucination_resistance_score_5=5.0 - 2.0 * len(accuracy.hallucination_flags),
        grounding_score_5=accuracy.accuracy_score_5,
        clarity_score_5=pedagogy.clarity_score_5,
        level_adaptation_score_5=pedagogy.level_adaptation_score_5,
        critical_thinking_score_5=pedagogy.critical_thinking_score_5,
        accuracy_rationale=(
            f"[DRY-RUN] Matched {accuracy.expected_facts_matched}/"
            f"{accuracy.expected_facts_total} expected facts with "
            f"{len(accuracy.hallucination_flags)} hallucination flags."
        ),
        pedagogy_rationale=(
            f"[DRY-RUN] Simulated SDG 4 pedagogy assessment for level '{case.target_level}'."
        ),
        detected_issues=list(accuracy.hallucination_flags),
    )
    return build_judge_score_from_verdict(verdict, judge_model=judge_model)


def _get_genai_client() -> Any:
    """Return a lazily initialized `google.genai.Client` shared across judge calls."""
    global _genai_client
    if _genai_client is None:
        from google import genai

        _genai_client = genai.Client()
    return _genai_client


async def _invoke_judge_model(
    prompt: str,
    judge_model: str,
) -> JudgeVerdict | str:
    """Call the configured judge model (native Gemini or Ollama via LiteLLM)."""
    if judge_model.startswith(("ollama_chat/", "ollama/")):
        import litellm

        response = await litellm.acompletion(
            model=format_ollama_litellm_model(judge_model),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        choices = getattr(response, "choices", None) or []
        if not choices:
            return ""
        message = getattr(choices[0], "message", None)
        return str(getattr(message, "content", "") or "")

    from google.genai import types

    client = _get_genai_client()
    response = await client.aio.models.generate_content(
        model=judge_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=JudgeVerdict,
        ),
    )
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, JudgeVerdict):
        return parsed
    return str(getattr(response, "text", "") or "")


async def grade_case_with_judge(
    case: BenchmarkCase,
    response_text: str,
    accuracy: HistoricalAccuracyScore,
    pedagogy: PedagogicalQualityScore,
    judge_model: str,
    *,
    extra_identifiers: Sequence[str] = (),
    candidate_error: str | None = None,
    dry_run: bool = False,
    judge_invoker: JudgeInvoker | None = None,
    max_rate_limit_retries: int = 4,
    base_backoff_s: float = 15.0,
) -> JudgeScore:
    """Grade one candidate response blindly with an LLM judge and deterministic fallback."""
    if candidate_error:
        return build_fallback_judge_score(
            accuracy=accuracy,
            pedagogy=pedagogy,
            judge_model=judge_model,
            reason=f"Candidate agent failed: {candidate_error}",
        )
    if not response_text.strip():
        return build_fallback_judge_score(
            accuracy=accuracy,
            pedagogy=pedagogy,
            judge_model=judge_model,
            reason="Empty candidate response",
        )
    if dry_run and judge_invoker is None:
        return build_dry_run_judge_score(
            case=case,
            accuracy=accuracy,
            pedagogy=pedagogy,
            judge_model=judge_model,
        )

    prompt = build_blind_judge_prompt(
        case=case,
        candidate_response=response_text,
        extra_identifiers=extra_identifiers,
    )
    invoker = judge_invoker or _invoke_judge_model
    is_local_judge = judge_model.startswith(("ollama_chat/", "ollama/"))

    attempt = 0
    try:
        while True:
            try:
                raw_output = await invoker(prompt, judge_model)
                break
            except Exception as exc:
                if (
                    is_local_judge
                    or not is_rate_limit_error(exc)
                    or attempt >= max_rate_limit_retries
                ):
                    raise
                delay = base_backoff_s * (2**attempt)
                attempt += 1
                await asyncio.sleep(delay)

        verdict = parse_judge_verdict(raw_output)
        return build_judge_score_from_verdict(verdict, judge_model=judge_model)
    except Exception as exc:
        return build_fallback_judge_score(
            accuracy=accuracy,
            pedagogy=pedagogy,
            judge_model=judge_model,
            reason=f"{type(exc).__name__}: {exc}",
        )
