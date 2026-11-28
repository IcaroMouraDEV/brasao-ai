"""Local Web UI API routes and ADK workflow execution helpers for Brasão AI.

Exposes zero-cloud-dependency interactive endpoints (`GET /ui`, `GET /demo`,
`GET /api/ui/config`, `POST /api/chat`, `POST /api/compare`) alongside existing
ADK, A2A, and Reasoning Engine routes in `app/fast_api_app.py`.
"""

from __future__ import annotations

import asyncio
import re
import time
import uuid
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, field_validator

from app.agent import TERMINAL_NODE_NAME, create_history_tutor_agent
from app.app_utils.sanitization import sanitize_student_response
from app.config import ModelProfile, Settings, load_settings
from app.ui import PROMPT_PRESETS, render_web_ui_html

SchoolLevel = Literal["fundamental_2", "ensino_medio", "geral"]
CompareMode = Literal["local_vs_remote", "local_rag_ablation", "remote_rag_ablation"]

SCHOOL_LEVEL_LABELS: dict[SchoolLevel, str] = {
    "fundamental_2": "Ensino Fundamental II (6º ao 9º ano)",
    "ensino_medio": "Ensino Médio / ENEM",
    "geral": "Público Geral",
}

# Reference pricing estimates (USD per 1M tokens) matching `evals/rubrics.py`.
# Local Ollama inference always has $0.00 marginal API cost.
MODEL_COST_PER_1M_TOKENS_USD: dict[str, tuple[float, float]] = {
    "gemini-3.8-flash": (0.75, 3.75),
    "gemini-3-flash-preview": (0.50, 3.00),
    "gemini-3.5-flash-lite": (0.10, 0.40),
    "gemini-4-hydrogen": (1.50, 7.50),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-2.0-flash": (0.10, 0.40),
}

_BNCC_CODE_RE = re.compile(r"\b(?:EF\d{2}[A-Z]{2}\d{2}|EM\d{2}[A-Z]{2,3}\d{3})\b")


class CitationItem(BaseModel):
    """Structured citation extracted from curated corpus/RAG tool responses."""

    source_id: str
    document: str
    bncc_codes: list[str] = Field(default_factory=list)


class ChatTelemetry(BaseModel):
    """Operational telemetry returned alongside every interactive tutor response."""

    model_id: str
    latency_ms: float = Field(ge=0.0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    tokens_per_second: float = Field(default=0.0, ge=0.0)
    estimated_cost_usd: float = Field(default=0.0, ge=0.0)
    offline_capable: bool


class ChatRequest(BaseModel):
    """Request payload for `POST /api/chat`."""

    prompt: str = Field(..., min_length=1, max_length=4000)
    profile: ModelProfile | None = None
    school_level: SchoolLevel = "geral"
    enable_tools: bool = True

    @field_validator("prompt")
    @classmethod
    def _strip_non_empty_prompt(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Prompt must not be empty or whitespace-only.")
        return cleaned


class ChatResponse(BaseModel):
    """Response payload for `POST /api/chat`."""

    status: Literal["ok", "error"] = "ok"
    answer: str
    profile: ModelProfile
    school_level: SchoolLevel
    enable_tools: bool
    telemetry: ChatTelemetry
    bncc_codes: list[str] = Field(default_factory=list)
    citations: list[CitationItem] = Field(default_factory=list)
    error: str | None = None


class CompareRequest(BaseModel):
    """Request payload for `POST /api/compare`."""

    prompt: str = Field(..., min_length=1, max_length=4000)
    school_level: SchoolLevel = "geral"
    mode: CompareMode = "local_vs_remote"
    enable_tools: bool = True

    @field_validator("prompt")
    @classmethod
    def _strip_non_empty_prompt(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Prompt must not be empty or whitespace-only.")
        return cleaned


class CompareSideResult(BaseModel):
    """One labeled side of a side-by-side comparison."""

    label: str
    response: ChatResponse


class CompareResponse(BaseModel):
    """Response payload for `POST /api/compare`."""

    mode: CompareMode
    prompt: str
    school_level: SchoolLevel
    left: CompareSideResult
    right: CompareSideResult
    latency_delta_ms: float
    cost_delta_usd: float


class UiPreset(BaseModel):
    """Preset Brazilian History prompt shown in the local web UI."""

    id: str
    title: str
    category: str
    school_level: SchoolLevel
    bncc_codes: list[str] = Field(default_factory=list)
    prompt: str


class UiConfigResponse(BaseModel):
    """Metadata payload served by `GET /api/ui/config`."""

    default_profile: ModelProfile
    local_model_id: str
    remote_model_id: str
    ollama_api_base: str
    school_levels: dict[SchoolLevel, str]
    presets: list[UiPreset]


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


def format_student_prompt(prompt: str, school_level: SchoolLevel) -> str:
    """Attach a pt-BR school-level hint identically across local and remote profiles."""
    cleaned = prompt.strip()
    level_label = SCHOOL_LEVEL_LABELS[school_level]
    return f"[Nível escolar solicitado: {level_label} ({school_level})]\n{cleaned}"


def extract_workflow_output(
    events: list[Any],
    terminal_author: str | None = TERMINAL_NODE_NAME,
) -> tuple[str, int, int]:
    """Extract the sanitized terminal student response and cumulative token counts."""
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
                if text and not bool(getattr(part, "thought", False)):
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


def extract_citations_and_bncc(
    events: list[Any],
    answer_text: str,
) -> tuple[list[str], list[CitationItem]]:
    """Extract deduplicated BNCC codes and verified source citations from ADK events."""
    bncc_seen: set[str] = set()
    bncc_codes: list[str] = []
    citation_seen: set[str] = set()
    citations: list[CitationItem] = []

    def _add_bncc(code: str) -> None:
        cleaned = str(code).strip().upper()
        if cleaned and cleaned not in bncc_seen:
            bncc_seen.add(cleaned)
            bncc_codes.append(cleaned)

    def _add_citation(
        source_id: str,
        document: str,
        codes: list[str] | None = None,
    ) -> None:
        sid = str(source_id).strip()
        if not sid or sid in citation_seen:
            return
        citation_seen.add(sid)
        normalized_codes = [
            str(c).strip().upper() for c in (codes or []) if str(c).strip()
        ]
        for code in normalized_codes:
            _add_bncc(code)
        citations.append(
            CitationItem(
                source_id=sid,
                document=str(document).strip() or sid,
                bncc_codes=normalized_codes,
            )
        )

    for event in events:
        content = getattr(event, "content", None)
        parts = getattr(content, "parts", None) if content is not None else None
        if not parts:
            continue
        for part in parts:
            fn_resp = getattr(part, "function_response", None)
            payload = (
                getattr(fn_resp, "response", None) if fn_resp is not None else None
            )
            if not isinstance(payload, dict):
                continue

            for code in payload.get("bncc_codes") or []:
                _add_bncc(str(code))

            for raw_cit in payload.get("citations") or []:
                if isinstance(raw_cit, dict):
                    _add_citation(
                        str(raw_cit.get("source_id", "")),
                        str(raw_cit.get("document", "")),
                        list(raw_cit.get("bncc_codes") or []),
                    )

            for list_key in ("results", "events"):
                for item in payload.get(list_key) or []:
                    if not isinstance(item, dict):
                        continue
                    item_codes = [str(c) for c in (item.get("bncc_codes") or [])]
                    for code in item_codes:
                        _add_bncc(code)
                    for raw_cit in item.get("citations") or []:
                        if isinstance(raw_cit, dict):
                            _add_citation(
                                str(raw_cit.get("source_id", "")),
                                str(raw_cit.get("document", "")),
                                list(raw_cit.get("bncc_codes") or []),
                            )
                    if item.get("source_id"):
                        _add_citation(
                            str(item["source_id"]),
                            str(
                                item.get("title")
                                or item.get("document")
                                or item["source_id"]
                            ),
                            item_codes,
                        )

    for match in _BNCC_CODE_RE.findall(answer_text):
        _add_bncc(match)

    return bncc_codes, citations


async def invoke_tutor_workflow(
    settings: Settings,
    *,
    prompt: str,
    profile: ModelProfile,
    school_level: SchoolLevel,
    enable_tools: bool = True,
) -> ChatResponse:
    """Execute one student prompt through the ADK 2.0 `Workflow` with telemetry."""
    from google.adk.runners import InMemoryRunner
    from google.genai import types

    agent = create_history_tutor_agent(
        settings=settings,
        profile=profile,
        enable_tools=enable_tools,
    )
    mode_tag = "rag" if enable_tools else "notools"
    app_name = f"brasao_ai_ui_{profile}_{mode_tag}"
    user_id = "local_web_ui_student"
    session_id = f"ui_{profile}_{mode_tag}_{uuid.uuid4().hex[:12]}"

    runner = InMemoryRunner(agent=agent, app_name=app_name)
    session = await runner.session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
    )

    formatted_prompt = format_student_prompt(prompt, school_level)
    content = types.Content(
        role="user",
        parts=[types.Part.from_text(text=formatted_prompt)],
    )
    events: list[Any] = []

    started_at = time.perf_counter()
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session.id,
        new_message=content,
    ):
        events.append(event)
    latency_ms = (time.perf_counter() - started_at) * 1000.0

    answer, input_tokens, output_tokens = extract_workflow_output(
        events,
        terminal_author=TERMINAL_NODE_NAME,
    )
    if output_tokens == 0 and answer:
        output_tokens = int(len(answer.split()) * 1.3)
    if input_tokens == 0:
        input_tokens = int(len(formatted_prompt.split()) * 1.3)

    model_id = settings.resolved_model_label(profile)
    offline_capable = settings.is_offline_capable(profile)
    tps = round(output_tokens / (latency_ms / 1000.0), 2) if latency_ms > 0 else 0.0
    cost_usd = round(
        estimate_query_cost_usd(profile, model_id, input_tokens, output_tokens),
        6,
    )
    bncc_codes, citations = extract_citations_and_bncc(events, answer)

    return ChatResponse(
        status="ok",
        answer=answer,
        profile=profile,
        school_level=school_level,
        enable_tools=enable_tools,
        telemetry=ChatTelemetry(
            model_id=model_id,
            latency_ms=round(latency_ms, 2),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tokens_per_second=tps,
            estimated_cost_usd=cost_usd,
            offline_capable=offline_capable,
        ),
        bncc_codes=bncc_codes,
        citations=citations,
    )


async def _run_compare_side(
    settings: Settings,
    *,
    label: str,
    prompt: str,
    profile: ModelProfile,
    school_level: SchoolLevel,
    enable_tools: bool,
) -> CompareSideResult:
    """Run one side of a comparison, returning an error payload instead of aborting both sides."""
    try:
        response = await invoke_tutor_workflow(
            settings,
            prompt=prompt,
            profile=profile,
            school_level=school_level,
            enable_tools=enable_tools,
        )
    except Exception as exc:
        response = ChatResponse(
            status="error",
            answer="",
            profile=profile,
            school_level=school_level,
            enable_tools=enable_tools,
            telemetry=ChatTelemetry(
                model_id=settings.resolved_model_label(profile),
                latency_ms=0.0,
                input_tokens=0,
                output_tokens=0,
                tokens_per_second=0.0,
                estimated_cost_usd=0.0,
                offline_capable=settings.is_offline_capable(profile),
            ),
            error=f"{type(exc).__name__}: {exc}",
        )
    return CompareSideResult(label=label, response=response)


def attach_web_ui_routes(app: FastAPI) -> None:
    """Register local Web UI pages and interactive chat/compare API endpoints."""

    @app.get("/ui", response_class=HTMLResponse, include_in_schema=False)
    @app.get("/demo", response_class=HTMLResponse, include_in_schema=False)
    async def serve_web_ui() -> HTMLResponse:
        settings = load_settings()
        return HTMLResponse(content=render_web_ui_html(settings))

    @app.get("/api/ui/config", response_model=UiConfigResponse, tags=["web-ui"])
    async def get_ui_config() -> UiConfigResponse:
        settings = load_settings()
        return UiConfigResponse(
            default_profile=settings.model_profile,
            local_model_id=settings.resolved_model_label("local"),
            remote_model_id=settings.resolved_model_label("remote"),
            ollama_api_base=settings.ollama_api_base,
            school_levels=SCHOOL_LEVEL_LABELS,
            presets=[UiPreset.model_validate(item) for item in PROMPT_PRESETS],
        )

    @app.post("/api/chat", response_model=ChatResponse, tags=["web-ui"])
    async def api_chat(request: ChatRequest) -> ChatResponse:
        settings = load_settings()
        active_profile: ModelProfile = request.profile or settings.model_profile
        try:
            return await invoke_tutor_workflow(
                settings,
                prompt=request.prompt,
                profile=active_profile,
                school_level=request.school_level,
                enable_tools=request.enable_tools,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=(
                    f"Failed to execute {active_profile} profile "
                    f"({settings.resolved_model_label(active_profile)}): "
                    f"{type(exc).__name__}: {exc}"
                ),
            ) from exc

    @app.post("/api/compare", response_model=CompareResponse, tags=["web-ui"])
    async def api_compare(request: CompareRequest) -> CompareResponse:
        settings = load_settings()

        if request.mode == "local_vs_remote":
            left, right = await asyncio.gather(
                _run_compare_side(
                    settings,
                    label="Local (Ollama)",
                    prompt=request.prompt,
                    profile="local",
                    school_level=request.school_level,
                    enable_tools=request.enable_tools,
                ),
                _run_compare_side(
                    settings,
                    label="Remoto (Gemini)",
                    prompt=request.prompt,
                    profile="remote",
                    school_level=request.school_level,
                    enable_tools=request.enable_tools,
                ),
            )
        elif request.mode == "local_rag_ablation":
            # Run sequentially on local hardware so GPU/CPU contention does not distort latency.
            left = await _run_compare_side(
                settings,
                label="Local (Ollama) — RAG Ativo",
                prompt=request.prompt,
                profile="local",
                school_level=request.school_level,
                enable_tools=True,
            )
            right = await _run_compare_side(
                settings,
                label="Local (Ollama) — Sem RAG",
                prompt=request.prompt,
                profile="local",
                school_level=request.school_level,
                enable_tools=False,
            )
        else:
            left, right = await asyncio.gather(
                _run_compare_side(
                    settings,
                    label="Remoto (Gemini) — RAG Ativo",
                    prompt=request.prompt,
                    profile="remote",
                    school_level=request.school_level,
                    enable_tools=True,
                ),
                _run_compare_side(
                    settings,
                    label="Remoto (Gemini) — Sem RAG",
                    prompt=request.prompt,
                    profile="remote",
                    school_level=request.school_level,
                    enable_tools=False,
                ),
            )

        latency_delta = round(
            left.response.telemetry.latency_ms - right.response.telemetry.latency_ms,
            2,
        )
        cost_delta = round(
            left.response.telemetry.estimated_cost_usd
            - right.response.telemetry.estimated_cost_usd,
            6,
        )
        return CompareResponse(
            mode=request.mode,
            prompt=request.prompt,
            school_level=request.school_level,
            left=left,
            right=right,
            latency_delta_ms=latency_delta,
            cost_delta_usd=cost_delta,
        )
