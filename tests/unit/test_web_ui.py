"""Offline FastAPI TestClient unit tests for Brasão AI local web UI and demo API."""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app import web_api
from app.config import ModelProfile, Settings
from app.fast_api_app import app as fastapi_app
from app.workflow import (
    FACT_VERIFIER_NODE_NAME,
    HISTORICAL_RESEARCHER_NODE_NAME,
    PEDAGOGICAL_ADAPTER_NODE_NAME,
    TOPIC_ROUTER_NODE_NAME,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MAKEFILE_PATH = REPO_ROOT / "Makefile"


def _make_fake_event(
    *,
    author: str,
    text: str | None = None,
    function_response: dict[str, Any] | None = None,
    prompt_tokens: int = 0,
    output_tokens: int = 0,
    partial: bool = False,
) -> Any:
    parts: list[Any] = []
    if text is not None:
        parts.append(SimpleNamespace(text=text, thought=False, function_response=None))
    if function_response is not None:
        parts.append(
            SimpleNamespace(
                text=None,
                thought=False,
                function_response=SimpleNamespace(response=function_response),
            )
        )
    usage = SimpleNamespace(
        prompt_token_count=prompt_tokens,
        candidates_token_count=output_tokens,
    )
    return SimpleNamespace(
        author=author,
        partial=partial,
        usage_metadata=usage,
        content=SimpleNamespace(parts=parts),
    )


def _fake_chat_response(
    *,
    profile: ModelProfile,
    school_level: web_api.SchoolLevel,
    enable_tools: bool,
    answer: str,
    latency_ms: float = 350.0,
    input_tokens: int = 120,
    output_tokens: int = 180,
) -> web_api.ChatResponse:
    settings = Settings()
    model_id = settings.resolved_model_label(profile)
    cost = round(
        web_api.estimate_query_cost_usd(profile, model_id, input_tokens, output_tokens),
        6,
    )
    return web_api.ChatResponse(
        status="ok",
        answer=answer,
        profile=profile,
        school_level=school_level,
        enable_tools=enable_tools,
        telemetry=web_api.ChatTelemetry(
            model_id=model_id,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tokens_per_second=round(output_tokens / (latency_ms / 1000.0), 2),
            estimated_cost_usd=cost,
            offline_capable=settings.is_offline_capable(profile),
        ),
        bncc_codes=["EF07HI10", "EF07HI12"],
        citations=[
            web_api.CitationItem(
                source_id="corpus:colonizacao_povos_indigenas",
                document="Corpus Curado BNCC — Povos Indígenas e Colonização",
                bncc_codes=["EF07HI10", "EF07HI12"],
            )
        ],
    )


def test_ui_routes_serve_self_contained_offline_html() -> None:
    client = TestClient(fastapi_app)

    for path in ("/ui", "/demo"):
        resp = client.get(path)
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        body = resp.text

        assert "Brasão AI" in body
        assert "fundamental_2" in body
        assert "ensino_medio" in body
        assert "geral" in body
        assert "local_vs_remote" in body
        assert "local_rag_ablation" in body
        assert "/api/chat" in body
        assert "/api/compare" in body

        # Zero external CDN dependencies so the UI works 100% offline in schools.
        assert "http://" not in body
        assert "https://" not in body
        assert "<link " not in body.lower()


def test_ui_config_endpoint_exposes_models_and_presets() -> None:
    client = TestClient(fastapi_app)
    resp = client.get("/api/ui/config")
    assert resp.status_code == 200
    payload = resp.json()

    assert payload["default_profile"] in ("local", "remote")
    assert payload["local_model_id"].startswith("ollama_chat/")
    assert payload["remote_model_id"]
    assert set(payload["school_levels"].keys()) == {
        "fundamental_2",
        "ensino_medio",
        "geral",
    }
    assert len(payload["presets"]) >= 4
    first_preset = payload["presets"][0]
    assert first_preset["prompt"]
    assert first_preset["school_level"] in ("fundamental_2", "ensino_medio", "geral")


def test_existing_adk_and_reasoning_engine_routes_preserved() -> None:
    registered_paths = {getattr(route, "path", "") for route in fastapi_app.routes}
    assert "/api/reasoning_engine" in registered_paths
    assert "/api/stream_reasoning_engine" in registered_paths
    assert "/run" in registered_paths
    assert "/list-apps" in registered_paths
    assert "/ui" in registered_paths
    assert "/api/chat" in registered_paths
    assert "/api/compare" in registered_paths


def test_api_chat_local_and_remote_contracts(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    async def fake_invoke(
        settings: Settings,
        *,
        prompt: str,
        profile: ModelProfile,
        school_level: web_api.SchoolLevel,
        enable_tools: bool = True,
    ) -> web_api.ChatResponse:
        calls.append(
            {
                "prompt": prompt,
                "profile": profile,
                "school_level": school_level,
                "enable_tools": enable_tools,
            }
        )
        return _fake_chat_response(
            profile=profile,
            school_level=school_level,
            enable_tools=enable_tools,
            answer=(
                f"Resposta em pt-BR ({profile}, {school_level}): "
                "O território brasileiro era habitado por milhões de indígenas antes de 1500 (EF07HI10)."
            ),
            latency_ms=420.0 if profile == "remote" else 950.0,
        )

    monkeypatch.setattr(web_api, "invoke_tutor_workflow", fake_invoke)
    client = TestClient(fastapi_app)

    local_resp = client.post(
        "/api/chat",
        json={
            "prompt": "O Brasil estava vazio antes de 1500?",
            "profile": "local",
            "school_level": "fundamental_2",
            "enable_tools": True,
        },
    )
    assert local_resp.status_code == 200
    local_data = local_resp.json()
    assert local_data["status"] == "ok"
    assert local_data["profile"] == "local"
    assert local_data["school_level"] == "fundamental_2"
    assert local_data["enable_tools"] is True
    assert local_data["telemetry"]["model_id"].startswith("ollama_chat/")
    assert local_data["telemetry"]["offline_capable"] is True
    assert local_data["telemetry"]["estimated_cost_usd"] == 0.0
    assert local_data["telemetry"]["latency_ms"] == 950.0
    assert local_data["telemetry"]["input_tokens"] == 120
    assert local_data["telemetry"]["output_tokens"] == 180
    assert "EF07HI10" in local_data["bncc_codes"]
    assert len(local_data["citations"]) == 1

    remote_resp = client.post(
        "/api/chat",
        json={
            "prompt": "Explique a Lei Áurea de 1888.",
            "profile": "remote",
            "school_level": "ensino_medio",
            "enable_tools": False,
        },
    )
    assert remote_resp.status_code == 200
    remote_data = remote_resp.json()
    assert remote_data["profile"] == "remote"
    assert remote_data["enable_tools"] is False
    assert remote_data["telemetry"]["offline_capable"] is False
    assert remote_data["telemetry"]["estimated_cost_usd"] > 0.0
    assert len(calls) == 2
    assert calls[0]["profile"] == "local"
    assert calls[1]["profile"] == "remote"
    assert calls[1]["enable_tools"] is False


def test_api_chat_validation_and_error_handling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = TestClient(fastapi_app)

    blank_resp = client.post("/api/chat", json={"prompt": "   "})
    assert blank_resp.status_code == 422

    bad_level_resp = client.post(
        "/api/chat",
        json={"prompt": "Quem foi Tiradentes?", "school_level": "pos_doutorado"},
    )
    assert bad_level_resp.status_code == 422

    async def failing_invoke(*args: Any, **kwargs: Any) -> web_api.ChatResponse:
        raise ConnectionError("Ollama server unreachable on localhost:11434")

    monkeypatch.setattr(web_api, "invoke_tutor_workflow", failing_invoke)
    err_resp = client.post(
        "/api/chat",
        json={"prompt": "Quem foi Zumbi?", "profile": "local"},
    )
    assert err_resp.status_code == 502
    assert "ConnectionError" in err_resp.json()["detail"]


def test_api_compare_local_vs_remote_and_rag_ablation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[ModelProfile, bool, str]] = []

    async def fake_invoke(
        settings: Settings,
        *,
        prompt: str,
        profile: ModelProfile,
        school_level: web_api.SchoolLevel,
        enable_tools: bool = True,
    ) -> web_api.ChatResponse:
        calls.append((profile, enable_tools, prompt))
        return _fake_chat_response(
            profile=profile,
            school_level=school_level,
            enable_tools=enable_tools,
            answer=f"Resposta ({profile}, tools={enable_tools})",
            latency_ms=1100.0 if profile == "local" else 400.0,
        )

    monkeypatch.setattr(web_api, "invoke_tutor_workflow", fake_invoke)
    client = TestClient(fastapi_app)

    resp = client.post(
        "/api/compare",
        json={
            "prompt": "A Inconfidência Mineira queria abolir a escravidão?",
            "school_level": "ensino_medio",
            "mode": "local_vs_remote",
            "enable_tools": True,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["mode"] == "local_vs_remote"
    assert data["left"]["response"]["profile"] == "local"
    assert data["left"]["response"]["telemetry"]["offline_capable"] is True
    assert data["right"]["response"]["profile"] == "remote"
    assert data["right"]["response"]["telemetry"]["offline_capable"] is False
    assert data["latency_delta_ms"] == 700.0
    assert data["cost_delta_usd"] < 0.0

    calls.clear()
    ablation_resp = client.post(
        "/api/compare",
        json={
            "prompt": "O que foi a Lei Áurea?",
            "school_level": "ensino_medio",
            "mode": "local_rag_ablation",
        },
    )
    assert ablation_resp.status_code == 200
    ablation = ablation_resp.json()
    assert ablation["left"]["response"]["profile"] == "local"
    assert ablation["left"]["response"]["enable_tools"] is True
    assert ablation["right"]["response"]["profile"] == "local"
    assert ablation["right"]["response"]["enable_tools"] is False
    assert calls == [
        ("local", True, "O que foi a Lei Áurea?"),
        ("local", False, "O que foi a Lei Áurea?"),
    ]


def test_api_compare_preserves_working_side_when_other_profile_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def partially_failing_invoke(
        settings: Settings,
        *,
        prompt: str,
        profile: ModelProfile,
        school_level: web_api.SchoolLevel,
        enable_tools: bool = True,
    ) -> web_api.ChatResponse:
        if profile == "remote":
            raise RuntimeError("Missing GOOGLE_API_KEY while offline")
        return _fake_chat_response(
            profile="local",
            school_level=school_level,
            enable_tools=enable_tools,
            answer="Resposta local 100% offline sobre o Quilombo dos Palmares.",
            latency_ms=800.0,
        )

    monkeypatch.setattr(web_api, "invoke_tutor_workflow", partially_failing_invoke)
    client = TestClient(fastapi_app)

    resp = client.post(
        "/api/compare",
        json={
            "prompt": "Explique o Quilombo dos Palmares.",
            "school_level": "fundamental_2",
            "mode": "local_vs_remote",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["left"]["response"]["status"] == "ok"
    assert "Palmares" in data["left"]["response"]["answer"]
    assert data["right"]["response"]["status"] == "error"
    assert "GOOGLE_API_KEY" in (data["right"]["response"]["error"] or "")


@pytest.mark.asyncio
async def test_invoke_tutor_workflow_extracts_terminal_text_telemetry_and_citations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_agent_args: dict[str, Any] = {}

    def fake_create_agent(
        settings: Settings | None = None,
        profile: ModelProfile | None = None,
        *,
        enable_tools: bool = True,
    ) -> Any:
        captured_agent_args["profile"] = profile
        captured_agent_args["enable_tools"] = enable_tools
        return SimpleNamespace(name="brasao_ai")

    class FakeSessionService:
        async def create_session(
            self,
            *,
            app_name: str,
            user_id: str,
            session_id: str,
        ) -> Any:
            return SimpleNamespace(id=session_id)

    class FakeInMemoryRunner:
        def __init__(self, *, agent: Any, app_name: str) -> None:
            self.agent = agent
            self.app_name = app_name
            self.session_service = FakeSessionService()

        async def run_async(
            self,
            *,
            user_id: str,
            session_id: str,
            new_message: Any,
        ):
            assert new_message.parts[0].text.startswith(
                "[Nível escolar solicitado: Ensino Fundamental II"
            )
            yield _make_fake_event(
                author=TOPIC_ROUTER_NODE_NAME,
                text="intent: explanation\ntarget_level: fundamental_2",
                prompt_tokens=40,
                output_tokens=12,
            )
            yield _make_fake_event(
                author=HISTORICAL_RESEARCHER_NODE_NAME,
                function_response={
                    "status": "ok",
                    "results": [
                        {
                            "id": "quilombo_palmares",
                            "title": "Quilombo dos Palmares",
                            "bncc_codes": ["EF07HI15", "EF07HI16"],
                            "citations": [
                                {
                                    "source_id": "corpus:quilombo_palmares",
                                    "document": "Corpus Curado BNCC — Quilombo dos Palmares",
                                    "bncc_codes": ["EF07HI15", "EF07HI16"],
                                }
                            ],
                        }
                    ],
                },
                prompt_tokens=60,
                output_tokens=20,
            )
            yield _make_fake_event(
                author=PEDAGOGICAL_ADAPTER_NODE_NAME,
                text="### Rascunho Didático\nRascunho interno não verificado.",
                prompt_tokens=90,
                output_tokens=30,
            )
            yield _make_fake_event(
                author=FACT_VERIFIER_NODE_NAME,
                text=(
                    "### Rascunho Didático\n"
                    "O Quilombo dos Palmares foi o maior símbolo de resistência negra "
                    "no Brasil colonial, liderado por Ganga Zumba e Zumbi (EF07HI15)."
                ),
                prompt_tokens=110,
                output_tokens=45,
            )

    monkeypatch.setattr(web_api, "create_history_tutor_agent", fake_create_agent)
    monkeypatch.setattr("google.adk.runners.InMemoryRunner", FakeInMemoryRunner)

    result = await web_api.invoke_tutor_workflow(
        Settings(),
        prompt="Explique o Quilombo dos Palmares.",
        profile="local",
        school_level="fundamental_2",
        enable_tools=True,
    )

    assert captured_agent_args == {"profile": "local", "enable_tools": True}
    assert result.status == "ok"
    assert "Rascunho Didático" not in result.answer
    assert "Quilombo dos Palmares" in result.answer
    assert result.telemetry.input_tokens == 300
    assert result.telemetry.output_tokens == 107
    assert result.telemetry.offline_capable is True
    assert result.telemetry.estimated_cost_usd == 0.0
    assert result.bncc_codes == ["EF07HI15", "EF07HI16"]
    assert len(result.citations) == 1
    assert result.citations[0].source_id == "corpus:quilombo_palmares"


def test_makefile_exposes_web_ui_target_and_phony() -> None:
    content = MAKEFILE_PATH.read_text(encoding="utf-8")
    phony_match = re.search(r"^\.PHONY:\s*(.+)$", content, flags=re.MULTILINE)
    assert phony_match is not None
    assert "web-ui" in phony_match.group(1).split()

    target_match = re.search(
        r"^web-ui:\s*\n\t(.+)$",
        content,
        flags=re.MULTILINE,
    )
    assert target_match is not None
    recipe = target_match.group(1).strip()
    assert "uv run uvicorn app.fast_api_app:app --reload --port 8000" == recipe
