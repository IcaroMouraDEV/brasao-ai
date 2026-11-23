"""Offline unit tests for configuration, model factory parity, and manifest sync."""

from __future__ import annotations

from pathlib import Path

import pytest
from google.adk.models import Gemini
from google.adk.models.lite_llm import LiteLlm

from app.agent import ROOT_AGENT_NAME, create_history_tutor_agent
from app.config import Settings, load_settings
from app.models import build_model, format_ollama_litellm_model


def test_format_ollama_litellm_model_enforces_chat_prefix() -> None:
    assert format_ollama_litellm_model("gemma3:4b") == "ollama_chat/gemma3:4b"
    assert (
        format_ollama_litellm_model("ollama/llama3.1:8b") == "ollama_chat/llama3.1:8b"
    )
    assert format_ollama_litellm_model("ollama_chat/qwen3:8b") == "ollama_chat/qwen3:8b"


def test_remote_profile_returns_native_gemini_model() -> None:
    settings = Settings(model_profile="remote", gemini_model="gemini-3.8-flash")
    model = build_model(settings, profile="remote")
    assert isinstance(model, Gemini)
    assert model.model == "gemini-3.8-flash"
    assert settings.resolved_model_label("remote") == "gemini-3.8-flash"
    assert settings.is_offline_capable("remote") is False


def test_local_profile_metadata_marks_offline_capable() -> None:
    settings = Settings(model_profile="local", ollama_model="gemma3:4b")
    assert settings.resolved_model_label("local") == "ollama_chat/gemma3:4b"
    assert settings.is_offline_capable("local") is True


def test_load_settings_rejects_invalid_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_PROFILE", "invalid_provider")
    with pytest.raises(ValueError, match="Unsupported MODEL_PROFILE"):
        load_settings()


def test_adk_agent_factory_maintains_parity_between_local_and_remote() -> None:
    settings = Settings(gemini_model="gemini-3.8-flash", ollama_model="gemma3:4b")
    remote_agent = create_history_tutor_agent(settings=settings, profile="remote")
    local_agent = create_history_tutor_agent(settings=settings, profile="local")

    assert remote_agent.name == local_agent.name == ROOT_AGENT_NAME
    assert remote_agent.instruction == local_agent.instruction
    assert len(remote_agent.tools) == len(local_agent.tools) == 3
    assert isinstance(remote_agent.model, Gemini)
    assert remote_agent.model.model == "gemini-3.8-flash"
    assert isinstance(local_agent.model, LiteLlm)
    assert local_agent.model.model == "ollama_chat/gemma3:4b"


def test_root_agent_name_matches_agents_cli_manifest() -> None:
    manifest_text = Path("agents-cli-manifest.yaml").read_text(encoding="utf-8")
    assert f'root_agent_name: "{ROOT_AGENT_NAME}"' in manifest_text
