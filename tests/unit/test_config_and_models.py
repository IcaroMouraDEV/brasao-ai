"""Offline unit tests for configuration, model factory parity, and manifest sync."""

from __future__ import annotations

from pathlib import Path

import pytest
from google.adk import Workflow
from google.adk.models import Gemini
from google.adk.models.lite_llm import LiteLlm

from app.agent import (
    ROOT_AGENT_NAME,
    WORKFLOW_NODE_ORDER,
    create_history_tutor_agent,
    iter_workflow_agents,
)
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
    remote_workflow = create_history_tutor_agent(settings=settings, profile="remote")
    local_workflow = create_history_tutor_agent(settings=settings, profile="local")

    assert isinstance(remote_workflow, Workflow)
    assert isinstance(local_workflow, Workflow)
    assert remote_workflow.name == local_workflow.name == ROOT_AGENT_NAME

    remote_nodes = iter_workflow_agents(remote_workflow)
    local_nodes = iter_workflow_agents(local_workflow)

    assert tuple(node.name for node in remote_nodes) == WORKFLOW_NODE_ORDER
    assert tuple(node.name for node in local_nodes) == WORKFLOW_NODE_ORDER

    total_remote_tools = 0
    total_local_tools = 0
    for remote_node, local_node in zip(remote_nodes, local_nodes, strict=True):
        assert remote_node.name == local_node.name
        assert remote_node.instruction == local_node.instruction
        assert remote_node.output_key == local_node.output_key
        assert len(remote_node.tools) == len(local_node.tools)
        total_remote_tools += len(remote_node.tools)
        total_local_tools += len(local_node.tools)
        assert isinstance(remote_node.model, Gemini)
        assert remote_node.model.model == "gemini-3.8-flash"
        assert isinstance(local_node.model, LiteLlm)
        assert local_node.model.model == "ollama_chat/gemma3:4b"

    assert total_remote_tools == total_local_tools == 3


def test_root_agent_name_matches_agents_cli_manifest() -> None:
    manifest_text = Path("agents-cli-manifest.yaml").read_text(encoding="utf-8")
    assert f'root_agent_name: "{ROOT_AGENT_NAME}"' in manifest_text
