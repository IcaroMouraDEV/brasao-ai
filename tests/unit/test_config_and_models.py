"""Offline unit tests for configuration, model factory parity, and manifest sync."""

from __future__ import annotations

from pathlib import Path

import pytest
from google.adk import Workflow
from google.adk.models import Gemini
from google.adk.models.lite_llm import LiteLlm
from litellm.llms.ollama.chat.transformation import OllamaChatConfig
from litellm.utils import get_optional_params
from pydantic import ValidationError

from app.agent import (
    ROOT_AGENT_NAME,
    WORKFLOW_NODE_ORDER,
    create_history_tutor_agent,
    iter_workflow_agents,
)
from app.config import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_OLLAMA_NUM_CTX,
    Settings,
    load_settings,
)
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


def test_settings_defaults_keep_gemini_model_and_ollama_num_ctx() -> None:
    settings = Settings()
    assert DEFAULT_GEMINI_MODEL == "gemini-3.8-flash"
    assert settings.gemini_model == "gemini-3.8-flash"
    assert DEFAULT_OLLAMA_NUM_CTX == 8192
    assert settings.ollama_num_ctx == 8192
    assert settings.ollama_think is None


def test_load_settings_parses_ollama_num_ctx_and_think(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MODEL_PROFILE", "local")
    monkeypatch.setenv("OLLAMA_NUM_CTX", "16384")
    monkeypatch.setenv("OLLAMA_THINK", "false")
    settings = load_settings()
    assert settings.ollama_num_ctx == 16384
    assert settings.ollama_think is False

    monkeypatch.setenv("OLLAMA_THINK", "true")
    assert load_settings().ollama_think is True

    monkeypatch.setenv("OLLAMA_THINK", "")
    monkeypatch.setenv("OLLAMA_NUM_CTX", "")
    defaulted = load_settings()
    assert defaulted.ollama_num_ctx == 8192
    assert defaulted.ollama_think is None


def test_load_settings_rejects_invalid_ollama_num_ctx_and_think(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MODEL_PROFILE", "local")
    monkeypatch.setenv("OLLAMA_NUM_CTX", "not_a_number")
    with pytest.raises(ValueError, match="OLLAMA_NUM_CTX"):
        load_settings()

    monkeypatch.setenv("OLLAMA_NUM_CTX", "512")
    with pytest.raises(ValidationError):
        load_settings()

    monkeypatch.setenv("OLLAMA_NUM_CTX", "8192")
    monkeypatch.setenv("OLLAMA_THINK", "maybe")
    with pytest.raises(ValueError, match="OLLAMA_THINK"):
        load_settings()


def test_build_model_wires_ollama_num_ctx_and_optional_think() -> None:
    default_local = build_model(
        Settings(model_profile="local", ollama_model="qwen3:8b"),
        profile="local",
    )
    assert isinstance(default_local, LiteLlm)
    assert default_local.model == "ollama_chat/qwen3:8b"
    assert default_local._additional_args["num_ctx"] == 8192
    assert "think" not in default_local._additional_args

    tuned_local = build_model(
        Settings(
            model_profile="local",
            ollama_model="qwen3:8b",
            ollama_num_ctx=16384,
            ollama_think=False,
        ),
        profile="local",
    )
    assert isinstance(tuned_local, LiteLlm)
    assert tuned_local._additional_args["num_ctx"] == 16384
    assert tuned_local._additional_args["think"] is False

    # Verify LiteLLM's OllamaChatConfig forwards num_ctx into options and think at top-level.
    optional_params = get_optional_params(
        model="qwen3:8b",
        custom_llm_provider="ollama_chat",
        temperature=tuned_local._additional_args["temperature"],
        max_tokens=tuned_local._additional_args["max_tokens"],
        num_ctx=tuned_local._additional_args["num_ctx"],
        think=tuned_local._additional_args["think"],
    )
    ollama_request = OllamaChatConfig().transform_request(
        model="qwen3:8b",
        messages=[{"role": "user", "content": "Explique a Lei Áurea."}],
        optional_params=dict(optional_params),
        litellm_params={},
        headers={},
    )
    assert ollama_request["options"]["num_ctx"] == 16384
    assert ollama_request["think"] is False

    # Remote profile remains native Gemini regardless of Ollama-specific tuning flags.
    remote = build_model(
        Settings(
            model_profile="remote",
            ollama_num_ctx=16384,
            ollama_think=False,
        ),
        profile="remote",
    )
    assert isinstance(remote, Gemini)
    assert remote.model == "gemini-3.8-flash"
