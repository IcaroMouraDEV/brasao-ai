"""Model factory abstracting native Gemini (remote) and Ollama via LiteLLM (local)."""

from __future__ import annotations

import os
from typing import Any

from google.adk.models import Gemini
from google.genai import types

from app.config import ModelProfile, Settings


def format_ollama_litellm_model(ollama_model: str) -> str:
    """Ensure the Ollama model uses the `ollama_chat/` prefix required for ADK tool calling."""
    cleaned = ollama_model.strip()
    if cleaned.startswith("ollama_chat/"):
        return cleaned
    if cleaned.startswith("ollama/"):
        cleaned = cleaned.removeprefix("ollama/")
    return f"ollama_chat/{cleaned}"


def build_model(settings: Settings, profile: ModelProfile | None = None) -> Any:
    """Return an ADK 2.0 compatible model instance for the requested profile.

    - `remote`: returns ADK's native `Gemini` model with retry options.
    - `local`: configures `OLLAMA_API_BASE` and returns `LiteLlm(model="ollama_chat/<tag>")`.
    """
    active = profile or settings.model_profile
    if active == "remote":
        return Gemini(
            model=settings.gemini_model,
            retry_options=types.HttpRetryOptions(attempts=3),
        )

    os.environ["OLLAMA_API_BASE"] = settings.ollama_api_base
    from google.adk.models.lite_llm import LiteLlm

    litellm_kwargs: dict[str, Any] = {
        "model": format_ollama_litellm_model(settings.ollama_model),
        "temperature": settings.temperature,
        "max_tokens": 1024,
        "num_ctx": settings.ollama_num_ctx,
        # Bypass LiteLLM's optional MCP proxy gateway check when tools are passed,
        # preventing ModuleNotFoundError('mcp') on tool-calling Ollama nodes.
        "_skip_mcp_handler": True,
    }
    if settings.ollama_think is not None:
        litellm_kwargs["think"] = settings.ollama_think

    return LiteLlm(**litellm_kwargs)
