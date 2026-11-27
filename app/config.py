"""Typed runtime configuration for local (Ollama) and remote (Gemini) model profiles."""

from __future__ import annotations

import os
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

ModelProfile = Literal["local", "remote"]

DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
DEFAULT_OLLAMA_MODEL = "gemma3:4b"
DEFAULT_OLLAMA_API_BASE = "http://localhost:11434"
DEFAULT_OLLAMA_EMBED_MODEL = "nomic-embed-text"
DEFAULT_OLLAMA_NUM_CTX = 8192
DEFAULT_RAG_INDEX_PATH = "app/data/rag_index.sqlite3"


def _parse_optional_bool(raw: str | None, *, env_name: str) -> bool | None:
    """Parse an optional boolean environment variable (`None` preserves model default)."""
    if raw is None:
        return None
    normalized = raw.strip().lower()
    if normalized in {"", "none", "auto", "default"}:
        return None
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"Unsupported {env_name} value '{raw}'. Expected 'true', 'false', or empty."
    )


def _parse_ollama_num_ctx(raw: str | None) -> int:
    """Parse `OLLAMA_NUM_CTX` with fallback to `DEFAULT_OLLAMA_NUM_CTX` when unset/empty."""
    if raw is None or not raw.strip():
        return DEFAULT_OLLAMA_NUM_CTX
    try:
        return int(raw.strip())
    except ValueError as exc:
        raise ValueError(
            f"Unsupported OLLAMA_NUM_CTX value '{raw}'. Expected a positive integer."
        ) from exc


class Settings(BaseModel):
    """Runtime settings shared across ADK 2.0 agents and comparative evaluations."""

    model_profile: ModelProfile = Field(
        default="remote",
        description="Active execution profile: 'remote' (Gemini) or 'local' (Ollama via LiteLLM).",
    )
    gemini_model: str = Field(
        default=DEFAULT_GEMINI_MODEL,
        description="Remote Google Gemini model identifier used by ADK 2.0.",
    )
    ollama_model: str = Field(
        default=DEFAULT_OLLAMA_MODEL,
        description="Local Ollama model tag (wrapped internally with 'ollama_chat/').",
    )
    ollama_api_base: str = Field(
        default=DEFAULT_OLLAMA_API_BASE,
        description="Base URL of the local or containerized Ollama server.",
    )
    ollama_embed_model: str = Field(
        default=DEFAULT_OLLAMA_EMBED_MODEL,
        description="Local Ollama embedding model shared by hybrid RAG across both profiles.",
    )
    ollama_num_ctx: int = Field(
        default=DEFAULT_OLLAMA_NUM_CTX,
        ge=1024,
        description=(
            "Context window size (`num_ctx`) passed to Ollama via LiteLLM for the local profile."
        ),
    )
    ollama_think: bool | None = Field(
        default=None,
        description=(
            "Optional Ollama reasoning toggle (`think`). `None` preserves the model default."
        ),
    )
    rag_index_path: str = Field(
        default=DEFAULT_RAG_INDEX_PATH,
        description="Path to the generated offline SQLite hybrid RAG index.",
    )
    temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
        description="Sampling temperature kept identical across local and remote profiles.",
    )
    eval_judge_model: str = Field(
        default=DEFAULT_GEMINI_MODEL,
        description="Model identifier used by LLM-as-a-judge evaluation metrics.",
    )

    def resolved_model_label(self, profile: ModelProfile | None = None) -> str:
        """Return a human-readable identifier for the target model profile."""
        active = profile or self.model_profile
        if active == "local":
            return f"ollama_chat/{self.ollama_model}"
        return self.gemini_model

    def is_offline_capable(self, profile: ModelProfile | None = None) -> bool:
        """Return True when the target profile can run without external internet access."""
        active = profile or self.model_profile
        return active == "local"


def load_settings(profile_override: ModelProfile | None = None) -> Settings:
    """Load settings from environment variables (and `.env` when present)."""
    load_dotenv(override=False)

    raw_profile = (
        (profile_override or os.getenv("MODEL_PROFILE", "remote")).strip().lower()
    )
    if raw_profile not in ("local", "remote"):
        raise ValueError(
            f"Unsupported MODEL_PROFILE '{raw_profile}'. Expected 'local' or 'remote'."
        )

    return Settings(
        model_profile=raw_profile,  # type: ignore[arg-type]
        gemini_model=os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip(),
        ollama_model=os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL).strip(),
        ollama_api_base=os.getenv("OLLAMA_API_BASE", DEFAULT_OLLAMA_API_BASE).strip(),
        ollama_embed_model=os.getenv(
            "OLLAMA_EMBED_MODEL", DEFAULT_OLLAMA_EMBED_MODEL
        ).strip(),
        ollama_num_ctx=_parse_ollama_num_ctx(os.getenv("OLLAMA_NUM_CTX")),
        ollama_think=_parse_optional_bool(
            os.getenv("OLLAMA_THINK"), env_name="OLLAMA_THINK"
        ),
        rag_index_path=os.getenv("RAG_INDEX_PATH", DEFAULT_RAG_INDEX_PATH).strip(),
        temperature=float(os.getenv("MODEL_TEMPERATURE", "0.2")),
        eval_judge_model=os.getenv("EVAL_JUDGE_MODEL", DEFAULT_GEMINI_MODEL).strip(),
    )
