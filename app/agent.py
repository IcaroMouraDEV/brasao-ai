"""Google ADK 2.0 root agent and App entrypoint for the Brazilian History AI Tutor."""

from __future__ import annotations

from google.adk.agents import Agent
from google.adk.apps import App

from app.config import ModelProfile, Settings, load_settings
from app.models import build_model
from app.prompts import HISTORY_TUTOR_INSTRUCTION
from app.tools import (
    build_quiz_blueprint,
    lookup_timeline,
    search_historical_corpus,
)

# Keep in sync with create_params.root_agent_name in agents-cli-manifest.yaml.
ROOT_AGENT_NAME = "brasao_ai"


def create_history_tutor_agent(
    settings: Settings | None = None,
    profile: ModelProfile | None = None,
) -> Agent:
    """Instantiate the ADK 2.0 `Agent` with identical prompts and tools across profiles."""
    resolved_settings = settings or load_settings(profile_override=profile)
    active_profile = profile or resolved_settings.model_profile
    model = build_model(resolved_settings, profile=active_profile)

    return Agent(
        name=ROOT_AGENT_NAME,
        model=model,
        description=(
            "UN SDG 4 educational tutor specialized in the History of Brazil, "
            "equipped with curated BNCC-aligned corpus, timeline, and quiz tools."
        ),
        instruction=HISTORY_TUTOR_INSTRUCTION,
        tools=[
            search_historical_corpus,
            lookup_timeline,
            build_quiz_blueprint,
        ],
    )


root_agent = create_history_tutor_agent()

app = App(
    root_agent=root_agent,
    name="app",
)
