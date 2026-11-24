"""Google ADK 2.0 root workflow and App entrypoint for the Brazilian History AI Tutor."""

from __future__ import annotations

from google.adk import Workflow
from google.adk.apps import App

from app.config import ModelProfile, Settings
from app.workflow import (
    FACT_VERIFIER_NODE_NAME,
    HISTORICAL_RESEARCHER_NODE_NAME,
    PEDAGOGICAL_ADAPTER_NODE_NAME,
    TERMINAL_NODE_NAME,
    TOPIC_ROUTER_NODE_NAME,
    WORKFLOW_NODE_ORDER,
    create_history_tutor_workflow,
    iter_workflow_agents,
)

# Keep in sync with create_params.root_agent_name in agents-cli-manifest.yaml.
ROOT_AGENT_NAME = "brasao_ai"


def create_history_tutor_agent(
    settings: Settings | None = None,
    profile: ModelProfile | None = None,
    *,
    enable_tools: bool = True,
) -> Workflow:
    """Instantiate the ADK 2.0 `Workflow` with identical nodes, prompts, and tools across profiles."""
    return create_history_tutor_workflow(
        settings=settings,
        profile=profile,
        name=ROOT_AGENT_NAME,
        enable_tools=enable_tools,
    )


root_agent = create_history_tutor_agent()

app = App(
    root_agent=root_agent,
    name="app",
)

__all__ = [
    "FACT_VERIFIER_NODE_NAME",
    "HISTORICAL_RESEARCHER_NODE_NAME",
    "PEDAGOGICAL_ADAPTER_NODE_NAME",
    "ROOT_AGENT_NAME",
    "TERMINAL_NODE_NAME",
    "TOPIC_ROUTER_NODE_NAME",
    "WORKFLOW_NODE_ORDER",
    "app",
    "create_history_tutor_agent",
    "create_history_tutor_workflow",
    "iter_workflow_agents",
    "root_agent",
]
