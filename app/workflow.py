"""Google ADK 2.0 multi-node Workflow definition for Brasão AI."""

from __future__ import annotations

from google.adk import Agent, Workflow
from google.adk.workflow import START

from app.config import ModelProfile, Settings, load_settings
from app.models import build_model
from app.prompts import (
    FACT_VERIFIER_INSTRUCTION,
    HISTORICAL_RESEARCHER_INSTRUCTION,
    HISTORICAL_RESEARCHER_NO_TOOLS_INSTRUCTION,
    PEDAGOGICAL_ADAPTER_INSTRUCTION,
    PEDAGOGICAL_ADAPTER_NO_TOOLS_INSTRUCTION,
    TOPIC_ROUTER_INSTRUCTION,
)
from app.tools import (
    build_quiz_blueprint,
    lookup_timeline,
    search_historical_corpus,
)

TOPIC_ROUTER_NODE_NAME = "TopicRouter"
HISTORICAL_RESEARCHER_NODE_NAME = "HistoricalResearcher"
PEDAGOGICAL_ADAPTER_NODE_NAME = "PedagogicalAdapter"
FACT_VERIFIER_NODE_NAME = "FactVerifier"

WORKFLOW_NODE_ORDER: tuple[str, ...] = (
    TOPIC_ROUTER_NODE_NAME,
    HISTORICAL_RESEARCHER_NODE_NAME,
    PEDAGOGICAL_ADAPTER_NODE_NAME,
    FACT_VERIFIER_NODE_NAME,
)
TERMINAL_NODE_NAME = FACT_VERIFIER_NODE_NAME

ROUTING_STATE_KEY = "routing_plan"
EVIDENCE_STATE_KEY = "historical_evidence"
DRAFT_STATE_KEY = "pedagogical_draft"
VERIFIED_STATE_KEY = "verified_response"


def create_history_tutor_workflow(
    settings: Settings | None = None,
    profile: ModelProfile | None = None,
    *,
    name: str = "brasao_ai",
    enable_tools: bool = True,
) -> Workflow:
    """Build the ADK 2.0 multi-node `Workflow` with strict local/remote model parity.

    Graph topology:
        START -> TopicRouter -> HistoricalResearcher -> PedagogicalAdapter -> FactVerifier

    Every LLM node calls `app.models.build_model(resolved_settings, profile=active_profile)`
    so swapping `MODEL_PROFILE` (`local` vs. `remote`) swaps the entire workflow graph
    without changing prompts, tools, or node wiring. When `enable_tools=False`, curated
    corpus/RAG, timeline, and quiz tools are omitted identically across profiles for
    controlled pure-memory ablation experiments.
    """
    resolved_settings = settings or load_settings(profile_override=profile)
    active_profile = profile or resolved_settings.model_profile

    topic_router = Agent(
        name=TOPIC_ROUTER_NODE_NAME,
        model=build_model(resolved_settings, profile=active_profile),
        description=(
            "Classifies student intent (explanation, timeline, quiz, out_of_scope) "
            "and target school level (fundamental_2, ensino_medio, geral)."
        ),
        instruction=TOPIC_ROUTER_INSTRUCTION,
        output_key=ROUTING_STATE_KEY,
        disallow_transfer_to_parent=True,
        disallow_transfer_to_peers=True,
    )

    historical_researcher = Agent(
        name=HISTORICAL_RESEARCHER_NODE_NAME,
        model=build_model(resolved_settings, profile=active_profile),
        description=(
            "Queries search_historical_corpus and lookup_timeline to gather grounded "
            "Brazilian History facts, perspectives, and BNCC codes."
            if enable_tools
            else "Gathers Brazilian History facts and perspectives from parametric memory "
            "(no external tools/RAG ablation mode)."
        ),
        instruction=(
            HISTORICAL_RESEARCHER_INSTRUCTION
            if enable_tools
            else HISTORICAL_RESEARCHER_NO_TOOLS_INSTRUCTION
        ),
        tools=(
            [
                search_historical_corpus,
                lookup_timeline,
            ]
            if enable_tools
            else []
        ),
        output_key=EVIDENCE_STATE_KEY,
        disallow_transfer_to_parent=True,
        disallow_transfer_to_peers=True,
    )

    pedagogical_adapter = Agent(
        name=PEDAGOGICAL_ADAPTER_NODE_NAME,
        model=build_model(resolved_settings, profile=active_profile),
        description=(
            "Drafts a pt-BR explanation, timeline synthesis, or formative quiz aligned "
            "with UN SDG 4, BNCC, and the student's target school level."
        ),
        instruction=(
            PEDAGOGICAL_ADAPTER_INSTRUCTION
            if enable_tools
            else PEDAGOGICAL_ADAPTER_NO_TOOLS_INSTRUCTION
        ),
        tools=[build_quiz_blueprint] if enable_tools else [],
        output_key=DRAFT_STATE_KEY,
        disallow_transfer_to_parent=True,
        disallow_transfer_to_peers=True,
    )

    fact_verifier = Agent(
        name=FACT_VERIFIER_NODE_NAME,
        model=build_model(resolved_settings, profile=active_profile),
        description=(
            "Audits the pedagogical draft against retrieved historical evidence to "
            "prevent fabricated dates, myths, or anachronisms before replying."
        ),
        instruction=FACT_VERIFIER_INSTRUCTION,
        output_key=VERIFIED_STATE_KEY,
        disallow_transfer_to_parent=True,
        disallow_transfer_to_peers=True,
    )

    workflow_description = (
        "UN SDG 4 multi-node educational workflow specialized in the History of Brazil, "
        "orchestrating TopicRouter -> HistoricalResearcher -> PedagogicalAdapter -> "
        "FactVerifier over curated BNCC-aligned corpus, timeline, and quiz tools."
        if enable_tools
        else "UN SDG 4 multi-node educational workflow specialized in the History of Brazil, "
        "orchestrating TopicRouter -> HistoricalResearcher -> PedagogicalAdapter -> "
        "FactVerifier in pure-memory ablation mode (tools and hybrid RAG disabled)."
    )

    return Workflow(
        name=name,
        description=workflow_description,
        edges=[
            (
                START,
                topic_router,
                historical_researcher,
                pedagogical_adapter,
                fact_verifier,
            )
        ],
    )


def iter_workflow_agents(workflow: Workflow) -> list[Agent]:
    """Return the workflow's `Agent` nodes in topological traversal order from `START`."""
    if workflow.graph is None:
        return []

    outgoing: dict[str, list[Agent]] = {}
    for edge in workflow.graph.edges:
        if isinstance(edge.to_node, Agent):
            outgoing.setdefault(edge.from_node.name, []).append(edge.to_node)

    ordered: list[Agent] = []
    seen: set[str] = set()
    queue: list[str] = [START.name]

    while queue:
        current = queue.pop(0)
        for next_agent in outgoing.get(current, []):
            if next_agent.name not in seen:
                seen.add(next_agent.name)
                ordered.append(next_agent)
                queue.append(next_agent.name)

    return ordered
