"""Offline unit tests for the ADK 2.0 multi-node Brasão AI Workflow."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from types import SimpleNamespace
from typing import Any

import pytest
from google.adk import Agent, Workflow
from google.adk.a2a.utils.agent_card_builder import AgentCardBuilder
from google.adk.apps import App
from google.adk.models import Gemini
from google.adk.models.base_llm import BaseLlm
from google.adk.models.lite_llm import LiteLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.adk.workflow import START
from google.genai import types
from pydantic import Field

import app.workflow as workflow_module
from app.agent import (
    FACT_VERIFIER_NODE_NAME,
    HISTORICAL_RESEARCHER_NODE_NAME,
    PEDAGOGICAL_ADAPTER_NODE_NAME,
    ROOT_AGENT_NAME,
    TERMINAL_NODE_NAME,
    TOPIC_ROUTER_NODE_NAME,
    WORKFLOW_NODE_ORDER,
    create_history_tutor_agent,
    iter_workflow_agents,
    root_agent,
)
from app.agent import app as adk_app
from app.config import Settings
from app.workflow import (
    DRAFT_STATE_KEY,
    EVIDENCE_STATE_KEY,
    ROUTING_STATE_KEY,
    VERIFIED_STATE_KEY,
    create_history_tutor_workflow,
)
from evals.runner import _extract_workflow_output


def _tool_names(agent: Agent) -> list[str]:
    names: list[str] = []
    for tool in agent.tools:
        if hasattr(tool, "__name__"):
            names.append(tool.__name__)
        elif hasattr(tool, "name"):
            names.append(str(tool.name))
        else:
            names.append(type(tool).__name__)
    return names


class _DeterministicWorkflowMockLlm(BaseLlm):
    """Offline mock LLM that simulates the 4-node Brazilian History workflow."""

    model: str = "mock-workflow-llm"
    captured_instructions: list[str] = Field(default_factory=list)

    async def generate_content_async(
        self,
        llm_request: LlmRequest,
        stream: bool = False,
    ) -> AsyncGenerator[LlmResponse, None]:
        instruction = ""
        if llm_request.config and llm_request.config.system_instruction:
            instruction = str(llm_request.config.system_instruction)
        self.captured_instructions.append(instruction)

        if "Você é o nó `TopicRouter`" in instruction:
            text = (
                "intent: explanation\n"
                "target_level: ensino_medio\n"
                "historical_topic: Lei Áurea e Abolição da Escravidão\n"
                "time_range: 1888\n"
                "student_goal: Compreender a Abolição de 1888 sob múltiplas perspectivas."
            )
            prompt_tokens, out_tokens = 40, 25
        elif "Você é o nó `HistoricalResearcher`" in instruction:
            text = (
                "status: ok\n"
                "Título: Abolição da Escravidão e Lei Áurea (1888)\n"
                "BNCC: EF08HI19, EM13CHS502\n"
                "Fatos: Assinada em 13 de maio de 1888; resultado de longa resistência negra, "
                "quilombos e mobilização abolicionista; ausência de políticas de reparação."
            )
            prompt_tokens, out_tokens = 80, 45
        elif "Você é o nó `PedagogicalAdapter`" in instruction:
            text = (
                "Rascunho provisório: Em 13 de maio de 1888 a Lei Áurea encerrou formalmente a "
                "escravidão no Brasil após séculos de resistência negra e movimento abolicionista "
                "(BNCC EF08HI19, EM13CHS502). Como a ausência de políticas de integração impactou "
                "a população negra no pós-abolição?"
            )
            prompt_tokens, out_tokens = 120, 60
        else:
            assert "Você é o nó `FactVerifier`" in instruction
            text = (
                "A Abolição da Escravidão no Brasil, formalizada pela Lei Áurea em 13 de maio de "
                "1888 (BNCC EF08HI19, EM13CHS502), não foi um gesto isolado: resultou de séculos "
                "de resistência negra, quilombos e mobilização abolicionista. Como a falta de "
                "políticas públicas de reparação no pós-abolição ainda influencia desigualdades hoje?"
            )
            prompt_tokens, out_tokens = 150, 70

        yield LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part.from_text(text=text)],
            ),
            usage_metadata=types.GenerateContentResponseUsageMetadata(
                prompt_token_count=prompt_tokens,
                candidates_token_count=out_tokens,
                total_token_count=prompt_tokens + out_tokens,
            ),
        )


def test_root_agent_and_app_export_valid_adk_workflow() -> None:
    assert isinstance(root_agent, Workflow)
    assert root_agent.name == ROOT_AGENT_NAME == "brasao_ai"
    assert isinstance(adk_app, App)
    assert adk_app.name == "app"
    assert adk_app.root_agent is root_agent


def test_workflow_graph_topology_nodes_tools_and_state_keys() -> None:
    settings = Settings(model_profile="remote")
    workflow = create_history_tutor_workflow(settings=settings, profile="remote")

    assert workflow.graph is not None
    edge_pairs = [
        (edge.from_node.name, edge.to_node.name) for edge in workflow.graph.edges
    ]
    assert edge_pairs == [
        (START.name, TOPIC_ROUTER_NODE_NAME),
        (TOPIC_ROUTER_NODE_NAME, HISTORICAL_RESEARCHER_NODE_NAME),
        (HISTORICAL_RESEARCHER_NODE_NAME, PEDAGOGICAL_ADAPTER_NODE_NAME),
        (PEDAGOGICAL_ADAPTER_NODE_NAME, FACT_VERIFIER_NODE_NAME),
    ]

    nodes = iter_workflow_agents(workflow)
    assert tuple(node.name for node in nodes) == WORKFLOW_NODE_ORDER
    assert nodes[-1].name == TERMINAL_NODE_NAME == FACT_VERIFIER_NODE_NAME

    by_name = {node.name: node for node in nodes}
    assert _tool_names(by_name[TOPIC_ROUTER_NODE_NAME]) == []
    assert _tool_names(by_name[HISTORICAL_RESEARCHER_NODE_NAME]) == [
        "search_historical_corpus",
        "lookup_timeline",
    ]
    assert _tool_names(by_name[PEDAGOGICAL_ADAPTER_NODE_NAME]) == [
        "build_quiz_blueprint",
    ]
    assert _tool_names(by_name[FACT_VERIFIER_NODE_NAME]) == []

    assert by_name[TOPIC_ROUTER_NODE_NAME].output_key == ROUTING_STATE_KEY
    assert by_name[HISTORICAL_RESEARCHER_NODE_NAME].output_key == EVIDENCE_STATE_KEY
    assert by_name[PEDAGOGICAL_ADAPTER_NODE_NAME].output_key == DRAFT_STATE_KEY
    assert by_name[FACT_VERIFIER_NODE_NAME].output_key == VERIFIED_STATE_KEY

    # Downstream nodes interpolate upstream state keys using ADK optional template syntax.
    assert f"{{{ROUTING_STATE_KEY}?}}" in str(
        by_name[HISTORICAL_RESEARCHER_NODE_NAME].instruction
    )
    assert f"{{{ROUTING_STATE_KEY}?}}" in str(
        by_name[PEDAGOGICAL_ADAPTER_NODE_NAME].instruction
    )
    assert f"{{{EVIDENCE_STATE_KEY}?}}" in str(
        by_name[PEDAGOGICAL_ADAPTER_NODE_NAME].instruction
    )
    assert f"{{{EVIDENCE_STATE_KEY}?}}" in str(
        by_name[FACT_VERIFIER_NODE_NAME].instruction
    )
    assert f"{{{DRAFT_STATE_KEY}?}}" in str(
        by_name[FACT_VERIFIER_NODE_NAME].instruction
    )


def test_every_workflow_node_uses_build_model_with_strict_local_remote_parity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []
    real_build_model = workflow_module.build_model

    def spy_build_model(settings: Settings, profile: Any = None) -> Any:
        resolved = profile or settings.model_profile
        calls.append((resolved, settings.resolved_model_label(resolved)))
        return real_build_model(settings, profile=profile)

    monkeypatch.setattr(workflow_module, "build_model", spy_build_model)

    settings = Settings(gemini_model="gemini-3.8-flash", ollama_model="gemma3:4b")
    remote_workflow = create_history_tutor_agent(settings=settings, profile="remote")
    local_workflow = create_history_tutor_agent(settings=settings, profile="local")

    assert len(calls) == 8
    assert calls[:4] == [("remote", "gemini-3.8-flash")] * 4
    assert calls[4:] == [("local", "ollama_chat/gemma3:4b")] * 4

    assert remote_workflow.name == local_workflow.name == ROOT_AGENT_NAME
    assert remote_workflow.description == local_workflow.description
    assert remote_workflow.graph is not None and local_workflow.graph is not None
    assert [
        (edge.from_node.name, edge.to_node.name) for edge in remote_workflow.graph.edges
    ] == [
        (edge.from_node.name, edge.to_node.name) for edge in local_workflow.graph.edges
    ]

    remote_nodes = iter_workflow_agents(remote_workflow)
    local_nodes = iter_workflow_agents(local_workflow)
    assert len(remote_nodes) == len(local_nodes) == 4

    for remote_node, local_node in zip(remote_nodes, local_nodes, strict=True):
        assert remote_node.name == local_node.name
        assert remote_node.description == local_node.description
        assert remote_node.instruction == local_node.instruction
        assert _tool_names(remote_node) == _tool_names(local_node)
        assert remote_node.output_key == local_node.output_key
        assert (
            remote_node.disallow_transfer_to_parent
            == local_node.disallow_transfer_to_parent
            is True
        )
        assert (
            remote_node.disallow_transfer_to_peers
            == local_node.disallow_transfer_to_peers
            is True
        )
        assert isinstance(remote_node.model, Gemini)
        assert remote_node.model.model == "gemini-3.8-flash"
        assert isinstance(local_node.model, LiteLlm)
        assert local_node.model.model == "ollama_chat/gemma3:4b"


@pytest.mark.asyncio
async def test_workflow_executes_end_to_end_offline_and_propagates_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_llm = _DeterministicWorkflowMockLlm()
    monkeypatch.setattr(
        workflow_module,
        "build_model",
        lambda settings, profile=None: mock_llm,
    )

    workflow = create_history_tutor_agent(
        settings=Settings(model_profile="local"),
        profile="local",
    )
    runner = InMemoryRunner(agent=workflow, app_name="brasao_ai_unit_test")
    session = await runner.session_service.create_session(
        app_name="brasao_ai_unit_test",
        user_id="student_1",
        session_id="session_1",
    )

    events = [
        event
        async for event in runner.run_async(
            user_id="student_1",
            session_id=session.id,
            new_message=types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text="Explique a Lei Áurea de 1888 para o Ensino Médio."
                    )
                ],
            ),
        )
    ]

    emitted_authors = [
        event.author
        for event in events
        if event.content
        and event.content.parts
        and any(part.text for part in event.content.parts)
    ]
    assert emitted_authors == list(WORKFLOW_NODE_ORDER)
    assert len(mock_llm.captured_instructions) == 4
    # Verify upstream output_key values were injected into downstream instructions.
    assert "Lei Áurea e Abolição da Escravidão" in mock_llm.captured_instructions[1]
    assert "BNCC: EF08HI19, EM13CHS502" in mock_llm.captured_instructions[2]
    assert "Rascunho provisório:" in mock_llm.captured_instructions[3]

    final_text, in_tokens, out_tokens = _extract_workflow_output(
        events, terminal_author=TERMINAL_NODE_NAME
    )
    assert "Abolição da Escravidão no Brasil" in final_text
    assert "Rascunho provisório:" not in final_text
    assert "intent: explanation" not in final_text
    assert in_tokens == 40 + 80 + 120 + 150
    assert out_tokens == 25 + 45 + 60 + 70

    final_session = await runner.session_service.get_session(
        app_name="brasao_ai_unit_test",
        user_id="student_1",
        session_id=session.id,
    )
    assert final_session is not None
    assert ROUTING_STATE_KEY in final_session.state
    assert EVIDENCE_STATE_KEY in final_session.state
    assert DRAFT_STATE_KEY in final_session.state
    assert VERIFIED_STATE_KEY in final_session.state


def test_extract_workflow_output_falls_back_to_last_author_and_skips_partials() -> None:
    events = [
        SimpleNamespace(
            author="brasao_ai",
            partial=True,
            usage_metadata=SimpleNamespace(
                prompt_token_count=999, candidates_token_count=999
            ),
            content=SimpleNamespace(parts=[SimpleNamespace(text="partial chunk")]),
        ),
        SimpleNamespace(
            author="brasao_ai",
            partial=False,
            usage_metadata=SimpleNamespace(
                prompt_token_count=30, candidates_token_count=15
            ),
            content=SimpleNamespace(
                parts=[SimpleNamespace(text="Resposta final única.")]
            ),
        ),
    ]
    text, in_tokens, out_tokens = _extract_workflow_output(
        events, terminal_author=TERMINAL_NODE_NAME
    )
    assert text == "Resposta final única."
    assert in_tokens == 30
    assert out_tokens == 15


@pytest.mark.asyncio
async def test_a2a_agent_card_builder_supports_workflow_root_agent() -> None:
    card = await AgentCardBuilder(
        agent=root_agent,
        rpc_url="http://localhost:8000/a2a/app",
        agent_version="0.1.0",
    ).build()
    assert card.name == ROOT_AGENT_NAME
    assert len(card.skills) >= 1
