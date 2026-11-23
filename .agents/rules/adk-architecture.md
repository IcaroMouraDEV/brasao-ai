# ADK 2.0, `google-agents-cli`, and Dual-Model Rules

Follow these rules whenever modifying `app/`, `deployment/`, `tests/`, or `evals/`.

## Official `google-agents-cli` & ADK 2.0 Standards

- Preserve the official `google-agents-cli` layout (`agents-cli-manifest.yaml`, `app/agent.py`, `app/fast_api_app.py`, `app/app_utils/`, `deployment/terraform/`, `.github/workflows/`).
- `app/agent.py` MUST export both `root_agent` (with `name == "brasao_ai"`, matching `agents-cli-manifest.yaml`) and `app = App(root_agent=root_agent, name="app")`.
- Use the official `google.adk.*` namespace (`Agent`, `App`, `Gemini`, `LiteLlm`, `Workflow`).
- Do not modify `deployment/terraform/` or `.github/workflows/` unless explicitly tasked with infrastructure/CI changes. Never apply Terraform or deploy to GCP without human approval.

## Local vs. Remote Model Factory

- All model instantiation MUST go through `app.models.build_model(settings, profile)`.
- **Remote profile (`remote`):** Return `Gemini(model=settings.gemini_model, retry_options=types.HttpRetryOptions(attempts=3))`.
- **Local profile (`local`):** Set `OLLAMA_API_BASE` and return `LiteLlm(model=f"ollama_chat/{settings.ollama_model}")` from `google.adk.models.lite_llm`.
- Never use bare `ollama/<model>` in LiteLLM when tools are attached; `ollama_chat/<model>` is required for stable tool calling.
- Never apply profile-specific prompt hacks that favor one model over the other. Differences in accuracy, tool use, or pedagogy are experimental results and must remain observable.

## Tool Design Contracts

- Every tool function in `app/tools/` MUST have:
  - Explicit Python type annotations on all parameters and return values.
  - A concise English docstring describing purpose, arguments, and return structure.
  - Pure JSON-serializable `dict[str, Any]` payloads with an explicit `status` field (`"ok"` | `"not_found"` | `"error"`).
- Tools MUST stay deterministic and offline-testable against `app/data/brazil_history_corpus.json`.
