# AGENTS.md — Brasão AI (Brazilian History Tutor / UN SDG 4 / ADK 2.0 / `google-agents-cli`)

**Brasão AI** (named in tribute to historian Thiago Braga and the *Brasão de Armas* channel) is an educational AI agent specialized in the **History of Brazil**, aligned with **UN Sustainable Development Goal 4 (SDG 4 — Quality Education / ODS 4)**, scaffolded with Google's official **`google-agents-cli`** (`adk_python`, `agent_runtime`, `github_actions`, Terraform) on **Google Agent Development Kit (ADK) 2.0**, with a first-class comparative benchmark between a **local model (Ollama via LiteLLM)** and a **remote model (Gemini)**.

## Prerequisites

Install Google's Agents CLI (one-time):
```bash
uv tool install google-agents-cli
```

## Project Mission & Scope

- **Social Goal:** Support inclusive, equitable, quality education (UN SDG 4) in Brazil by offering an accurate, BNCC-aligned History of Brazil tutor that can also run **offline** on local hardware for low-connectivity schools.
- **Core Experiment:** Compare identical agent workflows, prompts, and curated tools across:
  - **Remote Profile (`remote`):** Native Google Gemini (`app.models.build_model` -> `google.adk.models.Gemini`).
  - **Local Profile (`local`):** Local open-weights model served by Ollama (native or Docker) via `google.adk.models.lite_llm.LiteLlm` (`ollama_chat/<model>`).
- **Evaluation Pillars:**
  1. Historical accuracy and hallucination resistance (ground-truth evalset + LLM-as-a-judge).
  2. SDG 4 pedagogical quality (clarity, grade-level adaptation, critical thinking, BNCC alignment).
  3. Operational performance (latency, tokens/second, estimated cost, offline viability).

## Language Policy

- **Engineering Artifacts (English-First):** All Python code, identifiers, types, file/directory names, docstrings, comments, logs, tests, commit messages, and technical docs MUST be written in **English**.
- **Educational & Domain Content (`pt-BR`):** Student-facing instructions (`app/prompts.py`), curated Brazilian History corpus (`app/data/brazil_history_corpus.json`), and evaluation prompts/references (`evals/datasets/`, `tests/eval/datasets/`) MUST be written in **Brazilian Portuguese (`pt-BR`)**, keeping authentic domain terms (`BNCC`, `ENEM`, `Quilombo`, `Inconfidência Mineira`, `Estado Novo`, `Lei Áurea`, `Diretas Já`, etc.).

## Architecture & Operational Invariants

1. **Strict Model Parity:** Never fork prompts, tools, or retrieval logic between `local` and `remote` profiles. Both MUST instantiate `app.agent.create_history_tutor_agent` with only `MODEL_PROFILE` (`local` | `remote`) swapped.
2. **Keep `root_agent_name` in Sync:** `ROOT_AGENT_NAME` in `app/agent.py` MUST always match `create_params.root_agent_name` (`"brasao_ai"`) in `agents-cli-manifest.yaml` so telemetry and `agents-cli` agree.
3. **Never Silently Change Default Models:** Keep `DEFAULT_GEMINI_MODEL` (`gemini-3.8-flash`) unless explicitly asked by the user. For GCP 404 model errors, fix `GOOGLE_CLOUD_LOCATION` (e.g., `global`) rather than altering the model name.
4. **Ollama Tool Calling Prefix:** Always use `ollama_chat/<model>` in LiteLLM — never bare `ollama/`.
5. **Grounded Historical Facts:** Pedagogical tools (`search_historical_corpus`, `lookup_timeline`, `build_quiz_blueprint`) MUST read from `app/data/brazil_history_corpus.json`.
6. **Human Approval for Cloud Deployments & Terraform:** Never run `agents-cli deploy` or `terraform apply` without explicit user approval. On Terraform Error 409 conflicts, use `terraform import` rather than retrying creation.
7. **Deterministic Offline Unit Tests:** Everything under `tests/unit/` MUST run offline with zero Gemini or Ollama API calls. Live tests live under `tests/integration/`, `tests/eval/`, and `evals/`.
8. **No Secrets in VCS:** Never commit `.env` or API keys. Copy `.env.example` locally.

## Development Lifecycle & Commands

| Command | Purpose |
| :--- | :--- |
| `agents-cli install` / `uv sync --extra lint` | Install project and lint dependencies |
| `make check` | Run `ruff` lint/format check + offline `pytest tests/unit` |
| `MODEL_PROFILE=remote agents-cli playground` | Interactive local playground (remote Gemini) |
| `MODEL_PROFILE=local agents-cli playground` | Interactive local playground (local Ollama) |
| `agents-cli eval run` | Run `tests/eval/` inference and LLM-as-a-judge grading |
| `agents-cli eval compare` | Diff two `agents-cli eval` result JSON files |
| `make benchmark-dry-run` | Offline smoke check of Local vs. Remote comparison harness |
| `make benchmark` | Run full Local (Ollama) vs. Remote (Gemini) benchmark report |
| `uv run pytest tests/unit tests/integration` | Run unit and live integration tests |
| `agents-cli infra single-project` / `cicd` | Provision GCP infrastructure via Terraform (`deployment/terraform/`) |
| `agents-cli deploy` | Deploy to Google Cloud Agent Runtime (requires human approval) |

## Repository Map

- `app/` — Official `google-agents-cli` ADK 2.0 agent package (`agent.py`, `config.py`, `models.py`, `prompts.py`, `data/`, `tools/`, `fast_api_app.py`, `app_utils/`).
- `deployment/terraform/` — Official Terraform IaC (`single-project/`, `cicd/`, `shared/`).
- `.github/workflows/` — GitHub Actions CI/CD pipelines (`pr_checks.yaml`, `staging.yaml`, `deploy-to-prod.yaml`).
- `tests/` — `unit/` (offline), `integration/`, `eval/` (`agents-cli eval`), and `load_test/`.
- `evals/` — Local vs. Remote side-by-side benchmark harness (accuracy, SDG 4 pedagogy, latency, tokens/s, cost, offline viability).
- `docs/` — Architecture ([docs/architecture.md](docs/architecture.md)), SDG 4 pedagogy ([docs/sdg4-pedagogy.md](docs/sdg4-pedagogy.md)), and evaluation plan ([docs/evaluation-plan.md](docs/evaluation-plan.md)).
- `.agents/` — Project-specific rules and procedural skills.
