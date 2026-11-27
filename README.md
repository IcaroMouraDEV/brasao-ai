# Brasão AI — Brazilian History Tutor (UN SDG 4 / ADK 2.0 / `google-agents-cli`)

**Brasão AI** (named in tribute to Thiago Braga and *Brasão de Armas*) is an educational AI agent specialized in the **History of Brazil**, aligned with **UN Sustainable Development Goal 4 (Quality Education / ODS 4)** and **BNCC**, scaffolded with Google's official **`google-agents-cli`** (`adk_python`, `agent_runtime`, `github_actions`, Terraform), and designed to compare:

- **Remote cloud inference:** Google Gemini via ADK 2.0 (`MODEL_PROFILE=remote`)
- **Local offline inference:** Open-weights models via **Ollama + LiteLLM** (`MODEL_PROFILE=local`, `ollama_chat/<model>`)

## Requirements

- **uv** (`uv sync`)
- **google-agents-cli** (`uv tool install google-agents-cli`, or invoke via `uvx google-agents-cli`)
- **Docker** (optional, to run local Ollama via `docker-compose.yml`)
- **Google Cloud SDK (`gcloud`) & Terraform** (for cloud deployment using `deployment/terraform/`)

## Quick Start

### 1. Configure Environment & Install Dependencies

```bash
cp .env.example .env
make install
```

Configure `.env` with either Vertex AI (`GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION=global`) or `GEMINI_API_KEY` for the remote Gemini profile.

### 2. Verify the Harness Offline

```bash
make check
make benchmark-dry-run
```

### 3. Start Local Ollama (Optional — via Docker)

```bash
make ollama-up        # CPU-only portable default (commodity school hardware)
make ollama-up-gpu    # Optional NVIDIA GPU override (requires nvidia-container-toolkit)
make ollama-pull OLLAMA_MODEL=gemma3:4b
```

### 4. Interactive Playground (`google-agents-cli`)

```bash
make playground-remote   # Gemini
make playground-local    # Local Ollama
```

### 5. Run Comparative Evaluations (Local vs. Remote)

```bash
# Side-by-side accuracy, SDG 4 pedagogy, latency, tokens/s, cost, and offline report:
make benchmark

# Or use Google's native agents-cli eval + LLM-as-a-judge + trace diff:
make eval-remote
make eval-local
agents-cli eval compare <remote_results.json> <local_results.json>
```

### 6. Cloud Infrastructure & Deployment (Terraform + Agent Runtime)

```bash
gcloud config set project <your-gcp-project-id>
agents-cli infra single-project   # or `agents-cli infra cicd` for GitHub Actions + WIF
agents-cli deploy
```

## Documentation

- [AGENTS.md](AGENTS.md) — Unified `agents-cli` & ODS 4 engineering harness
- [docs/architecture.md](docs/architecture.md) — ADK 2.0 architecture, Terraform IaC, and model parity
- [docs/sdg4-pedagogy.md](docs/sdg4-pedagogy.md) — UN SDG 4 and BNCC pedagogical framework
- [docs/evaluation-plan.md](docs/evaluation-plan.md) — Local vs. Gemini evaluation methodology
- [docs/experiments/compact-rag-gpu-optimization.md](docs/experiments/compact-rag-gpu-optimization.md) — Compact-RAG & GPU optimization Before vs. After benchmark report (`#20`)
