# Brasão AI — Brazilian History Tutor (UN SDG 4 / ADK 2.0 / `google-agents-cli`)

**Brasão AI** (named in tribute to historian Thiago Braga and the *Brasão de Armas* channel) is a **local-first** educational AI agent specialized in the **History of Brazil**, aligned with **UN Sustainable Development Goal 4 (Quality Education / ODS 4)** and **BNCC**, scaffolded with Google's official **`google-agents-cli`** (`adk_python`, `agent_runtime`, `github_actions`, Terraform) on **Google Agent Development Kit (ADK) 2.0**.

The project compares **strictly identical** 4-node ADK workflows, prompts, curated tools, and offline hybrid RAG across two execution profiles:

- **Remote cloud inference (`MODEL_PROFILE=remote`):** Native Google Gemini (`google.adk.models.Gemini` via Google AI Studio `GEMINI_API_KEY` or optional Vertex AI).
- **Local offline inference (`MODEL_PROFILE=local`):** Open-weights models served locally by **Ollama + LiteLLM** (`ollama_chat/<model>`) for zero-cost, offline-capable school deployment (UN SDG 4.a).

---

## Architecture & Curated Knowledge Base

- **4-Node ADK 2.0 `Workflow` (`ROOT_AGENT_NAME = "brasao_ai"`):**
  `TopicRouter -> HistoricalResearcher -> PedagogicalAdapter -> FactVerifier`
  - Identical node topology, prompts (`app/prompts.py`), tools, and sampling temperature (`MODEL_TEMPERATURE=0.2`) across `local` and `remote` profiles.
  - Profile-agnostic student output sanitization (`app/app_utils/sanitization.py`) prevents internal draft leaks, and non-quiz guards prevent spurious quiz blueprint dumps.
- **Grounded Brazilian History Corpus & Sources (`app/data/`):**
  - **23 curated BNCC-aligned topics** in `app/data/brazil_history_corpus.json` spanning Pre-Colonial Brazil to the 1988 Constitution and Contemporary Brazil.
  - **4 structured reference & primary-source collections** in `app/data/sources/`:
    - `bncc_historia.json` (BNCC History skills & codes)
    - `constituicao_1988.json` (1988 Federal Constitution excerpts)
    - `enem_referencias.json` (ENEM Humanities reference competencies)
    - `fontes_primarias.json` (curated Brazilian primary historical documents)
- **Offline Hybrid RAG (`make rag-index`):**
  - Shared SQLite index (`app/data/rag_index.sqlite3`, `158` chunks) combining **field-aware BM25 lexical search** with local **`nomic-embed-text` dense embeddings** fused via **Reciprocal Rank Fusion (RRF)**.
  - Compacted `search_historical_corpus` tool payloads (`<= 8 KiB` / `~1,961` tokens — a **-67.7%** reduction from uncompacted payloads) fit cleanly inside `OLLAMA_NUM_CTX=8192` without context truncation.
  - Three deterministic, offline-testable pedagogical tools: `search_historical_corpus`, `lookup_timeline`, and `build_quiz_blueprint`.
- **Local-First Execution:** Everything — offline unit tests, hybrid RAG indexing, local Ollama inference, Remote Gemini API calls (`GEMINI_API_KEY`), comparative benchmarks with SVG charts, and the zero-CDN local Web UI — runs **100% on a local machine** with no GCP account required.

---

## Requirements

- **uv** (`uv sync --extra lint`)
- **google-agents-cli** (`uv tool install google-agents-cli`, or invoked automatically via `uvx google-agents-cli`)
- **Docker** (optional, to run local Ollama via `docker-compose.yml` or GPU-enabled `docker-compose.gpu.yml` with `nvidia-container-toolkit`)
- **Google Cloud SDK (`gcloud`) & Terraform** *(optional — only needed if deploying the cloud IaC reference in `deployment/terraform/`)*

---

## Quick Start (Local-First Workflow)

### 1. Configure Environment & Install Dependencies

```bash
cp .env.example .env
make install
```

Edit `.env` to configure your preferred profile(s):

- **Remote Gemini (easiest local setup):** Set `GEMINI_API_KEY` from [Google AI Studio](https://aistudio.google.com/apikey) (`REMOTE_REQUEST_DELAY_S=6.0` respects Free Tier RPM limits; set `0.0` on paid tier).
- **Local Ollama:** Defaults in `.env.example` use `OLLAMA_MODEL=qwen2.5:3b`, `OLLAMA_EMBED_MODEL=nomic-embed-text`, and `OLLAMA_NUM_CTX=8192`. For thinking models such as `qwen3:8b`, set `OLLAMA_THINK=false` for lower-latency runs.

### 2. Verify the Harness Offline

```bash
make check               # ruff lint/format + offline pytest tests/unit
uv run ty check          # static type check
make benchmark-dry-run   # offline end-to-end benchmark smoke check
```

### 3. Start Local Ollama & Build the Offline Hybrid RAG Index

```bash
# Start Ollama (choose CPU-only portable default OR optional NVIDIA GPU override):
make ollama-up           # CPU-only default (commodity school hardware)
make ollama-up-gpu       # Optional NVIDIA GPU override (docker-compose.gpu.yml; requires nvidia-container-toolkit)

# Pull a verified tool-calling LLM and the local embedding model:
make ollama-pull                          # pulls default OLLAMA_MODEL=qwen2.5:3b
make ollama-pull OLLAMA_MODEL=qwen3:8b    # optional 8B model (best offline quality; pair with OLLAMA_THINK=false)
make ollama-pull-embed                    # pulls nomic-embed-text

# Build the shared offline SQLite hybrid RAG index (158 chunks):
make rag-index
# Or build a zero-embedding BM25-only index without running Ollama:
make rag-index-lexical
```

> [!WARNING]
> **Why `gemma3:4b` is unsupported:** Ollama's `gemma3:4b` manifest exposes only `["completion", "vision"]` and **lacks native tool calling**, causing `HistoricalResearcher` to fail immediately with **HTTP 400** (`does not support tools`). Always use a verified tool-calling Ollama model (`qwen2.5:3b`, `qwen2.5:1.5b`, `qwen3:8b`, or `llama3.2:3b`).

### 4. Launch the Local Offline Web UI or Playground

```bash
# Zero-CDN local Web UI (interactive chat, school-level adapter, citations, & side-by-side comparison):
make web-ui
# Open http://localhost:8000/ui (or http://localhost:8000/demo)

# Or use Google's native google-agents-cli interactive playground:
make playground-remote   # Remote Gemini
make playground-local    # Local Ollama
```

The local Web UI (`app/web_api.py` + `app/ui.py`) serves zero external CDN assets and exposes `GET /ui`, `GET /demo`, `GET /api/ui/config`, `POST /api/chat`, and `POST /api/compare` (Local vs. Remote and RAG-ablation side-by-side modes with live latency, tokens/s, cost, BNCC code, and citation telemetry).

### 5. Run Comparative Benchmarks & Evaluations

```bash
# Full Local (Ollama) vs. Remote (Gemini) benchmark over the frozen 24-case dataset:
make benchmark

# Full-featured research run with blind LLM-as-a-judge, incremental checkpoint/resume, and offline SVG charts:
uv run python -m evals.runner \
  --profiles remote local \
  --use-judge \
  --remote-delay 6.0 \
  --checkpoint evals/reports/latest_run.json \
  --resume evals/reports/latest_run.json \
  --charts

# Pure parametric memory vs. Hybrid RAG ablation (disables corpus/timeline/quiz tools identically across profiles):
uv run python -m evals.runner --profiles local --disable-tools --use-judge

# Native google-agents-cli eval + LLM-as-a-judge + trace diff (12-case ADK dataset):
make eval-remote
make eval-local
uvx google-agents-cli eval compare <remote_results.json> <local_results.json>
```

---

## Key Experimental Findings

All empirical experiments use the frozen stratified benchmark subset (`evals/datasets/brazil_history_benchmark.json`), identical 4-node ADK workflows, `MODEL_TEMPERATURE=0.2`, and blind model-redacted LLM-as-a-judge grading (`evals/judge.py`). Local runs were measured honestly on commodity CPU-only hardware (4-core Intel i5-10300H, `size_vram: 0`) to mirror public school lab constraints (UN SDG 4.a).

| Experiment / Report | Configuration | Offline | Completed | Det. Acc (/5) | Blind Judge (/5) | Mean Latency | Est. Cost / 1k Queries | Key Finding |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **[#7 Baseline](docs/experiments/baseline-local-vs-gemini.md)** | Remote `gemini-3.5-flash-lite` (Uncompacted RAG) | No | 3/8 *(503 spikes)* | 4.17 | **5.00** *(n=3)* | 61.3 s | $2.41 | Near-perfect accuracy & pedagogy, but Free Tier suffered 503 outages |
| **[#7 Baseline](docs/experiments/baseline-local-vs-gemini.md)** | Local `qwen2.5:1.5b` (Uncompacted RAG, `n_ctx=4096`) | **Yes** | **8/8 (100%)** | 2.55 | 1.61 *(n=3)* | 241.1 s | **$0.00** | 100% offline reliability inside 1.56 GiB RAM, but ~6.2k-token tool payloads caused context shifts |
| **[#8 RAG Ablation](docs/experiments/rag-ablation-local-vs-gemini.md)** | Local `qwen2.5:1.5b` (**No RAG**, `--disable-tools`) | **Yes** | **8/8 (100%)** | 1.88 | 0.86 *(n=6)* | 205.9 s | **$0.00** | Pure parametric 1.5B memory hallucinates laws, figures, and chronology |
| **[#8 RAG Ablation](docs/experiments/rag-ablation-local-vs-gemini.md)** | Local `qwen2.5:1.5b` (**Hybrid RAG**) | **Yes** | **8/8 (100%)** | **2.55** *(+35.6%)* | **1.61** *(+0.75)* | 280.9 s | **$0.00** | Offline hybrid RAG raises fact recall **+13.5 pp** (`37.5% -> 51.0%`), but uncompacted payloads overflow `n_ctx=4096` |
| **[#20 Compact RAG](docs/experiments/compact-rag-gpu-optimization.md)** | Remote `gemini-3.5-flash-lite` (**Compact RAG**) | No | **8/8 (100%)** | **4.69** | **4.99** *(n=8)* | **11.8 s** *(-80.7%)* | $2.64 | **5.2× faster** (`230.6 tok/s`), `93.8%` fact recall, `0` hallucinations — classroom-ready online |
| **[#20 Compact RAG](docs/experiments/compact-rag-gpu-optimization.md)** | Local `qwen2.5:1.5b` (Compact RAG, `n_ctx=8192`) | **Yes** | **8/8 (100%)** | **3.54** *(+38.8%)* | **2.10** *(2.72 paired)* | 294.1 s *(CPU)* | **$0.00** | **+19.8 pp** fact recall (`51.0% -> 70.8%`), `-20%` output tokens, zero quiz-blueprint leaks |
| **[#20 Compact RAG](docs/experiments/compact-rag-gpu-optimization.md)** | Local `qwen2.5:3b` (Compact RAG, `n_ctx=8192`, 4 cases) | **Yes** | **4/4 (100%)** | 1.56 | **3.23** *(+185.8%)* | 501.4 s *(CPU)* | **$0.00** | Reversed `#7`/`#8` RAG regression (`1.13 -> 3.23/5`), **0 hallucinations**, `-20.7%` p95 latency |
| **[#20 Compact RAG](docs/experiments/compact-rag-gpu-optimization.md)** | Local `qwen3:8b` (`OLLAMA_THINK=false`, `n_ctx=8192`) | **Yes** | 1/2 *(CPU timeout)* | 3.75 | **4.83** *(n=1)* | 2,269.9 s *(CPU)* | **$0.00** | Matches remote quality offline (`5.00` acc / `4.67` ped), proving **GPU offload (`make ollama-up-gpu`)** is the final unlock for interactive 8B school labs |

---

## Makefile Targets Reference

Run `make help` at any time to list all available targets:

| Target | Purpose |
| :--- | :--- |
| `make install` | Install runtime, lint, and test dependencies (`uv sync --extra lint`) |
| `make check` | Run `ruff` lint/format checks + offline `pytest tests/unit` |
| `make lint` / `make format` | Check or autofix formatting and lint rules with `ruff` |
| `make test` / `make test-all` | Run offline unit tests (`tests/unit`) or unit + live integration tests |
| `make ollama-up` | Start local Ollama container in CPU-only mode (`docker-compose.yml`) |
| `make ollama-up-gpu` | Start local Ollama container with NVIDIA GPU passthrough (`docker-compose.gpu.yml`) |
| `make ollama-pull` | Pull `OLLAMA_MODEL` into Docker Ollama (default: `qwen2.5:3b`) |
| `make ollama-pull-embed` | Pull `OLLAMA_EMBED_MODEL` into Docker Ollama (default: `nomic-embed-text`) |
| `make rag-index` | Build SQLite hybrid RAG index (BM25 + `nomic-embed-text` dense vectors) |
| `make rag-index-lexical` | Build offline BM25-only RAG index (no Ollama embeddings required) |
| `make web-ui` | Launch FastAPI + local zero-CDN Web UI at `http://localhost:8000/ui` |
| `make playground-remote` / `make playground-local` | Launch `google-agents-cli playground` for Gemini or Ollama |
| `make benchmark-dry-run` / `make benchmark` | Run offline smoke check or full Local vs. Remote comparative benchmark |
| `make eval-remote` / `make eval-local` | Run `google-agents-cli eval` against `tests/eval/datasets/brazil_history_dataset.json` |

---

## Optional Cloud IaC Reference (GCP / Terraform)

> [!NOTE]
> **100% Local-First:** You do **not** need a Google Cloud project, Terraform, or cloud deployment to develop, evaluate, benchmark, or use Brasão AI locally with either Local Ollama or Remote Gemini (`GEMINI_API_KEY`).

The repository preserves the official `google-agents-cli` Terraform IaC (`deployment/terraform/`) and GitHub Actions CI/CD workflows (`.github/workflows/`) as an optional cloud deployment reference for teams targeting Google Cloud Agent Runtime:

```bash
gcloud config set project <your-gcp-project-id>
agents-cli infra single-project   # or `agents-cli infra cicd` for GitHub Actions + WIF
agents-cli deploy
```

---

## Documentation

- [AGENTS.md](AGENTS.md) — Engineering invariants, model parity rules, and command map
- [docs/architecture.md](docs/architecture.md) — ADK 2.0 4-node workflow, hybrid RAG, and dual-model factory
- [docs/sdg4-pedagogy.md](docs/sdg4-pedagogy.md) — UN SDG 4 and BNCC pedagogical framework
- [docs/evaluation-plan.md](docs/evaluation-plan.md) — Local vs. Gemini evaluation methodology
- [docs/experiments/baseline-local-vs-gemini.md](docs/experiments/baseline-local-vs-gemini.md) — Baseline Local vs. Gemini benchmark report (`#7`)
- [docs/experiments/rag-ablation-local-vs-gemini.md](docs/experiments/rag-ablation-local-vs-gemini.md) — Hybrid RAG ablation study (`#8`)
- [docs/experiments/compact-rag-gpu-optimization.md](docs/experiments/compact-rag-gpu-optimization.md) — Compact-RAG & GPU optimization Before vs. After report (`#20`)
