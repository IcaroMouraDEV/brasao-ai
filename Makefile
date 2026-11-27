.PHONY: help install lint format test test-all check web-ui ollama-up ollama-up-gpu ollama-pull ollama-pull-embed rag-index rag-index-lexical playground-remote playground-local eval-remote eval-local benchmark-dry-run benchmark

# Default tool-capable local Ollama model.
# Note: `gemma3:4b` lacks native Ollama tool calling and returns HTTP 400 on ADK tool nodes;
# verified tool-calling alternatives include `qwen2.5:3b` (default), `qwen2.5:1.5b`, `qwen3:8b`, and `llama3.2:3b`.
OLLAMA_MODEL ?= qwen2.5:3b
OLLAMA_EMBED_MODEL ?= nomic-embed-text
AGENTS_CLI ?= uvx google-agents-cli

help:
	@echo "Brasao AI — Local-First Development & Evaluation Targets"
	@echo ""
	@echo "Setup & Offline Quality Checks:"
	@echo "  make install             Install runtime and lint/test dependencies via uv"
	@echo "  make check               Run ruff lint/format checks + offline unit tests"
	@echo "  make lint                Run ruff check and ruff format --check"
	@echo "  make format              Auto-format and autofix with ruff"
	@echo "  make test                Run deterministic offline unit tests (tests/unit)"
	@echo "  make test-all            Run unit and live integration tests"
	@echo ""
	@echo "Local Ollama & Offline Hybrid RAG:"
	@echo "  make ollama-up           Start local Ollama in CPU-only mode (commodity hardware default)"
	@echo "  make ollama-up-gpu       Start local Ollama with NVIDIA GPU override (docker-compose.gpu.yml)"
	@echo "  make ollama-pull         Pull OLLAMA_MODEL (default: $(OLLAMA_MODEL))"
	@echo "  make ollama-pull-embed   Pull OLLAMA_EMBED_MODEL (default: $(OLLAMA_EMBED_MODEL))"
	@echo "  make rag-index           Build offline SQLite hybrid RAG index (BM25 + dense embeddings)"
	@echo "  make rag-index-lexical   Build offline BM25-only RAG index (no Ollama embeddings required)"
	@echo ""
	@echo "Interactive Interfaces:"
	@echo "  make web-ui              Run local zero-CDN Web UI and API at http://localhost:8000/ui"
	@echo "  make playground-remote   Start google-agents-cli playground with remote Gemini"
	@echo "  make playground-local    Start google-agents-cli playground with local Ollama"
	@echo ""
	@echo "Comparative Benchmarks & ADK Evaluations:"
	@echo "  make benchmark-dry-run   Offline smoke test of Local vs. Remote benchmark harness"
	@echo "  make benchmark           Run full Local (Ollama) vs. Remote (Gemini) benchmark"
	@echo "  make eval-remote         Run google-agents-cli eval with remote Gemini"
	@echo "  make eval-local          Run google-agents-cli eval with local Ollama"

install:
	uv sync --extra lint

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .

test:
	uv run pytest tests/unit

test-all:
	uv run pytest tests/unit tests/integration

check: lint test

web-ui:
	uv run uvicorn app.fast_api_app:app --reload --port 8000

ollama-up:
	docker compose up -d ollama

ollama-up-gpu:
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d ollama

ollama-pull:
	docker compose exec ollama ollama pull $(OLLAMA_MODEL)

ollama-pull-embed:
	docker compose exec ollama ollama pull $(OLLAMA_EMBED_MODEL)

rag-index:
	uv run python scripts/build_rag_index.py --embed-model $(OLLAMA_EMBED_MODEL)

rag-index-lexical:
	uv run python scripts/build_rag_index.py --no-embeddings

playground-remote:
	MODEL_PROFILE=remote $(AGENTS_CLI) playground

playground-local:
	MODEL_PROFILE=local $(AGENTS_CLI) playground

eval-remote:
	MODEL_PROFILE=remote $(AGENTS_CLI) eval run --dataset tests/eval/datasets/brazil_history_dataset.json

eval-local:
	MODEL_PROFILE=local $(AGENTS_CLI) eval run --dataset tests/eval/datasets/brazil_history_dataset.json

benchmark-dry-run:
	uv run python -m evals.runner --dry-run

benchmark:
	uv run python -m evals.runner --profiles remote local
