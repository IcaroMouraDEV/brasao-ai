.PHONY: install lint format test check ollama-up ollama-pull ollama-pull-embed rag-index rag-index-lexical playground-remote playground-local eval-remote eval-local benchmark-dry-run benchmark

OLLAMA_MODEL ?= gemma3:4b
OLLAMA_EMBED_MODEL ?= nomic-embed-text
AGENTS_CLI ?= uvx google-agents-cli

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

ollama-up:
	docker compose up -d ollama

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
