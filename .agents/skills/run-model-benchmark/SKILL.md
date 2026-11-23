---
name: run-model-benchmark
description: >-
  Run and analyze the comparative benchmark between a local Ollama model and a remote Gemini model
  for the Brazilian History ADK 2.0 agent. Use when the user asks to benchmark, evaluate, or
  compare local vs. remote model results.
---

# Run Local vs. Remote Model Benchmark

Follow this runbook to execute a fair, reproducible comparison between the local Ollama model and the remote Gemini model using both the operational benchmark runner (`evals/runner.py`) and Google's native `agents-cli eval`.

## Prerequisites

1. Verify `.env` (copied from `.env.example`):
   - Remote Gemini auth configured via Vertex AI (`GOOGLE_CLOUD_PROJECT` + `GOOGLE_CLOUD_LOCATION=global`) or `GEMINI_API_KEY`.
   - `OLLAMA_API_BASE` points to the running Ollama server (default `http://localhost:11434`).
   - `OLLAMA_MODEL` is pulled locally (e.g., `gemma3:4b`, `qwen3:8b`, or `llama3.1:8b`).
2. If using Docker for Ollama:
   ```bash
   make ollama-up
   make ollama-pull OLLAMA_MODEL=gemma3:4b
   ```

## Execution Steps

1. Confirm offline checks pass first:
   ```bash
   make check
   make benchmark-dry-run
   ```
2. Run the side-by-side comparative benchmark (Accuracy, SDG 4 Pedagogy, Latency, Tokens/s, Cost, Offline Viability):
   ```bash
   make benchmark
   ```
3. (Optional) Run `agents-cli eval` with LLM-as-a-judge (`tests/eval/response_quality.py`) for both profiles and compare traces natively:
   ```bash
   MODEL_PROFILE=remote agents-cli eval run --dataset tests/eval/datasets/brazil_history_dataset.json
   MODEL_PROFILE=local agents-cli eval run --dataset tests/eval/datasets/brazil_history_dataset.json
   agents-cli eval compare <remote_results.json> <local_results.json>
   ```
4. Inspect generated reports in `evals/reports/`:
   - `comparison_<timestamp>.json` — Raw outputs, telemetry, and scores.
   - `comparison_<timestamp>.md` — Side-by-side summary table.

## Validation

- Confirm both `remote` and `local` executed the exact same case IDs and `PROMPT_VERSION`.
- Verify `offline_capable` is `true` for `local` and `false` for `remote`, with no silent fallback.
