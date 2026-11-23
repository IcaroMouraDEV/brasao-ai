## Summary

<!-- Describe what changed and why. Link the issue this PR closes (e.g., Closes #2). -->

Closes #

## Type of Change

- [ ] Feature (`feat`)
- [ ] Bug fix (`fix`)
- [ ] CI / Harness / Tooling (`ci` / `chore`)
- [ ] Brazilian History Corpus / SDG 4 Pedagogy (`content` / `feat`)
- [ ] Benchmark / Evaluation (`eval` / `experiment`)
- [ ] Documentation (`docs`)

## Model Parity (`local` vs. `remote`) Checklist

- [ ] Strict model parity preserved: both `local` (Ollama via `ollama_chat/<model>`) and `remote` (Gemini) use the exact same `app.agent.create_history_tutor_agent`, prompts, tools, and retrieval logic without profile-specific prompt hacks
- [ ] `ROOT_AGENT_NAME` (`brasao_ai`) in `app/agent.py` remains in sync with `agents-cli-manifest.yaml`
- [ ] `DEFAULT_GEMINI_MODEL` (`gemini-3.8-flash`) was not silently changed

## Offline Unit Tests & Quality Gates

- [ ] `uv run ruff check .` passes
- [ ] `uv run ruff format --check .` passes
- [ ] `uv run ty check` passes
- [ ] `uv run pytest tests/unit` passes completely offline (zero Gemini or Ollama API calls)
- [ ] `uv run python -m evals.runner --dry-run` completes cleanly
- [ ] No secrets, `.env` files, or local generated reports (`security-reports/`, `evals/reports/*.json`) are staged

## Benchmark & SDG 4 Pedagogical Impact

<!-- State "None (CI/tooling only)" if this PR does not affect prompts, tools, corpus, or models. -->

- **Expected impact on `local` vs. `remote` benchmark** (historical accuracy, hallucination resistance, SDG 4 pedagogy, latency, tokens/s, offline viability):
- [ ] Frozen benchmark datasets (`evals/datasets/brazil_history_benchmark.json`, `tests/eval/datasets/brazil_history_dataset.json`) were not altered mid-comparison
- [ ] Live comparative benchmark (`make benchmark`) run or follow-up planned when prompts, corpus, tools, or models changed
