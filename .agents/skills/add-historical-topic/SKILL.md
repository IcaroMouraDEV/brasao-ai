---
name: add-historical-topic
description: >-
  Add a new BNCC-aligned Brazilian History topic to the curated corpus and create matching
  evaluation cases for the local vs. Gemini benchmark. Use when expanding historical coverage or
  fixing factual gaps in the tutor.
---

# Add a Curated Brazilian History Topic

Follow this runbook whenever adding a new period, event, or historical theme to the agent's knowledge base.

## Steps

1. **Draft the Corpus Entry** in `app/data/brazil_history_corpus.json`:
   - Choose a unique snake_case `id` (e.g., `revolucao_pernambucana_1817`).
   - Provide accurate `start_year`, `end_year`, `period`, and valid `bncc_codes`.
   - Write `title`, `summary`, `key_facts`, `primary_actors`, `critical_perspectives`, `common_misconceptions`, and `recommended_sources` in Brazilian Portuguese (`pt-BR`).
   - Include at least one minority, Indigenous, Afro-Brazilian, or regional perspective aligned with SDG 4.7.
2. **Add Matching Evaluation Cases**:
   - Add a detailed benchmark case (`expected_facts` + `forbidden_hallucinations`) to `evals/datasets/brazil_history_benchmark.json`.
   - Optionally add a matching `eval_case` to `tests/eval/datasets/brazil_history_dataset.json` for `agents-cli eval run`.
3. **Validate Offline Integrity**:
   ```bash
   make check
   ```
   Confirm corpus schema validation, timeline lookup, accent-insensitive retrieval, and dataset schema checks all pass.
