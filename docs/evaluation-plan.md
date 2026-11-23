# Comparative Evaluation Plan: Local Model (Ollama) vs. Remote Model (Gemini)

## Research Question

> To what extent can a locally hosted open-weights model running on Ollama match a remote Gemini model when powering a tool-assisted Google ADK 2.0 Brazilian History tutor aligned with UN SDG 4, and what are the operational trade-offs in latency, cost, and offline viability?

## Experimental Controls (Fair-Comparison Protocol)

To guarantee scientific validity:

- **Frozen System Prompt:** Identical `pt-BR` pedagogical system instruction for both profiles.
- **Frozen Tools & Corpus:** Identical `lookup_timeline`, `search_historical_corpus`, and `build_quiz_blueprint` implementations and identical `brazil_history_corpus.json` version.
- **Frozen Benchmark Dataset:** All models run the exact same prompt suite (`evals/datasets/brazil_history_benchmark.json`).
- **Controlled Sampling:** Low temperature (`0.2` recommended for deterministic educational comparison) documented in every report header.
- **Blind / Bias-Aware Judging:** When using LLM-as-a-judge, responses are scored against explicit per-case checklists (`expected_facts`, `forbidden_hallucinations`, `target_level`) rather than open-ended preference alone, reducing self-preference bias if Gemini is used as a judge.

## Evaluation Dimensions

### 1. Historical Accuracy & Hallucination Resistance (Weight: 45%)

Measured per benchmark case on a `0.0` to `5.0` scale (plus deterministic keyword/fact checks):

- **Fact Coverage (`0–5`):** Presence and correctness of `expected_facts` (dates, historical processes, actors, causes/consequences).
- **Hallucination Resistance (`0–5`):** Absence of fabricated dates, anachronisms, or `forbidden_hallucinations` (e.g., repeating historical myths when asked a misconception trap question).
- **Source & Corpus Grounding (`0–5`):** Alignment with curated corpus facts and explicit honesty when information is uncertain.

### 2. SDG 4 Pedagogical Quality (Weight: 35%)

Measured per benchmark case on a `0.0` to `5.0` scale:

- **Clarity & Structure (`0–5`):** Well-organized, accessible `pt-BR` explanation suitable for students.
- **Grade-Level Adaptation (`0–5`):** Vocabulary and analytical depth match the requested level (`fundamental_2`, `ensino_medio`, or `geral`).
- **Critical Thinking & Diversity (`0–5`):** Encourages reflection, avoids reductive stereotypes, and incorporates multiple historical perspectives (SDG 4.7 / BNCC).

### 3. Operational Performance & Offline Viability (Weight: 20% + Descriptive Table)

Captured automatically by `evals/runner.py` for every request:

- **End-to-End Latency (`latency_ms`):** Wall-clock response time (mean, p50, p95).
- **Throughput (`tokens_per_second`):** Output tokens divided by generation time (when token usage is reported by the provider).
- **Token Usage & Estimated Cost (`estimated_cost_usd`):** Calculated per 1,000 student interactions (local marginal API cost = `$0.00`, remote cost calculated from published Gemini pricing table configured in `evals/rubrics.py`).
- **Offline Viability (`offline_capable`):** Binary capability flag plus hardware footprint notes (RAM/VRAM requirement for the chosen Ollama tag).

## Benchmark Dataset Categories

The initial dataset (`evals/datasets/brazil_history_benchmark.json`) covers:

1. **Factual & Chronological Synthesis** (Colonial, Empire, Republic, Contemporary).
2. **Misconception & Hallucination Traps** (e.g., *Lei Áurea* myth, "peaceful" Independence myth, "discovery" Eurocentrism).
3. **Grade-Level Adaptation** (`fundamental_2` vs. `ensino_medio` / ENEM).
4. **Formative Assessment Generation** (quiz creation with pedagogical rationale).
5. **Out-of-Scope / Epistemic Honesty** (testing whether the agent avoids inventing non-existent historical treaties/events).

## Report Artifacts

Each benchmark run writes two files to `evals/reports/`:

- `comparison_<timestamp>.json` — Full structured telemetry, model responses, deterministic checks, and rubric scores.
- `comparison_<timestamp>.md` — Human-readable executive summary table comparing Local vs. Remote across all three dimensions, ready for academic reports or presentations.
