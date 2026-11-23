# SDG 4 Pedagogical & Historical Integrity Rules

Follow these rules whenever editing `app/prompts.py`, `app/data/brazil_history_corpus.json`, `app/tools/`, `tests/eval/`, or `evals/`.

## Historical Accuracy & Anti-Hallucination

- Every historical entry added to `app/data/brazil_history_corpus.json` MUST include:
  - `id`, `title`, `period`, `start_year`, `end_year`
  - `bncc_codes` (valid BNCC History codes such as `EF07HI10`, `EF08HI11`, `EF09HI02`, `EM13CHS101`)
  - `summary`, `key_facts`, `primary_actors`, `critical_perspectives`, `common_misconceptions`, and `recommended_sources`
- Never present historical myths or simplified caricatures as settled facts. Explicitly contrast multiple perspectives (Indigenous, Afro-Brazilian, regional, women, and institutional viewpoints).
- When a student asks about a topic outside the curated corpus or a fictional/anachronistic event, the agent MUST acknowledge uncertainty or correct the anachronism rather than inventing facts.

## UN SDG 4 (Quality Education) Alignment

- Target **SDG 4.1** (equitable primary/secondary learning outcomes), **SDG 4.7** (citizenship, human rights, cultural diversity), and **SDG 4.a** (accessible educational tools, including offline viability).
- Support grade-level adaptation across three Brazilian school bands:
  - `fundamental_2` (6th–9th grade: accessible vocabulary, concrete examples, clear cause-and-effect)
  - `ensino_medio` (High School / ENEM: historiographical analysis, primary-source reasoning, structural connections)
  - `geral` (General public: clear, engaging, nuanced overview)
- Encourage active learning by inviting reflection or formative exercises.

## Benchmark Fairness

- Benchmark datasets (`evals/datasets/brazil_history_benchmark.json` and `tests/eval/datasets/brazil_history_dataset.json`) MUST stay frozen during a comparison run so `local` and `remote` results are strictly comparable.
