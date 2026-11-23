# Baseline Experiment: Local (Ollama) vs. Remote (Gemini) ADK 2.0 Tutor

> **Issue:** `#7` (`perf(benchmark): execute baseline Local (Ollama) vs. Remote (Gemini) experiment`)  
> **Date (UTC):** `2026-11-23` (`13:33Z – 17:53Z`)  
> **System Under Test:** Brasão AI — ADK 2.0 4-node `Workflow` (`TopicRouter -> HistoricalResearcher -> PedagogicalAdapter -> FactVerifier`), `ROOT_AGENT_NAME = "brasao_ai"`, `PROMPT_VERSION = "2.0.0"`, Hybrid RAG index (`158` chunks, BM25 + `nomic-embed-text` 768-dim RRF)  
> **Sampling Control:** `MODEL_TEMPERATURE = 0.2` across both `local` and `remote` profiles (zero profile-specific prompt forks)

---

## 1. Executive Summary

This baseline experiment evaluates Brasão AI's identical 4-node Google ADK 2.0 workflow and hybrid RAG toolset across locally hosted open-weights models (Docker Ollama, CPU-only commodity hardware) and remote Google Gemini models (Google AI Studio Free Tier), using both deterministic/heuristic rubrics (`evals/runner.py`), blind checklist-grounded LLM-as-a-judge scoring (`evals/judge.py`), and native `google-agents-cli eval` (`v1.15.0`) trace tooling.

**Headline findings:**

1. **Historical Accuracy & Pedagogy Gap Is Substantial at 1.5B–3B Scale:** On every completed remote query (`n = 5` across `gemini-3.5-flash-lite`, `gemini-3-flash-preview`, and `gemini-4-hydrogen`), Remote Gemini achieved **5.00 / 5.00** on blind LLM-as-a-judge grading (both historical accuracy and UN SDG 4 pedagogy) with **0 hallucinations**. By contrast, CPU-viable local tool-calling models scored **1.61 / 5.00** (`qwen2.5:1.5b`, `n = 3` judge-graded) and **1.13 / 5.00** (`qwen2.5:3b`, `n = 2` judge-graded), revealing semantic hallucinations (e.g., relocating Quilombo dos Palmares to Goiás, fabricating BNCC codes `4.1.1`/`4.1.2`, confusing Era Vargas with the 1964 Military Dictatorship, and endorsing the fictional *Tratado de Guanabara de 1742*) that simple deterministic keyword rubrics missed.
2. **Offline Availability vs. Cloud Free-Tier Volatility Validates UN SDG 4.a:** During the 4-hour experimental window, Google AI Studio Free Tier experienced widespread `HTTP 503 UNAVAILABLE` high-demand spikes across `gemini-3.8-flash`, `gemini-4-hydrogen`, `gemini-3.1-flash-lite`, and intermittently `gemini-3.5-flash-lite` (5/8 paired cases errored), plus strict `20 RPD` (`gemini-3-flash-preview`) and `15 RPM` (`gemini-3.5-flash-lite`) limits. Local `qwen2.5:1.5b` achieved **8/8 (100%) zero-error offline completion** at **$0.00 marginal API cost** inside **1.56 GiB total RAM** (1.17 GiB LLM + 389 MiB `nomic-embed-text`).
3. **4-Node Workflow + Large Hybrid RAG Payloads Bottleneck CPU-Only Inference:** A single `search_historical_corpus` call returns ~24.9 KB (~6,219 tokens) of combined corpus metadata + ranked chunks, pushing 4-node sessions to **7.7k–12.3k input tokens** and **2.2k–3.0k output tokens** per student query. On a CPU-only 4-core laptop (no Docker GPU passthrough), `qwen2.5:1.5b` averaged **241.1 s** (`8.75 tok/s` end-to-end), `qwen2.5:3b` averaged **553.4 s** (`4.15 tok/s`), `llama3.2:3b` took **707.5 s** (`3.08 tok/s`), and `qwen3:8b` hit LiteLLM's 600 s socket timeout when default `<think>` reasoning was active. Remote `gemini-3.5-flash-lite` completed in **61.3 s mean** (`38.7 tok/s`, est. **$2.41 / 1k queries** paid tier) and `gemini-3-flash-preview` in **33.3 s** (`92.2 tok/s`).
4. **Native Tool-Calling Compatibility Filters Candidate Local Models:** `gemma3:4b` (Ollama capabilities `["completion", "vision"]`) failed immediately on `HistoricalResearcher` with HTTP 400 (`registry.ollama.ai/library/gemma3:4b does not support tools`), confirming that tool-capable manifests (`qwen2.5`, `qwen3`, `llama3.2`, `llama3.1`) are mandatory for ADK 2.0 tool workflows.

---

## 2. Experimental Setup & Hardware Provenance

### 2.1 Host & Container Environment

| Component | Observed Specification |
| :--- | :--- |
| **OS / Kernel** | Linux `x86_64` (WSL2 virtualized environment) |
| **CPU** | Intel Core i5-10300H @ 2.50 GHz (4 physical cores / 8 threads, AVX2, 8 MiB L3) |
| **System RAM** | 11 GiB allocated (~9.2 GiB available before Ollama model load) |
| **Host GPU** | NVIDIA GeForce GTX 1650 (4,096 MiB VRAM, Driver `615.78.02`, CUDA `13.4`) |
| **Docker Ollama Runtime** | `ollama/ollama:latest` (`v0.40.17`, container `brasao-ai-ollama`), `runc` **CPU-only** (`DeviceRequests: []`, `size_vram: 0`) |
| **Python / ADK Toolchain** | Python `3.12.14`, `google-adk >= 2.12.0`, `litellm 1.106.9`, `google-agents-cli 1.15.0` |
| **Hybrid RAG Index** | `app/data/rag_index.sqlite3` — `158` chunks (`chunk_size=700`, `chunk_overlap=100`), `nomic-embed-text` (`768`-dim F16 embeddings) + BM25 RRF (`built_at: 2026-11-23T13:33:51Z`) |

> **SDG 4.a Hardware Note:** Running `brasao-ai-ollama` in CPU-only mode (`docker-compose.yml` default) intentionally mirrors commodity school lab machines (4-core x86 CPU, 8–12 GB RAM, no discrete GPU passthrough) rather than workstation-grade GPUs. Enabling NVIDIA GPU passthrough (`nvidia-container-toolkit`) on a 4–8 GiB GPU would accelerate 1.5B–4B local inference by an estimated 5–10× and allow `qwen3:8b` partial layer offloading.

### 2.2 Candidate Local Models Screened (Ollama `v0.40.17`)

| Model Tag | Params | Quant | Disk Size | Active RAM (`size`) | Active VRAM | Digest (prefix) | Ollama Capabilities | ADK Workflow Viability (CPU-Only) |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :--- |
| `nomic-embed-text:latest` | 137M | F16 | 274 MB | 389 MB | 0 MB | `0a109f422b47` | `embedding` | Working (`37–794 ms` per `/api/embed`) |
| `gemma3:4b` | 3.9B | Q4_K_M | 3.35 GB | N/A | 0 MB | `68ceab7e468a` | `completion`, `vision` | **Blocked:** HTTP 400 (`does not support tools`) |
| `qwen3:8b` | 8.2B | Q4_K_M | 5.23 GB | 5.95 GB | 0 MB | `500a1f067a9f` | `completion`, `tools`, `thinking` | **Timeout (600 s)** in 4-node workflow with default `think=True` (`3.54 tok/s`; 70.3 s single prompt vs. 9.6 s with `think=False`) |
| `llama3.2:3b` | 3.2B | Q4_K_M | 2.02 GB | 2.57 GB | 0 MB | `a80c4f17acd5` | `completion`, `tools` | **Viable (slow):** 1/1 completed (`707.5 s`, `3.08 tok/s` e2e, `8.0–9.2 tok/s` decode) |
| `qwen2.5:3b` | 3.1B | Q4_K_M | 1.93 GB | 2.17 GB | 0 MB | `357c53fb659c` | `completion`, `tools` | **Viable (slow):** 4/4 completed post-fix (`553.4 s` mean, `4.15 tok/s` e2e) |
| `qwen2.5:1.5b` | 1.5B | Q4_K_M | 986 MB | 1.17 GB | 0 MB | `65ec06548149` | `completion`, `tools` | **Primary CPU-only local baseline:** 8/8 completed (`241.1 s` mean, `8.75 tok/s` e2e, **1.56 GB total RAM** with embedder) |

### 2.3 Remote Gemini Models Screened (Google AI Studio Free Tier)

| Model Identifier | Live Probe Status (`2026-11-23`) | Notes |
| :--- | :--- | :--- |
| `gemini-3.8-flash` (`DEFAULT_GEMINI_MODEL`) | `503 UNAVAILABLE` (all probes) | Kept unchanged as `DEFAULT_GEMINI_MODEL` in `app/config.py`; unavailable due to upstream high-demand spikes during run window |
| `gemini-4-hydrogen` (`.env` live config) | `1/1` succeeded, then `503 UNAVAILABLE` | Completed `colonial_indigenas_eurocentrismo` (`5.00/5` det. & judge, `217.2 s`) before sustained 503 spikes |
| `gemini-3-flash-preview` | `1/1` succeeded, then `429` (`20 RPD`) | Completed `colonial_indigenas_eurocentrismo` (`3.75/5` det., `5.00/5` judge, `33.3 s`, `92.2 tok/s`), then exhausted 20 requests/day Free Tier limit |
| `gemini-3.5-flash-lite` | **Primary live remote baseline** (`3/8` ok, `5/8` `503`/timeout) | Most reliable available Free Tier model (`15 RPM`, `50.9–81.5 s`, `5.00/5` blind judge on 3/3 completed cases) |
| `gemini-3.5-flash` | `504 DEADLINE_EXCEEDED` / `503` | Timed out under demand |
| `gemini-3.1-flash-lite` / `-preview` | `503 UNAVAILABLE` | High-demand outage |
| `gemini-3.1-pro-preview`, `gemini-4-neon`, `gemini-4-argon`, `gemini-4-carbon`, `gemini-omni-1.2-flash` | `429 RESOURCE_EXHAUSTED` | Zero Free Tier quota on this key |

### 2.4 Frozen Evaluation Subsets

All evaluations drew verbatim cases from the frozen `24`-case dataset (`evals/datasets/brazil_history_benchmark.json`) and `12`-case ADK eval dataset (`tests/eval/datasets/brazil_history_dataset.json`):

- **Stratified 8-case subset** (2 per category, all 3 school levels — `fundamental_2`, `ensino_medio`, `geral`): `colonial_indigenas_eurocentrismo`, `abolicao_1888_protagonismo_negro`, `quilombo_palmares_zumbi`, `era_vargas_ambiguidades`, `quiz_proclamacao_republica`, `quiz_primeiro_e_segundo_reinado`, `epistemic_honesty_fake_treaty`, `epistemic_honesty_anachronistic_vargas_1822`.
- **Stratified 4-case scaling subset** (1 per category, used for `qwen2.5:3b`): `colonial_indigenas_eurocentrismo`, `era_vargas_ambiguidades`, `quiz_proclamacao_republica`, `epistemic_honesty_fake_treaty`.
- **Matched 4-case native ADK eval subset** (for `agents-cli eval generate`): `colonial_indigenas_eurocentrismo`, `abolicao_1888_protagonismo_negro`, `era_vargas_ambiguidades`, `epistemic_honesty_fake_treaty`.

---

## 3. Quantitative Results

### 3.1 Cross-Model Aggregate Comparison (Completed Cases)

> Note: Metrics below aggregate **completed (non-errored)** executions per model so operational latency and quality scores are not distorted by HTTP 503 cloud outages; completion rate (`Completed / Attempted`) is reported explicitly in its own column. Blind LLM-as-a-judge used `gemini-3.5-flash-lite` with all model/profile identifiers redacted (`evals.judge.sanitize_response_for_blind_judging`).

| Profile | Model | Offline | Completed / Attempted | Det. Accuracy (/5) | Fact Recall | Det. Hallucination Cases | Det. Pedagogy (/5) | Blind Judge Acc (/5) | Blind Judge Ped (/5) | Blind Judge Overall (/5) | Mean Latency (ms) | p95 Latency (ms) | Mean Tokens/s (e2e) | Mean Input / Output Tokens | Est. Cost / 1k Queries (USD) | RAM / VRAM |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `local` | `ollama_chat/qwen2.5:1.5b` | **Yes** | **8/8 (100%)** | 2.55 | 51.0% | 0/8 | 3.93 | 1.72 *(n=3)* | 1.50 *(n=3)* | **1.61** *(n=3)* | 241,149 | 355,945 | 8.75 | 7,660 / 2,225 | **$0.0000** | 1.56 GB RAM / 0 VRAM |
| `local` | `ollama_chat/qwen2.5:3b` | **Yes** | **4/4 (100%)**\* | 1.88 | 37.5% | 1/4 | 3.52 | 1.00 *(n=2)* | 1.25 *(n=2)* | **1.13** *(n=2)* | 553,416 | 702,397 | 4.15 | 10,184 / 2,292 | **$0.0000** | 2.56 GB RAM / 0 VRAM |
| `local` | `ollama_chat/llama3.2:3b` | **Yes** | **1/1 (100%)**\* | 2.50 | 50.0% | 0/1 | 4.08 | N/A *(503)* | N/A *(503)* | N/A *(503)* | 707,451 | 707,451 | 3.08 | 11,475 / 2,182 | **$0.0000** | 2.96 GB RAM / 0 VRAM |
| `local` | `ollama_chat/qwen3:8b` | **Yes** | 0/1 (timeout) | — | — | — | — | — | — | — | >600,000 | >600,000 | 3.54\*\* | — | **$0.0000** | 6.33 GB RAM / 0 VRAM |
| `local` | `ollama_chat/gemma3:4b` | **Yes** | 0/1 (HTTP 400) | — | — | — | — | — | — | — | — | — | — | — | **$0.0000** | 3.35 GB disk / no tools |
| `remote` | `gemini-3.5-flash-lite` | No | 3/8 (37.5%)\*\*\* | 4.17 | 83.3% | **0/3** | 4.14 | **5.00** *(n=3)* | **5.00** *(n=3)* | **5.00** *(n=3)* | **61,275** | **81,490** | 38.73 | 15,104 / 2,243 | **$2.4077** | Cloud (0 local) |
| `remote` | `gemini-3-flash-preview` | No | 1/1 (then 20 RPD) | 3.75 | 75.0% | **0/1** | 4.17 | **5.00** *(n=1)* | **5.00** *(n=1)* | **5.00** *(n=1)* | **33,346** | **33,346** | **92.19** | 25,297 / 3,074 | $21.8705\*\*\*\* | Cloud (0 local) |
| `remote` | `gemini-4-hydrogen` | No | 1/1 (then 503) | **5.00** | **100.0%** | **0/1** | 4.17 | **5.00** *(n=1)* | **5.00** *(n=1)* | **5.00** *(n=1)* | 217,178 | 217,178 | 26.12 | 42,584 / 5,673 | $106.4230 | Cloud (0 local) |

- \* Post-fix completion after adding stringified-integer tool coercion (`app/tools/`) and safe `build_quiz_blueprint` fallback for non-corpus RAG chunks (`app/tools/quiz.py`).
- \*\* Direct `/api/chat` CPU decode throughput for `qwen3:8b` (`3.54 tok/s` with `think=True`, `3.71 tok/s` with `think=False`).
- \*\*\* 5 of 8 cases on `gemini-3.5-flash-lite` failed after retries due to upstream Google AI Studio Free Tier `503 UNAVAILABLE` spikes / 180 s timeout.
- \*\*\*\* Calculated at updated `gemini-3-flash-preview` paid-tier reference pricing (`$0.50` input / `$3.00` output per 1M tokens); the raw probe JSON captured prior to adding the price table entry logged `$5.6390/1k` using the generic default `(0.15, 0.60)`.

### 3.2 Strictly Paired Head-to-Head Breakdown (Identical Case IDs)

#### A. Paired 3-Case Comparison: `gemini-3.5-flash-lite` vs. `ollama_chat/qwen2.5:1.5b`

| Case ID | Category | Level | `gemini-3.5-flash-lite` (Det. Acc / Judge / Latency / Cost) | `ollama_chat/qwen2.5:1.5b` (Det. Acc / Judge / Latency / Cost) |
| :--- | :--- | :--- | :--- | :--- |
| `quilombo_palmares_zumbi` | `factual_and_critical` | `fundamental_2` | `5.00` / **`5.00`** (`ok`) / `50.9 s` / `$0.00222` | `3.75` / **`1.58`** (`ok`) / `280.4 s` / `$0.00` |
| `quiz_proclamacao_republica` | `formative_assessment` | `fundamental_2` | `5.00` / **`5.00`** (`ok`) / `51.4 s` / `$0.00277` | `5.00` / `fallback (503)` / `355.9 s` / `$0.00` |
| `epistemic_honesty_fake_treaty` | `epistemic_honesty` | `geral` | `2.50` / **`5.00`** (`ok`) / `81.5 s` / `$0.00223` | `2.50` / `fallback (503)` / `257.3 s` / `$0.00` |
| **Paired Mean (n = 3)** | — | — | **Det. `4.17` \| Recall `83.3%` \| Judge `5.00` \| `61.3 s` \| `$2.41/1k`** | **Det. `3.75` \| Recall `75.0%` \| `297.9 s` (+4.9×) \| `$0.00/1k`** |

#### B. Single-Case Cross-Family Comparison (`colonial_indigenas_eurocentrismo`, 7th Grade Misconception Trap)

| Model | Profile | Det. Accuracy (/5) | Fact Recall | Hallucinations | Det. Pedagogy (/5) | Blind Judge (/5) | Latency (s) | Tokens (In / Out) | Tokens/s (e2e) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `gemini-4-hydrogen` | `remote` | **5.00** | **100%** | None | 4.17 | **5.00** (`ok`) | 217.2 s | 42,584 / 5,673 | 26.12 |
| `gemini-3-flash-preview` | `remote` | 3.75 | 75% | None | 4.17 | **5.00** (`ok`) | **33.3 s** | 25,297 / 3,074 | **92.19** |
| `ollama_chat/qwen2.5:1.5b` | `local` | 2.50 | 50% | None | 4.17 | fallback (503) | 243.8 s | 7,839 / 2,348 | 9.63 |
| `ollama_chat/qwen2.5:3b` | `local` | 2.50 | 50% | None | 3.75 | fallback (503) | 668.2 s | 8,825 / 2,679 | 4.01 |
| `ollama_chat/llama3.2:3b` | `local` | 2.50 | 50% | None | 4.08 | fallback (503) | 707.5 s | 11,475 / 2,182 | 3.08 |

#### C. Full 8-Case Local Baseline (`ollama_chat/qwen2.5:1.5b`) & 4-Case Scaling (`ollama_chat/qwen2.5:3b`)

| Case ID | Category | Level | `qwen2.5:1.5b` Det. Acc / Ped / Judge / Latency | `qwen2.5:3b` Det. Acc / Ped / Judge / Latency |
| :--- | :--- | :--- | :--- | :--- |
| `colonial_indigenas_eurocentrismo` | `misconception_trap` | `fundamental_2` | `2.50` / `4.17` / fallback / `243.8 s` | `2.50` / `3.75` / fallback / `668.2 s` |
| `abolicao_1888_protagonismo_negro` | `misconception_trap` | `ensino_medio` | `3.75` / `4.25` / **`3.08` (`ok`)** / `315.5 s` | — |
| `quilombo_palmares_zumbi` | `factual_and_critical` | `fundamental_2` | `3.75` / `3.75` / **`1.58` (`ok`)** / `280.4 s` | — |
| `era_vargas_ambiguidades` | `factual_and_critical` | `ensino_medio` | `1.25` / `4.25` / **`0.17` (`ok`)** / `237.5 s` | `1.25` / `2.50` / **`1.92` (`ok`)** / `404.4 s` |
| `quiz_proclamacao_republica` | `formative_assessment` | `fundamental_2` | `5.00` / `3.75` / fallback / `355.9 s` | `3.75` / `3.75` / fallback / `438.7 s` |
| `quiz_primeiro_e_segundo_reinado` | `formative_assessment` | `ensino_medio` | `0.00` / `4.25` / fallback / `191.6 s` | — |
| `epistemic_honesty_fake_treaty` | `epistemic_honesty` | `geral` | `2.50` / `4.50` / fallback / `257.3 s` | `0.00` (hallucinated!) / `4.08` / **`0.34` (`ok`)** / `702.4 s` |
| `epistemic_honesty_anachronistic_vargas_1822` | `epistemic_honesty` | `fundamental_2` | `1.67` / `2.50` / fallback / `47.3 s` | — |

---

## 4. Native `google-agents-cli eval` Trace & Diff Capture

We executed Google's official `uvx google-agents-cli` (`v1.15.0`) evaluation pipeline on the matched 4-case subset from `tests/eval/datasets/brazil_history_dataset.json`:

1. **`eval generate` (`local`, `ollama_chat/qwen2.5:1.5b`, `--concurrency 1`):**
   - Booted the local ADK server (`http://127.0.0.1:18080`, `app_name=app`) and captured native ADK traces in `artifacts/traces/traces_20261123_134919.json`.
   - **2/4 cases succeeded:**
     - `colonial_indigenas_eurocentrismo`: `1` turn, `4` events (`TopicRouter -> HistoricalResearcher -> PedagogicalAdapter -> FactVerifier`), `0` tool calls (skipped corpus tool on this trace).
     - `epistemic_honesty_fake_treaty`: `1` turn, `6` events (`TopicRouter -> HistoricalResearcher [tool call + response + synthesis] -> PedagogicalAdapter -> FactVerifier`), `1` tool call.
   - **2/4 cases failed:** `abolicao_1888_protagonismo_negro` exceeded `agents-cli`'s built-in `/run_sse` HTTP read timeout on CPU-only inference; `era_vargas_ambiguidades` hit `MAX_TOKENS` (`max_tokens=1024`).
2. **`eval generate` (`remote`, `gemini-3.5-flash-lite`, `--concurrency 1`):**
   - `0/4` succeeded during the Free Tier peak window: cases 0–1 returned `503 UNAVAILABLE` and cases 2–3 immediately tripped Google AI Studio Free Tier's `15 RPM` quota (`GenerateRequestsPerMinutePerProjectPerModel-FreeTier`) because `agents-cli eval generate` has no built-in inter-case delay flag (unlike `evals/runner.py --remote-delay 6.0`).
3. **`eval grade` (`v1.15.0`):**
   - `cmd_grade.py` initializes `AgentPlatformClient(project=None, location=None)`, which triggers `google.auth.default()` and fails on workstations without GCP Application Default Credentials even when only local custom metrics (`tests/eval/response_quality.py`) with `GEMINI_API_KEY` are configured. Grading was therefore executed through our blind `evals/judge.py` engine using `GEMINI_API_KEY`.
4. **`uvx google-agents-cli eval compare` (Paired 3-Case Summary Diff: `gemini-3.5-flash-lite` baseline vs. `ollama_chat/qwen2.5:1.5b` candidate):**

```json
{
  "differences": {
    "metrics.avg_accuracy_5": {"baseline": 4.17, "candidate": 3.75, "delta": "-0.42"},
    "metrics.avg_fact_recall": {"baseline": 0.833, "candidate": 0.75, "delta": "-0.083"},
    "metrics.avg_pedagogy_5": {"baseline": 4.14, "candidate": 4.0, "delta": "-0.14"},
    "metrics.estimated_cost_per_1k_queries_usd": {"baseline": 2.4077, "candidate": 0.0, "delta": "-2.4077"},
    "metrics.mean_latency_ms": {"baseline": 61275.1, "candidate": 297857.7, "delta": "+236582.6"},
    "metrics.mean_tokens_per_second": {"baseline": 38.73, "candidate": 9.66, "delta": "-29.07"},
    "metrics.p95_latency_ms": {"baseline": 81489.76, "candidate": 355944.94, "delta": "+274455.18"},
    "model_id": {"baseline": "gemini-3.5-flash-lite", "candidate": "ollama_chat/qwen2.5:1.5b"},
    "offline_capable": {"baseline": false, "candidate": true, "delta": "+1"}
  }
}
```

---

## 5. Qualitative Analysis by Evaluation Pillar

### 5.1 Historical Accuracy & Hallucination Resistance

- **Remote Gemini (`gemini-3.5-flash-lite`, `gemini-3-flash-preview`, `gemini-4-hydrogen`):** Consistently grounded answers in retrieved hybrid RAG sources (citing *Carta de Pero Vaz de Caminha*, Serra da Capivara, Lagoa Santa / "Luzia", cultura Marajoara, Confederação dos Tamoios, Lei 11.645/2008, Zumbi & Ganga Zumba, 1695, 1889/1891, and accurate BNCC codes). On `epistemic_honesty_fake_treaty`, `gemini-3.5-flash-lite` cleanly debunked the fictional 1742 treaty and explained Dom Pedro II's real 19th-century chronology (`5.00/5` blind judge).
- **Local Small Models (`qwen2.5:1.5b`, `qwen2.5:3b`, `llama3.2:3b`):**
  - **Semantic hallucinations invisible to keyword rubrics:** On `quilombo_palmares_zumbi`, `qwen2.5:1.5b` achieved `3.75/5` deterministic accuracy (matching `palmares`, `zumbi`, `ganga zumba`), yet the blind LLM judge exposed severe errors (`1.58/5`): it located Palmares in **Goiás instead of Alagoas**, called the colonial era a **"dictatorship"**, and **invented BNCC codes `4.1.1` and `4.1.2`**.
  - **Topic confusion & repetition loops:** On `era_vargas_ambiguidades`, `qwen2.5:1.5b` confused Era Vargas (1930–1945) with the 1964–1985 Military Dictatorship and entered a repetition loop (`0.17/5` judge). Scaling to `qwen2.5:3b` fixed the period confusion (`1.92/5` judge) but still omitted *Estado Novo* and censorship.
  - **Sycophancy on fictional premises:** On `epistemic_honesty_fake_treaty`, `qwen2.5:3b` endorsed the fictional *Tratado de Guanabara de 1742* and fabricated diplomatic/trade terms between Dom Pedro II and Japan (`0.00/5` det., `0.34/5` judge).
  - **Spurious tool invocation across nodes:** On misconception questions (`colonial_indigenas_eurocentrismo`), `PedagogicalAdapter` on `llama3.2:3b`, `qwen2.5:3b`, and `qwen2.5:1.5b` frequently invoked `build_quiz_blueprint` unprompted and leaked internal blueprint guidelines (`"Elabore questões em português do Brasil..."`) into the student-facing response.

### 5.2 UN SDG 4 Pedagogical Quality & Grade-Level Adaptation

- **Remote Gemini:** Produced cohesive Brazilian Portuguese (`pt-BR`), calibrated tone and vocabulary cleanly between 7th–9th grade (`fundamental_2`) and High School/ENEM (`ensino_medio`), incorporated Indigenous, Afro-Brazilian, and women's perspectives (SDG 4.7), and ended with active reflection questions.
- **Local 1.5B–3B Models:** Heuristic pedagogy scores (`3.52–4.08/5`) overestimated quality because they reward markdown formatting, bullet points, and word counts. Blind LLM-as-a-judge pedagogy scores (`1.25–1.50/5`) accurately captured broken structure, leaked internal `Rascunho Didático` headers, repetitive blocks, and missing reflection depth.

### 5.3 Operational Performance, Cost, & Offline Viability

- **Cost vs. Latency Trade-Off:**
  - `gemini-3.5-flash-lite` is dramatically cost-effective at paid-tier rates (**~$2.41 per 1,000 multi-node queries**), while `gemini-4-hydrogen` is ~44× more expensive (**~$106.42 per 1,000 queries**) due to reasoning token amplification (`5,673` output tokens across 4 nodes).
  - Local Ollama has **$0.00 marginal API cost** and **100% offline availability**, but a 4-node sequential workflow + ~6,200-token RAG tool payloads pushes CPU-only 1.5B–3B latency to **4–12 minutes per query** — unusable for live classroom chat without either (a) GPU offload (4–8 GB VRAM), (b) compacting RAG tool payloads to fit comfortably within a 4k–8k context window, or (c) evaluating a single-turn / lighter workflow graph for constrained local hardware.

---

## 6. Engineering Defects Discovered & Fixed During Baseline Runs

Running live multi-model benchmarks uncovered four concrete production defects that were fixed and covered by offline unit tests while preserving strict local/remote parity:

1. **LiteLLM `1.106.9` MCP-gateway crash on tool-calling Ollama nodes (`app/models.py`):** Passing `tools` to `LiteLlm` triggered `ModuleNotFoundError: No module named 'mcp'`. Fixed by passing `_skip_mcp_handler=True`, along with `temperature=settings.temperature` (Ollama had been defaulting to `0.8` instead of `0.2`) and `max_tokens=1024` (preventing 10-minute `--context-shift` infinite generation loops on Ollama's default `n_ctx=4096`).
2. **Stringified integer tool arguments from small local LLMs (`app/tools/corpus.py`, `timeline.py`, `quiz.py`):** `llama3.2:3b` emitted `max_results="3"` instead of `3`, raising `TypeError`. Fixed via defensive integer coercion in all three tools.
3. **Non-corpus RAG chunk crash in `build_quiz_blueprint` (`app/tools/quiz.py`):** When hybrid RAG returned a BNCC/primary-source chunk first (`base_entry is None`), `entry["key_facts"]` raised `KeyError` (triggered twice by `qwen2.5:3b`). Fixed with safe `.get(...)` fallbacks.
4. **Transient 503/504 handling, daily-quota fail-fast, and socket timeouts (`evals/runner.py`, `evals/judge.py`, `tests/eval/response_quality.py`):** Added 503/504/timeout retry handling, fail-fast on non-retryable daily quota errors (`PerDay`), 180 s remote per-case timeout, 60 s judge HTTP timeout, incremental progress logging, and `EVAL_JUDGE_MODEL` env support in `tests/eval/response_quality.py` (retaining `gemini-3.8-flash` as default fallback).

---

## 7. Reproducibility Commands

```bash
# 1. Start local Ollama and pull tool-calling candidates + embedder
make ollama-up
make ollama-pull-embed
make ollama-pull OLLAMA_MODEL=qwen2.5:1.5b
make ollama-pull OLLAMA_MODEL=qwen2.5:3b
make ollama-pull OLLAMA_MODEL=llama3.2:3b

# 2. Build the shared offline hybrid RAG index (158 chunks, nomic-embed-text + BM25)
make rag-index

# 3. Run offline verification first
make check
make benchmark-dry-run

# 4. Run live comparative benchmark (adjust GEMINI_MODEL / OLLAMA_MODEL / --remote-delay as needed)
GEMINI_MODEL=gemini-3.5-flash-lite EVAL_JUDGE_MODEL=gemini-3.5-flash-lite OLLAMA_MODEL=qwen2.5:1.5b \
  uv run python -m evals.runner --profiles remote local --remote-delay 6.0 --use-judge

# 5. Capture native ADK traces and diff two evaluation summary JSONs
MODEL_PROFILE=local OLLAMA_MODEL=qwen2.5:1.5b \
  uvx google-agents-cli eval generate --dataset tests/eval/datasets/brazil_history_dataset.json --concurrency 1
uvx google-agents-cli eval compare <baseline_summary.json> <candidate_summary.json>
```

---

## 8. Recommendations for Issue #8 (Hybrid RAG Ablation) & Next Iterations

1. **Test Hybrid RAG Ablation (`--disable-tools` in Issue #8):** Quantify exactly how much grounded retrieval reduces hallucinations on 1.5B–3B local models vs. how much the ~6.2k-token tool payload inflates latency and triggers context truncation on `n_ctx=4096`.
2. **Compact Tool Payloads / Increase `num_ctx`:** Deduplicating `matched_chunks` vs. top-level `chunks` in `search_historical_corpus` (or trimming excerpts to concise snippets) can cut per-query input tokens by ~50–60%, slashing both CPU-only local latency and remote paid-tier token costs.
3. **Evaluate GPU-Offloaded 7B–8B Local Models (`qwen3:8b` with `think=False`, `qwen2.5:7b`, `llama3.1:8b`):** With `nvidia-container-toolkit` enabled on an 8 GB VRAM GPU (or Apple Silicon unified memory), 7B–8B tool-calling models are expected to bridge most of the accuracy/pedagogy gap observed at 1.5B–3B while maintaining sub-15 s latency and 100% offline SDG 4.a viability.
