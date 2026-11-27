# Compact-RAG & GPU Optimization Experiment: Before (`#7`/`#8`) vs. After (`#17`–`#20`)

> **Issue:** `#20` (`perf(local): add NVIDIA GPU Docker Compose override and re-benchmark compact-RAG local models`)  
> **Date (UTC):** `2026-11-27` (`20:32Z – 22:46Z`)  
> **System Under Test:** Brasão AI — ADK 2.0 4-node `Workflow` (`TopicRouter -> HistoricalResearcher -> PedagogicalAdapter -> FactVerifier`), `ROOT_AGENT_NAME = "brasao_ai"`, compacted `search_historical_corpus` payload (`<= 8 KiB`, `#17`), `OLLAMA_NUM_CTX=8192` & `OLLAMA_THINK` support (`#17`), profile-agnostic output sanitization & quiz-tool guards (`#18`), incremental `--resume` checkpointing & offline SVG charts (`#19`), and optional `docker-compose.gpu.yml` NVIDIA GPU override (`#20`).  
> **Sampling & Parity Control:** `MODEL_TEMPERATURE = 0.2`, identical 4-node workflow topology, identical hybrid RAG index (`app/data/rag_index.sqlite3`, `158` chunks, BM25 + `nomic-embed-text` 768-dim RRF), zero profile-specific prompt forks.

---

## 1. Executive Summary

This experiment quantifies the real-world **Before (`#7`/`#8`) vs. After (`#17`–`#20`)** impact of:
1. Compacting `search_historical_corpus` payloads by **-67.7%** (`24.9 KB / ~6,219 tokens` → **`8,043 bytes / ~1,961 tokens`**) and raising Ollama's context window from `4096` → **`8192`** (`OLLAMA_NUM_CTX=8192`) with optional `OLLAMA_THINK=false` (`#17`).
2. Enforcing profile-agnostic draft-leak sanitization and non-quiz `build_quiz_blueprint` tool guards (`#18`).
3. Adding an optional NVIDIA GPU Docker Compose override (`docker-compose.gpu.yml` + `make ollama-up-gpu`) while keeping `docker-compose.yml` portable and CPU-only by default for commodity public school hardware (`#20`).

**Headline empirical findings (all measurements live and non-fabricated):**

1. **Host GPU Passthrough Probe — Honest CPU Fallback:** Although the host carries an **NVIDIA GeForce GTX 1650 (4,096 MiB VRAM, Driver `615.78.02`, CUDA `13.4`)**, `docker info` exposes only `runc` and `io.containerd.runc.v2` (`nvidia-ctk` / `nvidia-container-runtime` are not installed on the WSL2 host daemon). Running `docker compose -f docker-compose.yml -f docker-compose.gpu.yml config --quiet` validates cleanly, while `up -d ollama` fails at container start with `Error response from daemon: could not select device driver "nvidia" with capabilities: [[gpu]]`. Accordingly, all local runs below ran honestly in **CPU-only mode (`size_vram: 0`)** via `docker compose up -d ollama`, isolating the pure software impact of `#17`–`#19` on commodity school hardware.
2. **Remote Gemini (`gemini-3.5-flash-lite`) Achieved -80.7% Latency Reduction & 100% Completion (`4.99/5` Blind Judge):** Across all **8/8** stratified cases (`0` errors vs. `3/8` in `#7`), mean latency dropped from **`61.3 s` → `11.8 s`** (**-80.7% / 5.2× faster**, p95 `81.5 s → 15.3 s`), throughput surged **5.95×** (`38.7 → 230.6 tok/s`), deterministic accuracy rose from **`4.17 → 4.69 / 5.00`** (`83.3% → 93.8%` fact recall), and blind LLM-as-a-judge averaged **`4.99 / 5.00`** (`7/8` perfect `5.00/5`, `1/8` `4.92/5`) at **`$2.64 / 1k` queries**.
3. **`qwen2.5:1.5b` (CPU-Only) Jumped +38.8% in Deterministic Accuracy & +68.9% on Paired Blind Judge:** On the identical 8-case suite (`8/8` completed, `8/8` judge-graded), deterministic accuracy rose from **`2.55 → 3.54 / 5.00`** (**+19.8 pp fact recall**, `51.0% → 70.8%`), output tokens dropped **-20.0%** (`2,225 → 1,780`), and blind judge rose from `1.61/5` (`n=3` in `#7`) to **`2.10/5` (`n=8`)** — and **`2.72/5` (+1.11 pts / +68.9%)** on the exact 3 paired cases judged in `#7`, led by `quilombo_palmares_zumbi` jumping from **`1.58 → 4.75 / 5.00`** (`5.00/5` judge accuracy, eliminating the Goiás / invented BNCC `4.1.1` hallucinations).
4. **`qwen2.5:3b` (CPU-Only) Reversed Its `#7`/`#8` RAG Regression (+185.8% Blind Judge, 0 Hallucinations):** On the exact 4-case scaling subset (`4/4` completed & judge-graded), blind LLM-as-a-judge surged from **`1.13 → 3.23 / 5.00`** (**+2.10 pts / +185.8%**), deterministic hallucinations dropped from **`1/4 → 0/4`** (`epistemic_honesty_fake_treaty` jumped from `0.34/5` sycophancy to `3.42/5` rejection), output tokens fell **-35.3%** (`2,292 → 1,484`), and mean / p95 latency fell **`553.4 s → 501.4 s` (-9.4%)** and **`702.4 s → 557.2 s` (-20.7%)** even after doubling `num_ctx` from `4096` to `8192`.
5. **`qwen3:8b` (`OLLAMA_THINK=false`, `OLLAMA_NUM_CTX=8192`) Achieved `4.83 / 5.00` Blind Judge Offline — Confirming GPU Offload as the Final Unlock:** Whereas `qwen3:8b` timed out (`0/1`) in `#7` with default `<think>` reasoning, disabling thinking + compacting RAG allowed `qwen3:8b` to complete `colonial_indigenas_eurocentrismo` offline with **`5.00/5` blind judge accuracy, `4.67/5` pedagogy, and `4.83/5` overall** (matching `gemini-3-flash-preview` and exceeding `gemini-3.1-flash-lite-preview`). However, without GPU passthrough (`size_vram: 0`), CPU prefill at `13.1 tok/s` across 4 nodes took **`2,269.9 s` (`37.8 min`)** and still hit LiteLLM's `600 s` per-request socket timeout on `epistemic_honesty_fake_treaty` (`1/2` completed) — quantifying exactly why GPU offload (`docker-compose.gpu.yml`) or unified-memory hardware is required for classroom-speed 8B offline tutoring.
6. **Spurious Quiz-Tool & Draft-Leak Elimination (`#18`):** Across all **21 completed responses** in this study, spurious `build_quiz_blueprint` guidance leaks dropped to **`0/21` (0%)**, and internal workflow header leaks dropped to **`1/21` (4.8%)** (`0/8` Remote Gemini, `0/4` `qwen2.5:3b`, `0/1` `qwen3:8b`, `1/8` `qwen2.5:1.5b`).

---

## 2. Host GPU Passthrough Verification & Hardware Provenance

### 2.1 Host Probe Results (`2026-11-27`)

| Check | Command | Observed Result |
| :--- | :--- | :--- |
| **Host GPU (`nvidia-smi`)** | `nvidia-smi` | `NVIDIA GeForce GTX 1650` (`4,096 MiB` VRAM, Driver `615.78.02`, CUDA `13.4`, WSL2) |
| **Docker Runtimes** | `docker info --format '{{.Runtimes}}'` | `io.containerd.runc.v2`, `runc` (**no `nvidia` runtime registered**) |
| **NVIDIA Container Toolkit** | `which nvidia-ctk nvidia-container-runtime` | Not installed (empty output) |
| **Compose Override Validation** | `docker compose -f docker-compose.yml -f docker-compose.gpu.yml config --quiet` | **Exit 0 (`COMPOSE_CONFIG_OK`)** — valid Compose YAML merge |
| **GPU Container Start Probe** | `make ollama-up-gpu` (`docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d ollama`) | Container recreated with `DeviceRequests: [{"Driver":"nvidia","Count":-1,"Capabilities":[["gpu"]]}]`, then daemon returned `Error response from daemon: could not select device driver "nvidia" with capabilities: [[gpu]]` |
| **Active Fallback Runtime** | `make ollama-up` (`docker compose up -d ollama`) | `brasao-ai-ollama` (`ollama/ollama:latest`, `v0.40.17`), `Status=running`, `DeviceRequests=null`, **`size_vram: 0`** (CPU-only) |

### 2.2 Observed Ollama Memory & Context Footprint (`OLLAMA_NUM_CTX=8192`, `/api/ps`)

| Model Tag | Params / Quant | Active RAM (`size`) | Combined RAM (+ `nomic-embed-text` `389 MB`) | Active VRAM (`size_vram`) | Active `context_length` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `nomic-embed-text:latest` | 137M / F16 | 389 MB | — | **0 MB** | 2,048 |
| `qwen2.5:1.5b` | 1.5B / Q4_K_M | 1.37 GB *(vs. 1.17 GB @ 4096)* | **1.76 GB** | **0 MB** | **8,192** |
| `qwen2.5:3b` | 3.1B / Q4_K_M | 2.42 GB *(vs. 2.17 GB @ 4096)* | **2.81 GB** | **0 MB** | **8,192** |
| `qwen3:8b` (`OLLAMA_THINK=false`) | 8.2B / Q4_K_M | 6.68 GB *(vs. 5.95 GB @ 4096)* | **7.07 GB** | **0 MB** | **8,192** |

### 2.3 Compacted Hybrid RAG Payload Verification (`search_historical_corpus`)

Query: `"Povos indígenas antes de 1500 no Brasil"` (`max_results=3`, hybrid BM25 + `nomic-embed-text` RRF over `158` chunks):

| Metric | Before (`#7`/`#8` Uncompacted) | After (`#17` Compacted) | Reduction |
| :--- | :---: | :---: | :---: |
| **Tool Payload Size (UTF-8 Bytes)** | ~24,900 B | **8,043 B** *(BM25 lexical: 7,989 B)* | **-67.7%** |
| **Estimated Tool Payload Tokens (`~4 chars/tok`)** | ~6,219 tokens | **~1,961 tokens** | **-68.5%** |
| **Fits in Ollama Context Window Without Truncation?** | No (`6.2k` > `n_ctx=4096`) | **Yes (`1.96k` << `n_ctx=8192`)** | Context shift eliminated |

---

## 3. Quantitative "Before vs. After" Comparison

### 3.1 Aggregate Before (`#7`/`#8`) vs. After (`#17`–`#20`) Summary Table

> **Reporting Convention:** Consistent with `#7` and `#8`, metrics below summarize **completed (non-errored)** executions alongside explicit `Completed / Attempted` completion rates. All blind LLM-as-a-judge runs used `gemini-3.5-flash-lite` with model/profile identifiers redacted (`evals.judge.sanitize_response_for_blind_judging`).

| Configuration | Phase | Offline | Completed / Attempted | Det. Acc (/5) | Fact Recall | Det. Halluc. | Det. Ped (/5) | Blind Judge Acc (/5) | Blind Judge Ped (/5) | Blind Judge Overall (/5) | Mean Latency (s) | p95 Latency (s) | Mean Tokens (In / Out) | Tokens/s | Est. Cost / 1k (USD) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Remote `gemini-3.5-flash-lite`** | Before (`#7`) | No | 3/8 (37.5%) | 4.17 | 83.3% | 0/3 | 4.14 | 5.00 *(n=3)* | 5.00 *(n=3)* | 5.00 *(n=3)* | 61.3 s | 81.5 s | 15,104 / 2,243 | 38.73 | $2.4077 |
| **Remote `gemini-3.5-flash-lite`** | **After (`#20`)** | No | **8/8 (100%)** | **4.69** *(+12.5%)* | **93.8%** *(+10.5 pp)* | **0/8** | **4.29** | **5.00** *(n=8)* | **4.98** *(n=8)* | **4.99** *(n=8)* | **11.8 s** *(-80.7%)* | **15.3 s** *(-81.3%)* | 15,614 / 2,694 | **230.59** *(5.95×)* | $2.6390 |
| **Local `qwen2.5:1.5b` (No RAG)** | Before (`#8` Arm 1) | **Yes** | 8/8 (100%) | 1.88 | 37.5% | 0/8 | 3.81 | 0.61 *(n=6)* | 1.11 *(n=6)* | 0.86 *(n=6)* | 205.9 s | 429.5 s | 4,962 / 1,775 | 8.92 | $0.0000 |
| **Local `qwen2.5:1.5b` (RAG, `n_ctx=4096`)** | Before (`#7`) | **Yes** | 8/8 (100%) | 2.55 | 51.0% | 0/8 | 3.93 | 1.72 *(n=3)* | 1.50 *(n=3)* | 1.61 *(n=3)* | 241.1 s | 355.9 s | 7,660 / 2,225 | 8.75 | $0.0000 |
| **Local `qwen2.5:1.5b` (RAG, `n_ctx=4096`)** | Before (`#8` Arm 2) | **Yes** | 8/8 (100%) | 2.55 | 51.0% | 0/8 | 3.81 | — | — | — | 280.9 s | 414.6 s | 7,600 / 1,957 | 6.62 | $0.0000 |
| **Local `qwen2.5:1.5b` (Compact RAG, `n_ctx=8192`)** | **After (`#20`)** | **Yes** | **8/8 (100%)** | **3.54** *(+38.8%)* | **70.8%** *(+19.8 pp)* | **0/8** | 3.74 | **2.48** *(n=8)* | **1.73** *(n=8)* | **2.10** *(n=8; **2.72** on #7 n=3)* | 294.1 s\*\* | 526.6 s | 9,598 / **1,780** *(-20.0% out)* | 5.91 | **$0.0000** |
| **Local `qwen2.5:3b` (RAG, `n_ctx=4096`, 4 cases)** | Before (`#7`) | **Yes** | 4/4 (100%) | 1.88 | 37.5% | 1/4 | 3.52 | 1.00 *(n=2)* | 1.25 *(n=2)* | 1.13 *(n=2)* | 553.4 s | 702.4 s | 10,184 / 2,292 | 4.15 | $0.0000 |
| **Local `qwen2.5:3b` (Compact RAG, `n_ctx=8192`, 4 cases)** | **After (`#20`)** | **Yes** | **4/4 (100%)** | 1.56 | 31.2% | **0/4** | **3.60** | **3.25** *(n=4)* | **3.21** *(n=4)* | **3.23** *(+185.8%, n=4)* | **501.4 s** *(-9.4%)* | **557.2 s** *(-20.7%)* | 17,270 / **1,484** *(-35.3% out)* | 3.00 | **$0.0000** |
| **Local `qwen3:8b` (RAG, `think=True`, `n_ctx=4096`)** | Before (`#7`) | **Yes** | 0/1 (timeout) | — | — | — | — | — | — | — | >600 s | >600 s | — | 3.54\* | $0.0000 |
| **Local `qwen3:8b` (Compact RAG, `think=False`, `n_ctx=8192`)** | **After (`#20`)** | **Yes** | **1/2 (50%)**\*\*\* | **3.75** *(n=1)* | **75.0%** *(n=1)* | **0/1** | **4.50** | **5.00** *(n=1)* | **4.67** *(n=1)* | **4.83** *(n=1)* | 2,269.9 s *(n=1)* | 2,269.9 s | 35,651 / **1,371** | 0.60 *(e2e)* | **$0.0000** |

- \* Direct single-prompt `/api/chat` CPU decode throughput measured in `#7`.
- \*\* **Why `qwen2.5:1.5b` mean latency rose (`241.1 s → 294.1 s`) despite 40–46% speedups on 4 cases:** Raising `OLLAMA_NUM_CTX` from `4096` → `8192` stopped Ollama's silent `context-shift` truncation. On 5 of 8 cases (`abolicao`, `quiz_proclamacao`, `quiz_reinado`, `anachronistic_vargas`), compact payloads reduced latency sharply (`-17%` to `-46%`), whereas on 3 multi-call cases (`quilombo_palmares_zumbi`, `era_vargas_ambiguidades`, `epistemic_honesty_fake_treaty`), full untruncated `3.8k–4.4k` prompts on `FactVerifier` spent ~60 s per node in CPU prompt prefill (`~63 tok/s`) — directly driving the `+19.8 pp` fact recall and `+1.11` paired judge gain.
- \*\*\* On `qwen3:8b`, `colonial_indigenas_eurocentrismo` completed cleanly (`4.83/5` blind judge, `2,269.9 s`), while `epistemic_honesty_fake_treaty` exceeded LiteLLM's `600 s` per-request socket timeout on CPU prefill (`13.1 tok/s`). In the multi-model SVG chart (`compact_rag_cross_model`), `evals.charts` plots the raw 2-attempt aggregate (`1/2 ok`, `1,135.0 s` raw mean including the `0 ms` failed case) alongside the `50%` completion bar.

---

### 3.2 Visual Comparison Charts (Generated Offline via `evals.charts`)

#### A. Paired 8-Case Comparison: Remote `gemini-3.5-flash-lite` vs. Local `qwen2.5:1.5b` (Compact RAG, `num_ctx=8192`)

![Accuracy & Pedagogy — Gemini 3.5 Flash Lite vs. Qwen2.5 1.5B](charts/compact_rag_1p5b_vs_gemini/accuracy_pedagogy.svg)

![Latency vs. Accuracy — Gemini 3.5 Flash Lite vs. Qwen2.5 1.5B](charts/compact_rag_1p5b_vs_gemini/latency_vs_accuracy.svg)

![Cost & Offline Viability — Gemini 3.5 Flash Lite vs. Qwen2.5 1.5B](charts/compact_rag_1p5b_vs_gemini/cost_and_offline.svg)

#### B. Cross-Model Scaling Under Compact RAG (`gemini-3.5-flash-lite` vs. `qwen2.5:1.5b` vs. `qwen2.5:3b` vs. `qwen3:8b`)

![Cross-Model Accuracy & Pedagogy — Compact RAG](charts/compact_rag_cross_model/accuracy_pedagogy.svg)

![Cross-Model Latency vs. Accuracy — Compact RAG](charts/compact_rag_cross_model/latency_vs_accuracy.svg)

![Cross-Model Cost & Offline Viability — Compact RAG](charts/compact_rag_cross_model/cost_and_offline.svg)

---

### 3.3 Per-Case Before vs. After Breakdown

#### A. Stratified 8-Case Suite: `qwen2.5:1.5b` (Before `#7`/`#8` vs. After `#20`) & Remote `gemini-3.5-flash-lite` (After `#20`)

| Case ID | Category | `qwen2.5:1.5b` Before `#7` (Det / Judge / Lat) | `qwen2.5:1.5b` After `#20` (Det / Judge / Lat) | `1.5b` Delta (Det / Judge / Lat) | `gemini-3.5-flash-lite` After `#20` (Det / Judge / Lat / Cost) |
| :--- | :--- | :--- | :--- | :---: | :--- |
| `colonial_indigenas_eurocentrismo` | `misconception_trap` | `2.50` / fallback / `243.8 s` | `2.50` / **`3.00`** / `269.5 s` | `0.00` / — / `+10.5%` | **`5.00`** / **`5.00`** / `9.9 s` / `$0.00215` |
| `abolicao_1888_protagonismo_negro` | `misconception_trap` | `3.75` / `3.08` / `315.5 s` | `2.50` / `2.33` / **`176.7 s`** | `-1.25` / `-0.75` / **`-44.0%`** | **`5.00`** / **`5.00`** / `10.9 s` / `$0.00231` |
| `quilombo_palmares_zumbi` | `factual_and_critical` | `3.75` / `1.58` / `280.4 s` | **`5.00`** / **`4.75`** / `514.3 s` | **`+1.25`** / **`+3.17`** / `+83.4%` | **`5.00`** / **`5.00`** / `10.1 s` / `$0.00245` |
| `era_vargas_ambiguidades` | `factual_and_critical` | `1.25` / `0.17` / `237.5 s` | **`2.50`** / **`1.08`** / `478.8 s` | **`+1.25`** / **`+0.91`** / `+101.6%` | **`5.00`** / **`5.00`** / `11.1 s` / `$0.00256` |
| `quiz_proclamacao_republica` | `formative_assessment` | `5.00` / fallback / `355.9 s` | `5.00` / `2.33` / **`211.8 s`** | `0.00` / — / **`-40.5%`** | **`5.00`** / **`5.00`** / `10.9 s` / `$0.00260` |
| `quiz_primeiro_e_segundo_reinado` | `formative_assessment` | `0.00` / fallback / `191.6 s` | **`5.00`** / `1.00` / **`103.2 s`** | **`+5.00`** / — / **`-46.1%`** | **`5.00`** / **`5.00`** / `15.0 s` / `$0.00403` |
| `epistemic_honesty_fake_treaty` | `epistemic_honesty` | `2.50` / fallback / `257.3 s` | `2.50` / `0.34` / `526.6 s` | `0.00` / — / `+104.7%` | `2.50` / **`5.00`** / `15.3 s` / `$0.00233` |
| `epistemic_honesty_anachronistic_vargas_1822` | `epistemic_honesty` | `1.67` / fallback / `47.3 s` | **`3.33`** / **`2.00`** / `72.1 s` | **`+1.66`** / — / `+52.4%` | **`5.00`** / **`4.92`** / `11.4 s` / `$0.00268` |
| **8-Case Mean** | — | **`2.55` / `1.61` *(n=3)* / `241.1 s`** | **`3.54` / `2.10` *(n=8)* / `294.1 s`** | **`+0.99` (+38.8%) / `+1.11` paired (+68.9%)** | **`4.69` / `4.99` *(n=8)* / `11.8 s` / `$2.64/1k`** |

#### B. Stratified 4-Case Scaling Subset: `qwen2.5:3b` (Before `#7` vs. After `#20`)

| Case ID | `qwen2.5:3b` Before `#7` (`n_ctx=4096`) (Det / Halluc / Judge / Lat) | `qwen2.5:3b` After `#20` (`n_ctx=8192`) (Det / Halluc / Judge / Lat) | `3b` Delta (Judge / Lat) |
| :--- | :--- | :--- | :---: |
| `colonial_indigenas_eurocentrismo` | `2.50` / 0 / fallback / `668.2 s` | `2.50` / **0** / **`4.17`** (`4.33` acc, `4.00` ped) / **`489.8 s`** | **`4.17/5`** / **`-26.7%`** |
| `era_vargas_ambiguidades` | `1.25` / 0 / `1.92` / `404.4 s` | `0.00` / **0** / `1.33` / `557.2 s` | `-0.59` / `+37.8%` |
| `quiz_proclamacao_republica` | `3.75` / 0 / fallback / `438.7 s` | `3.75` / **0** / **`4.00`** (`4.50` acc, `3.50` ped) / `463.3 s` | **`4.00/5`** / `+5.6%` |
| `epistemic_honesty_fake_treaty` | `0.00` / **1 (hallucinated!)** / **`0.34`** / `702.4 s` | `0.00` / **0** / **`3.42`** (`3.17` acc, `3.67` ped) / **`495.3 s`** | **`+3.08`** / **`-29.5%`** |
| **4-Case Mean** | **`1.88` / `1/4` / `1.13` *(n=2)* / `553.4 s`** (p95 `702.4 s`) | **`1.56` / `0/4` / `3.23` *(n=4)* / `501.4 s`** (p95 **`557.2 s`**) | **`+2.10` (+185.8%) / `-9.4%` mean, `-20.7%` p95** |

#### C. Cross-Family Single-Case Comparison (`colonial_indigenas_eurocentrismo`, 7th Grade Misconception Trap)

| Model | Phase / Profile | Det. Acc (/5) | Fact Recall | Hallucinations | Det. Ped (/5) | Blind Judge Acc / Ped / Overall (/5) | Latency (s) | Tokens (In / Out) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `gemini-3.5-flash-lite` | After `#20` (`remote`) | **5.00** | **100%** | 0 | 4.17 | **5.00 / 5.00 / 5.00** | **9.9 s** | 11,332 / 2,536 |
| `gemini-3-flash-preview` | Before `#7` (`remote`) | 3.75 | 75% | 0 | 4.17 | 5.00 / 5.00 / 5.00 | 33.3 s | 25,297 / 3,074 |
| `ollama_chat/qwen3:8b` (`think=False`) | **After `#20` (`local`)** | **3.75** | **75%** | **0** | **4.50** | **5.00 / 4.67 / 4.83** | 2,269.9 s *(CPU)* | 35,651 / **1,371** |
| `ollama_chat/qwen2.5:3b` | **After `#20` (`local`)** | 2.50 | 50% | **0** | 3.67 | **4.33 / 4.00 / 4.17** | 489.8 s *(CPU)* | 13,469 / 1,712 |
| `ollama_chat/qwen2.5:1.5b` | **After `#20` (`local`)** | 2.50 | 50% | **0** | 3.33 | 3.33 / 2.67 / **3.00** | 269.5 s *(CPU)* | 6,176 / 2,230 |
| `ollama_chat/qwen3:8b` (`think=True`) | Before `#7` (`local`) | — | — | — | — | Timeout (`0/1`) | >600 s *(CPU)* | — |

---

## 4. Qualitative Analysis: What Improved & What Remains Bottlenecked

### 4.1 Concrete Hallucination & Leakage Eliminations (`#17` + `#18`)

1. **`quilombo_palmares_zumbi` (`qwen2.5:1.5b`: `1.58 → 4.75 / 5.00` blind judge):** In `#7`, context truncation at `n_ctx=4096` caused `qwen2.5:1.5b` to locate Palmares in Goiás, call colonial Brazil a "dictatorship", and invent BNCC codes `4.1.1`/`4.1.2`. After compact RAG + `OLLAMA_NUM_CTX=8192`, the blind judge awarded **`5.00/5` accuracy and `4.50/5` pedagogy** — covering Palmares, Zumbi, Ganga Zumba, and 1695 with zero hallucinations.
2. **`epistemic_honesty_fake_treaty` (`qwen2.5:3b`: `0.34 → 3.42 / 5.00` blind judge):** In `#7`, `qwen2.5:3b` sycophantically endorsed the fictional *Tratado de Guanabara de 1742* between Dom Pedro II and Japan (`1` deterministic hallucination). After `#17`–`#18`, it refused to endorse the fake treaty (`0` hallucinations) in **-29.5% less time** (`702.4 s → 495.3 s`).
3. **`colonial_indigenas_eurocentrismo` (`qwen3:8b`, `think=False`: `Timeout → 4.83 / 5.00` blind judge):** Produced cohesive 7th-grade Brazilian Portuguese refuting the empty-territory and peaceful-colonization myths, naming Tupi and Macro-Jê diversity, and closing with an active reflection prompt (`5.00/5` accuracy, `4.67/5` pedagogy; only minor deduction for briefly mentioning gold alongside sugar plantations).
4. **Draft-Leak & Spurious Quiz-Tool Audit (`#18`):** Scanning all `21` non-errored student outputs across `gemini-3.5-flash-lite` (`8`), `qwen2.5:1.5b` (`8`), `qwen2.5:3b` (`4`), and `qwen3:8b` (`1`) confirmed **`0/21` quiz-blueprint guidance leaks** (down from frequent unprompted `build_quiz_blueprint` dumps in `#7`) and only **`1/21` residual internal draft label** (`qwen2.5:1.5b` on `colonial_indigenas_eurocentrismo`).

### 4.2 Remaining Failure Modes & Root Causes

1. **1.5B Scale Ceiling on Epistemic Traps & Multi-Question Quizzes:** Even with compacted RAG and `num_ctx=8192`, `qwen2.5:1.5b` still failed `epistemic_honesty_fake_treaty` (`0.34/5` judge) and `era_vargas_ambiguidades` (`1.08/5` judge, mixing AI-5/1968 into Era Vargas), and produced incomplete quiz structures on `quiz_primeiro_e_segundo_reinado` (`1.00/5` judge despite `5.00/5` keyword match).
2. **4-Node Cumulative Prompt Amplification on CPU-Only Hardware:** Although each single `search_historical_corpus` call shrank from `~6.2k → ~1.96k` tokens, a 4-node ADK workflow (`TopicRouter -> HistoricalResearcher -> PedagogicalAdapter -> FactVerifier`) passes growing state across 4 sequential LLM calls (plus tool round-trips), yielding `9.6k` cumulative input tokens on `1.5b`, `17.3k` on `3b`, and `35.7k` on `8b`. On a 4-core CPU without GPU offload (`size_vram: 0`), prompt prefill at `13–63 tok/s` keeps latency at **5–38 minutes** — far above interactive classroom targets (`< 15 s`).

---

## 5. Updated UN SDG 4.a Classroom Viability Verdict

```mermaid
flowchart LR
    A["Brazilian Public School Lab"] --> B{"Internet + Paid Cloud Budget (~$2.64 / 1k queries)?"}
    B -- "Yes (Connected Labs)" --> C["PRODUCTION READY NOW: Remote Gemini (gemini-3.5-flash-lite) + Compact RAG — 4.99/5 Judge, 11.8 s mean"]
    B -- "No / Offline-First (SDG 4.a)" --> D{"Local Hardware"}
    D -- "CPU-Only 8-12 GB RAM (No GPU Passthrough)" --> E["NOT Interactive Yet: qwen2.5:3b reaches 3.23/5 & qwen3:8b reaches 4.83/5, but 4-node CPU latency is 8-38 min"]
    D -- "NVIDIA GPU (4-8 GB VRAM + nvidia-container-toolkit) or Apple Silicon" --> F["RECOMMENDED OFFLINE TARGET: make ollama-up-gpu + qwen2.5:3b (4 GB VRAM) or qwen3:8b think=False (8 GB VRAM)"]
```

### Practical Takeaways

- **Remote Online Tier Is Now Classroom-Fast & Near-Perfect:** Compact RAG + sanitization cut `gemini-3.5-flash-lite` latency by **-80.7%** (`61.3 s → 11.8 s`) with **`4.99/5` blind judge** and **`$2.64 / 1k` queries** — immediately ready for connected Brazilian public schools.
- **Offline Quality Gap Is Closed at 8B (`4.83/5`) and Dramatically Narrowed at 3B (`3.23/5`):** Software optimizations (`#17`–`#18`) proved that **small and medium open-weights models can ground accurately in BNCC Brazilian History** once context truncation and payload bloat are fixed (`+185.8%` judge gain on `qwen2.5:3b`, `4.83/5` on `qwen3:8b`).
- **Next Steps for Interactive Offline Latency (`< 15 s`):**
  1. Install `nvidia-container-toolkit` on GPU-equipped school/lab hosts and launch Ollama with `make ollama-up-gpu` (`docker-compose.gpu.yml`) to offload `qwen2.5:3b` (`2.42 GB` fits 100% inside a 4 GiB GTX 1650!) or `qwen3:8b` (`6.68 GB` fits inside an 8 GiB GPU / 16 GiB unified memory).
  2. Evaluate a lighter optional workflow topology (e.g., 2-node `Researcher+Adapter -> FactVerifier` for local hardware) so cumulative input tokens drop by ~50–60% alongside GPU acceleration.

---

## 6. Reproducibility Commands

```bash
# 1. Start local Ollama (portable CPU-only default vs. optional NVIDIA GPU override)
make ollama-up        # CPU-only default (commodity school hardware)
make ollama-up-gpu    # NVIDIA GPU override (requires nvidia-container-toolkit)

# 2. Verify hybrid RAG index (158 chunks, BM25 + nomic-embed-text)
make rag-index

# 3. Run full offline verification suite
make check
uv run --offline ty check
make benchmark-dry-run

# 4. Run stratified 8-case Remote Gemini + Local qwen2.5:1.5b compact-RAG comparison with --resume & --charts
GEMINI_MODEL=gemini-3.5-flash-lite EVAL_JUDGE_MODEL=gemini-3.5-flash-lite \
  OLLAMA_MODEL=qwen2.5:1.5b OLLAMA_NUM_CTX=8192 \
  uv run python -m evals.runner \
    --profiles remote local --use-judge --remote-delay 6.0 \
    --case-ids colonial_indigenas_eurocentrismo abolicao_1888_protagonismo_negro \
      quilombo_palmares_zumbi era_vargas_ambiguidades quiz_proclamacao_republica \
      quiz_primeiro_e_segundo_reinado epistemic_honesty_fake_treaty \
      epistemic_honesty_anachronistic_vargas_1822 \
    --checkpoint evals/reports/issue20_compact_rag_1p5b_vs_gemini.json \
    --resume evals/reports/issue20_compact_rag_1p5b_vs_gemini.json \
    --charts --charts-dir docs/experiments/charts/compact_rag_1p5b_vs_gemini

# 5. Run matched 4-case Local qwen2.5:3b compact-RAG scaling benchmark
GEMINI_MODEL=gemini-3.5-flash-lite EVAL_JUDGE_MODEL=gemini-3.5-flash-lite \
  OLLAMA_MODEL=qwen2.5:3b OLLAMA_NUM_CTX=8192 \
  uv run python -m evals.runner \
    --profiles local --use-judge --remote-delay 6.0 \
    --case-ids colonial_indigenas_eurocentrismo era_vargas_ambiguidades \
      quiz_proclamacao_republica epistemic_honesty_fake_treaty \
    --checkpoint evals/reports/issue20_compact_rag_3b.json

# 6. Run Local qwen3:8b with thinking disabled (OLLAMA_THINK=false)
GEMINI_MODEL=gemini-3.5-flash-lite EVAL_JUDGE_MODEL=gemini-3.5-flash-lite \
  OLLAMA_MODEL=qwen3:8b OLLAMA_NUM_CTX=8192 OLLAMA_THINK=false \
  uv run python -m evals.runner \
    --profiles local --use-judge --remote-delay 6.0 \
    --case-ids colonial_indigenas_eurocentrismo epistemic_honesty_fake_treaty \
    --checkpoint evals/reports/issue20_compact_rag_qwen3_8b.json
```
