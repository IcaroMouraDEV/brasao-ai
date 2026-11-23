# UN SDG 4 Alignment & Pedagogical Design

## Why UN SDG 4 (Quality Education)?

United Nations **Sustainable Development Goal 4 (ODS 4 — Educação de Qualidade)** aims to *"ensure inclusive and equitable quality education and promote lifelong learning opportunities for all."*

In Brazil, two structural gaps directly motivate this project's dual-model design:

1. **Pedagogical Support Gap (SDG 4.1 & 4.7):** Students preparing for school exams and ENEM often lack personalized, historically rigorous tutoring that connects Brazilian History to citizenship, human rights, and cultural diversity.
2. **Connectivity & Cost Inequality (SDG 4.a):** Thousands of Brazilian public schools and low-income households face unstable internet connectivity or cannot afford recurring cloud LLM API costs.

By systematically benchmarking a **local offline model (Ollama)** against a **remote frontier model (Gemini)**, this project measures the exact trade-off between **accessibility/offline sovereignty** and **historical/pedagogical quality**.

## BNCC Alignment (Base Nacional Comum Curricular)

Curated corpus entries and generated activities map to Brazilian **BNCC** History skills:

- **Ensino Fundamental — Anos Finais (`fundamental_2`, 6th–9th grade):**
  - `EF07HI10` / `EF07HI12`: Colonial Brazil, colonization dynamics, and territorial expansion.
  - `EF08HI11` / `EF08HI12`: Independence processes, First Reign, and Imperial Brazil.
  - `EF08HI19` / `EF08HI20`: Slavery, resistance (quilombos), and Abolition.
  - `EF09HI01` / `EF09HI02`: Proclamation of the Republic and First Republic.
  - `EF09HI06` / `EF09HI19` / `EF09HI20`: Era Vargas, Military Dictatorship, and Redemocratization.
- **Ensino Médio (`ensino_medio`, High School / ENEM):**
  - `EM13CHS101` / `EM13CHS102`: Analysis of historical processes, sources, and historiographical interpretations.
  - `EM13CHS502` / `EM13CHS603`: Citizenship, democracy, human rights, and structural inequalities.

## Pedagogical Pillars of the Agent

1. **Grounded Accuracy:** Prefer verified entries from the curated corpus (`search_historical_corpus`, `lookup_timeline`) before elaborating.
2. **Adaptive Scaffolding:** Adjust vocabulary, depth, and examples according to the student's target level (`fundamental_2`, `ensino_medio`, or `geral`).
3. **Multiple Perspectives (SDG 4.7):** Highlight Indigenous peoples, Afro-Brazilian resistance, women, and regional movements alongside institutional politics.
4. **Anti-Myth & Critical Thinking:** Proactively address common historical oversimplifications (e.g., framing the 1888 *Lei Áurea* solely as a benevolent royal act while omitting decades of abolitionist and Black resistance).
5. **Formative Assessment:** Use `build_quiz_blueprint` to generate structured exercises with explanatory feedback rather than raw answer keys.
