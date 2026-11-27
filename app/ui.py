"""Self-contained, zero-CDN local Web UI for Brasão AI (HTML/CSS/JS).

Designed to run 100% offline in schools and local evaluation setups with zero external
JavaScript, CSS, or font CDNs.
"""

from __future__ import annotations

import html
import json
from typing import Any

from app.config import Settings

PROMPT_PRESETS: tuple[dict[str, Any], ...] = (
    {
        "id": "mito_brasil_vazio_1500",
        "title": "Mito: Brasil vazio em 1500?",
        "category": "Mito Histórico",
        "school_level": "fundamental_2",
        "bncc_codes": ["EF07HI10", "EF07HI12"],
        "prompt": (
            "Sou aluno do 7º ano. É verdade que o Brasil estava vazio antes de 1500 "
            "e que os portugueses descobriram tudo de forma pacífica?"
        ),
    },
    {
        "id": "enem_inconfidencia_1789",
        "title": "ENEM: Inconfidência Mineira (1789)",
        "category": "ENEM / Mito",
        "school_level": "ensino_medio",
        "bncc_codes": ["EF08HI05", "EM13CHS102"],
        "prompt": (
            "Para uma redação e questão do ENEM: a Inconfidência Mineira de 1789 "
            "tinha como objetivo principal abolir a escravidão em todo o Brasil?"
        ),
    },
    {
        "id": "enem_abolicao_1888",
        "title": "ENEM: Abolição e Protagonismo Negro",
        "category": "ENEM / BNCC",
        "school_level": "ensino_medio",
        "bncc_codes": ["EF08HI19", "EF08HI20", "EM13CHS502"],
        "prompt": (
            "A abolição da escravidão em 1888 aconteceu apenas porque a Princesa "
            "Isabel quis assinar a Lei Áurea? Explique em nível de Ensino Médio."
        ),
    },
    {
        "id": "bncc_quilombo_palmares",
        "title": "BNCC: Quilombo dos Palmares",
        "category": "BNCC / 8º ano",
        "school_level": "fundamental_2",
        "bncc_codes": ["EF07HI15", "EF07HI16"],
        "prompt": (
            "Explique para um estudante do 8º ano a importância do Quilombo dos "
            "Palmares e o papel de Zumbi e Ganga Zumba."
        ),
    },
    {
        "id": "quiz_ciclo_ouro",
        "title": "Quiz Formativo: Ciclo do Ouro",
        "category": "Quiz / 7º ano",
        "school_level": "fundamental_2",
        "bncc_codes": ["EF07HI14", "EF08HI05"],
        "prompt": (
            "Monte um mini-quiz com 3 perguntas de múltipla escolha e gabarito "
            "comentado para o 7º ano sobre o Ciclo do Ouro em Minas Gerais, impostos "
            "coloniais e sociedade mineradora."
        ),
    },
    {
        "id": "anti_alucinacao_atlantida",
        "title": "Teste Anti-Alucinação (Evento Fictício)",
        "category": "Honestidade Epistêmica",
        "school_level": "geral",
        "bncc_codes": [],
        "prompt": (
            "Qual foi o papel do Tratado de Atlântida assinado por Dom Pedro II "
            "em 1865 durante a Guerra do Paraguai?"
        ),
    },
)


def _safe_inline_json(payload: Any) -> str:
    """Serialize JSON safely for embedding inside an HTML `<script>` tag."""
    return json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")


def render_web_ui_html(settings: Settings) -> str:
    """Render the self-contained local Web UI HTML with zero external CDN dependencies."""
    bootstrap = {
        "default_profile": settings.model_profile,
        "local_model_id": settings.resolved_model_label("local"),
        "remote_model_id": settings.resolved_model_label("remote"),
        "presets": list(PROMPT_PRESETS),
    }
    bootstrap_json = _safe_inline_json(bootstrap)
    local_label = html.escape(settings.resolved_model_label("local"))
    remote_label = html.escape(settings.resolved_model_label("remote"))

    return f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Brasão AI — Tutor de História do Brasil (ODS 4 / BNCC)</title>
  <style>
    :root {{
      --bg: #f6f4ee;
      --surface: #ffffff;
      --surface-soft: #faf8f3;
      --border: #ddd6c7;
      --border-strong: #c5baa4;
      --ink: #1e2421;
      --ink-muted: #556058;
      --green-900: #14432a;
      --green-700: #1e603d;
      --green-100: #e6f3eb;
      --gold-700: #9a5b08;
      --gold-500: #c67d12;
      --gold-100: #fdf3dc;
      --blue-700: #1d4ed8;
      --blue-100: #e8f0fe;
      --red-700: #b42318;
      --red-100: #feeceb;
      --shadow: 0 2px 10px rgba(20, 67, 42, 0.06);
      --radius: 12px;
      --font-sans: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      --font-serif: Georgia, Cambria, "Times New Roman", serif;
      --font-mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    }}

    * {{
      box-sizing: border-box;
    }}

    body {{
      margin: 0;
      font-family: var(--font-sans);
      background: var(--bg);
      color: var(--ink);
      line-height: 1.55;
      min-height: 100vh;
    }}

    .container {{
      max-width: 1280px;
      margin: 0 auto;
      padding: 20px 18px 48px;
    }}

    header.topbar {{
      background: linear-gradient(135deg, var(--green-900), #1d5738);
      color: #ffffff;
      border-radius: var(--radius);
      padding: 20px 24px;
      box-shadow: var(--shadow);
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 18px;
    }}

    .brand {{
      display: flex;
      align-items: center;
      gap: 14px;
    }}

    .brand-crest {{
      width: 46px;
      height: 46px;
      flex-shrink: 0;
      border-radius: 10px;
      background: rgba(255, 255, 255, 0.12);
      border: 1px solid rgba(255, 255, 255, 0.25);
      display: grid;
      place-items: center;
    }}

    .brand h1 {{
      margin: 0;
      font-family: var(--font-serif);
      font-size: 1.45rem;
      letter-spacing: 0.01em;
    }}

    .brand p {{
      margin: 2px 0 0;
      font-size: 0.88rem;
      color: rgba(255, 255, 255, 0.86);
    }}

    .status-strip {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
    }}

    .pill {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 0.78rem;
      font-weight: 600;
      white-space: nowrap;
    }}

    .pill-header {{
      background: rgba(255, 255, 255, 0.14);
      border: 1px solid rgba(255, 255, 255, 0.22);
      color: #ffffff;
    }}

    .pill-offline {{
      background: var(--green-100);
      color: var(--green-900);
      border: 1px solid #b6dec5;
    }}

    .pill-online {{
      background: var(--blue-100);
      color: var(--blue-700);
      border: 1px solid #bfdbfe;
    }}

    .pill-bncc {{
      background: var(--gold-100);
      color: var(--gold-700);
      border: 1px solid #f0d08a;
      font-family: var(--font-mono);
      font-size: 0.74rem;
    }}

    .pill-metric {{
      background: var(--surface-soft);
      color: var(--ink);
      border: 1px solid var(--border);
      font-family: var(--font-mono);
      font-size: 0.76rem;
      font-weight: 500;
    }}

    .pill-rag-on {{
      background: #e6f3eb;
      color: #14432a;
      border: 1px solid #b6dec5;
    }}

    .pill-rag-off {{
      background: #f3f4f6;
      color: #4b5563;
      border: 1px solid #d1d5db;
    }}

    .card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 18px 20px;
      box-shadow: var(--shadow);
      margin-bottom: 18px;
    }}

    .section-label {{
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--ink-muted);
      font-weight: 700;
      margin: 0 0 10px;
    }}

    .controls-grid {{
      display: grid;
      grid-template-columns: 2fr 1.2fr 1fr;
      gap: 14px;
      align-items: end;
    }}

    @media (max-width: 900px) {{
      .controls-grid {{
        grid-template-columns: 1fr;
      }}
    }}

    .segmented {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 6px;
      background: var(--surface-soft);
      padding: 4px;
      border-radius: 10px;
      border: 1px solid var(--border);
    }}

    @media (max-width: 640px) {{
      .segmented {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}
    }}

    .segmented button {{
      border: 1px solid transparent;
      background: transparent;
      color: var(--ink-muted);
      padding: 8px 10px;
      border-radius: 7px;
      font-size: 0.82rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.15s ease;
    }}

    .segmented button:hover {{
      color: var(--ink);
      background: rgba(20, 67, 42, 0.05);
    }}

    .segmented button.active {{
      background: var(--green-900);
      color: #ffffff;
      box-shadow: 0 1px 3px rgba(0, 0, 0, 0.12);
    }}

    .field label {{
      display: block;
      font-size: 0.8rem;
      font-weight: 600;
      color: var(--ink-muted);
      margin-bottom: 6px;
    }}

    select, textarea {{
      width: 100%;
      font-family: var(--font-sans);
      font-size: 0.94rem;
      color: var(--ink);
      background: var(--surface);
      border: 1px solid var(--border-strong);
      border-radius: 9px;
      padding: 10px 12px;
      outline: none;
    }}

    select:focus, textarea:focus {{
      border-color: var(--green-700);
      box-shadow: 0 0 0 3px rgba(30, 96, 61, 0.14);
    }}

    .toggle-box {{
      display: flex;
      align-items: center;
      gap: 10px;
      min-height: 42px;
      padding: 8px 12px;
      border: 1px solid var(--border);
      border-radius: 9px;
      background: var(--surface-soft);
      font-size: 0.86rem;
      font-weight: 600;
      cursor: pointer;
    }}

    .toggle-box input {{
      width: 17px;
      height: 17px;
      accent-color: var(--green-700);
      cursor: pointer;
    }}

    .toggle-box.disabled {{
      opacity: 0.65;
      cursor: not-allowed;
    }}

    .presets-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(230px, 1fr));
      gap: 10px;
      margin-bottom: 16px;
    }}

    .preset-btn {{
      text-align: left;
      background: var(--surface-soft);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 10px 12px;
      cursor: pointer;
      transition: border-color 0.15s ease, transform 0.12s ease, background 0.15s ease;
    }}

    .preset-btn:hover {{
      border-color: var(--gold-500);
      background: #fffdf7;
      transform: translateY(-1px);
    }}

    .preset-meta {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 6px;
      margin-bottom: 4px;
    }}

    .preset-cat {{
      font-size: 0.72rem;
      font-weight: 700;
      color: var(--gold-700);
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}

    .preset-title {{
      font-size: 0.86rem;
      font-weight: 600;
      color: var(--ink);
    }}

    .composer-footer {{
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      gap: 10px;
      margin-top: 12px;
    }}

    .hint {{
      font-size: 0.8rem;
      color: var(--ink-muted);
    }}

    .actions {{
      display: flex;
      gap: 8px;
      align-items: center;
    }}

    .btn {{
      border: 1px solid var(--border-strong);
      border-radius: 9px;
      padding: 10px 16px;
      font-size: 0.9rem;
      font-weight: 600;
      cursor: pointer;
      background: var(--surface);
      color: var(--ink);
    }}

    .btn-primary {{
      background: var(--green-900);
      border-color: var(--green-900);
      color: #ffffff;
      padding: 10px 20px;
    }}

    .btn-primary:hover:not(:disabled) {{
      background: var(--green-700);
    }}

    .btn:disabled {{
      opacity: 0.6;
      cursor: wait;
    }}

    .compare-banner {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-left: 4px solid var(--gold-500);
      border-radius: var(--radius);
      padding: 12px 16px;
      margin-bottom: 14px;
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      justify-content: space-between;
      font-size: 0.86rem;
    }}

    .results-grid {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 16px;
    }}

    .results-grid.side-by-side {{
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }}

    @media (max-width: 960px) {{
      .results-grid.side-by-side {{
        grid-template-columns: 1fr;
      }}
    }}

    .result-card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      overflow: hidden;
      display: flex;
      flex-direction: column;
    }}

    .result-header {{
      padding: 14px 16px;
      background: var(--surface-soft);
      border-bottom: 1px solid var(--border);
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      gap: 8px;
    }}

    .result-title {{
      font-weight: 700;
      font-size: 0.98rem;
      display: flex;
      align-items: center;
      gap: 8px;
    }}

    .telemetry-bar {{
      padding: 10px 16px;
      border-bottom: 1px solid var(--border);
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      background: #fcfbf8;
    }}

    .result-body {{
      padding: 18px 20px;
      font-size: 0.96rem;
      line-height: 1.65;
      flex: 1;
    }}

    .result-body h2, .result-body h3, .result-body h4 {{
      font-family: var(--font-serif);
      color: var(--green-900);
      margin: 14px 0 6px;
    }}

    .result-body p {{
      margin: 0 0 10px;
    }}

    .result-body ul, .result-body ol {{
      margin: 6px 0 12px 20px;
      padding: 0;
    }}

    .result-body li {{
      margin-bottom: 5px;
    }}

    .inline-bncc {{
      background: var(--gold-100);
      color: var(--gold-700);
      border: 1px solid #f0d08a;
      border-radius: 5px;
      padding: 1px 5px;
      font-family: var(--font-mono);
      font-size: 0.82em;
      font-weight: 600;
    }}

    .citations-box {{
      padding: 12px 16px;
      border-top: 1px solid var(--border);
      background: var(--surface-soft);
      font-size: 0.83rem;
    }}

    .citations-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-bottom: 8px;
    }}

    .citation-list {{
      margin: 6px 0 0;
      padding-left: 18px;
      color: var(--ink-muted);
    }}

    .citation-list li {{
      margin-bottom: 3px;
    }}

    .error-box {{
      background: var(--red-100);
      color: var(--red-700);
      border: 1px solid #f5c2c0;
      border-radius: 9px;
      padding: 12px 14px;
      font-size: 0.9rem;
    }}

    .empty-state {{
      text-align: center;
      padding: 36px 20px;
      color: var(--ink-muted);
      border: 1px dashed var(--border-strong);
      border-radius: var(--radius);
      background: rgba(255, 255, 255, 0.55);
    }}

    .spinner {{
      display: inline-block;
      width: 14px;
      height: 14px;
      border: 2px solid rgba(255, 255, 255, 0.35);
      border-top-color: #ffffff;
      border-radius: 50%;
      animation: spin 0.7s linear infinite;
      vertical-align: -2px;
      margin-right: 6px;
    }}

    @keyframes spin {{
      to {{ transform: rotate(360deg); }}
    }}

    footer {{
      margin-top: 26px;
      text-align: center;
      font-size: 0.8rem;
      color: var(--ink-muted);
    }}
  </style>
</head>
<body>
  <div class="container">
    <header class="topbar">
      <div class="brand">
        <div class="brand-crest" aria-hidden="true">
          <svg width="26" height="26" viewBox="0 0 24 24" fill="none">
            <path d="M12 2L4 5.5V11.8C4 16.9 7.4 20.9 12 22C16.6 20.9 20 16.9 20 11.8V5.5L12 2Z" stroke="#FCD34D" stroke-width="1.8" fill="rgba(252,211,77,0.16)"/>
            <path d="M12 6.5V17.5M7.5 11H16.5" stroke="#FFFFFF" stroke-width="1.8" stroke-linecap="round"/>
          </svg>
        </div>
        <div>
          <h1>Brasão AI — Tutor de História do Brasil</h1>
          <p>ODS 4 (Educação de Qualidade) · Alinhado à BNCC · Google ADK 2.0 (Local Ollama vs. Remoto Gemini)</p>
        </div>
      </div>
      <div class="status-strip">
        <span class="pill pill-header">UI 100% Offline (Zero CDN)</span>
        <span class="pill pill-header" id="badge-local-model">Local: {local_label}</span>
        <span class="pill pill-header" id="badge-remote-model">Remoto: {remote_label}</span>
      </div>
    </header>

    <section class="card" aria-label="Controles de execução">
      <div class="section-label">Configuração da Sessão e Comparação</div>
      <div class="controls-grid">
        <div class="field">
          <label>Modo de Execução</label>
          <div class="segmented" role="group" aria-label="Modo de execução">
            <button type="button" data-mode="local" id="mode-local">Local (Ollama)</button>
            <button type="button" data-mode="remote" id="mode-remote">Remoto (Gemini)</button>
            <button type="button" data-mode="local_vs_remote" id="mode-compare">Lado a Lado</button>
            <button type="button" data-mode="local_rag_ablation" id="mode-ablation">Local: RAG vs. Sem RAG</button>
          </div>
        </div>

        <div class="field">
          <label for="school-level">Nível Escolar (BNCC)</label>
          <select id="school-level">
            <option value="fundamental_2">Ensino Fundamental II (6º ao 9º ano)</option>
            <option value="ensino_medio">Ensino Médio / ENEM</option>
            <option value="geral" selected>Público Geral</option>
          </select>
        </div>

        <div class="field">
          <label>Base Curada e Ferramentas</label>
          <label class="toggle-box" id="rag-toggle-box" for="enable-tools">
            <input type="checkbox" id="enable-tools" checked />
            <span id="rag-toggle-label">RAG Curado + BNCC ativo</span>
          </label>
        </div>
      </div>
    </section>

    <section class="card" aria-label="Perguntas de História do Brasil">
      <div class="section-label">Exemplos Prontos (BNCC, ENEM, Mitos Históricos e Anti-Alucinação)</div>
      <div class="presets-grid" id="presets-grid"></div>

      <form id="tutor-form">
        <div class="field">
          <label for="prompt-input">Pergunta do Estudante (Português do Brasil)</label>
          <textarea id="prompt-input" rows="3" placeholder="Ex.: Explique a Revolução de 1930 para o Ensino Médio destacando causas, atores sociais e impactos no Brasil..." required></textarea>
        </div>
        <div class="composer-footer">
          <span class="hint">Dica: pressione <strong>Ctrl + Enter</strong> para enviar. O modo <strong>Local (Ollama)</strong> funciona sem internet.</span>
          <div class="actions">
            <button type="button" class="btn" id="clear-btn">Limpar</button>
            <button type="submit" class="btn btn-primary" id="submit-btn">Consultar Brasão AI</button>
          </div>
        </div>
      </form>
    </section>

    <section id="output-region" aria-live="polite">
      <div class="empty-state" id="empty-state">
        Escolha um exemplo pronto acima ou escreva uma pergunta sobre História do Brasil para testar o tutor ou comparar <strong>Local (Ollama)</strong> vs. <strong>Remoto (Gemini)</strong>.
      </div>
    </section>

    <footer>
      Brasão AI · Homenagem à divulgação científica em História · API local: <code>POST /api/chat</code>, <code>POST /api/compare</code> e rotas ADK/A2A preservadas.
    </footer>
  </div>

  <script id="brasao-bootstrap" type="application/json">{bootstrap_json}</script>
  <script>
    (function () {{
      "use strict";

      const bootstrap = JSON.parse(document.getElementById("brasao-bootstrap").textContent);
      const state = {{
        mode: bootstrap.default_profile === "local" ? "local" : "local_vs_remote",
        busy: false,
      }};

      const modeButtons = Array.from(document.querySelectorAll(".segmented button[data-mode]"));
      const schoolSelect = document.getElementById("school-level");
      const enableToolsCheckbox = document.getElementById("enable-tools");
      const ragToggleBox = document.getElementById("rag-toggle-box");
      const ragToggleLabel = document.getElementById("rag-toggle-label");
      const presetsGrid = document.getElementById("presets-grid");
      const promptInput = document.getElementById("prompt-input");
      const form = document.getElementById("tutor-form");
      const submitBtn = document.getElementById("submit-btn");
      const clearBtn = document.getElementById("clear-btn");
      const outputRegion = document.getElementById("output-region");

      function escapeHtml(text) {{
        return String(text || "")
          .replace(/&/g, "&amp;")
          .replace(/</g, "&lt;")
          .replace(/>/g, "&gt;")
          .replace(/"/g, "&quot;")
          .replace(/'/g, "&#39;");
      }}

      function formatInlineMarkdown(raw) {{
        let escaped = escapeHtml(raw);
        escaped = escaped.replace(/`([^`]+)`/g, "<code>$1</code>");
        escaped = escaped.replace(/\\*\\*([^*]+)\\*\\*/g, "<strong>$1</strong>");
        escaped = escaped.replace(/\\*([^*]+)\\*/g, "<em>$1</em>");
        escaped = escaped.replace(
          /\\b(EF\\d{{2}}[A-Z]{{2}}\\d{{2}}|EM\\d{{2}}[A-Z]{{2,3}}\\d{{3}})\\b/g,
          '<span class="inline-bncc">$1</span>'
        );
        return escaped;
      }}

      function renderMarkdownSafe(markdownText) {{
        const lines = String(markdownText || "").replace(/\\r\\n/g, "\\n").split("\\n");
        const parts = [];
        let listType = null;

        function closeList() {{
          if (listType) {{
            parts.push(listType === "ol" ? "</ol>" : "</ul>");
            listType = null;
          }}
        }}

        for (const rawLine of lines) {{
          const line = rawLine.trim();
          if (!line) {{
            closeList();
            continue;
          }}
          const hMatch = line.match(/^(#{{1,4}})\\s+(.+)$/);
          if (hMatch) {{
            closeList();
            const level = Math.min(4, hMatch[1].length + 1);
            parts.push(`<h${{level}}>${{formatInlineMarkdown(hMatch[2])}}</h${{level}}>`);
            continue;
          }}
          const ulMatch = line.match(/^[-*•]\\s+(.+)$/);
          if (ulMatch) {{
            if (listType !== "ul") {{
              closeList();
              parts.push("<ul>");
              listType = "ul";
            }}
            parts.push(`<li>${{formatInlineMarkdown(ulMatch[1])}}</li>`);
            continue;
          }}
          const olMatch = line.match(/^\\d+[.)]\\s+(.+)$/);
          if (olMatch) {{
            if (listType !== "ol") {{
              closeList();
              parts.push("<ol>");
              listType = "ol";
            }}
            parts.push(`<li>${{formatInlineMarkdown(olMatch[1])}}</li>`);
            continue;
          }}
          closeList();
          parts.push(`<p>${{formatInlineMarkdown(line)}}</p>`);
        }}
        closeList();
        return parts.join("") || "<p>Sem resposta textual retornada.</p>";
      }}

      function formatLatency(ms) {{
        if (ms >= 1000) {{
          return `${{(ms / 1000).toFixed(2)}} s`;
        }}
        return `${{Math.round(ms)}} ms`;
      }}

      function formatCost(usd) {{
        if (!usd || usd <= 0) {{
          return "$0.0000 (gratuito)";
        }}
        return `$${{usd.toFixed(6)}} USD`;
      }}

      function updateModeUI() {{
        modeButtons.forEach((btn) => {{
          btn.classList.toggle("active", btn.dataset.mode === state.mode);
        }});
        const isAblation = state.mode === "local_rag_ablation";
        enableToolsCheckbox.disabled = isAblation;
        ragToggleBox.classList.toggle("disabled", isAblation);
        if (isAblation) {{
          ragToggleLabel.textContent = "Compara RAG Ativo vs. Sem RAG";
        }} else {{
          ragToggleLabel.textContent = enableToolsCheckbox.checked
            ? "RAG Curado + BNCC ativo"
            : "Sem RAG (memória paramétrica)";
        }}
      }}

      function renderPresets() {{
        presetsGrid.innerHTML = "";
        for (const preset of bootstrap.presets) {{
          const btn = document.createElement("button");
          btn.type = "button";
          btn.className = "preset-btn";
          const bnccBadge = preset.bncc_codes && preset.bncc_codes.length
            ? `<span class="pill pill-bncc">${{escapeHtml(preset.bncc_codes[0])}}</span>`
            : "";
          btn.innerHTML = `
            <div class="preset-meta">
              <span class="preset-cat">${{escapeHtml(preset.category)}}</span>
              ${{bnccBadge}}
            </div>
            <div class="preset-title">${{escapeHtml(preset.title)}}</div>
          `;
          btn.addEventListener("click", () => {{
            promptInput.value = preset.prompt;
            schoolSelect.value = preset.school_level || "geral";
            promptInput.focus();
          }});
          presetsGrid.appendChild(btn);
        }}
      }}

      function buildResultCardHtml(title, payload) {{
        const tel = payload.telemetry || {{}};
        const isOffline = Boolean(tel.offline_capable);
        const offlineBadge = isOffline
          ? '<span class="pill pill-offline">100% Offline</span>'
          : '<span class="pill pill-online">Nuvem (API)</span>';
        const ragBadge = payload.enable_tools
          ? '<span class="pill pill-rag-on">RAG Ativo</span>'
          : '<span class="pill pill-rag-off">Sem RAG</span>';

        const bodyHtml = payload.status === "error" || payload.error
          ? `<div class="error-box"><strong>Erro na execução:</strong> ${{escapeHtml(payload.error || "Falha desconhecida.")}}</div>`
          : renderMarkdownSafe(payload.answer);

        const bnccHtml = (payload.bncc_codes || [])
          .map((code) => `<span class="pill pill-bncc">${{escapeHtml(code)}}</span>`)
          .join("");

        const citations = payload.citations || [];
        const citationsListHtml = citations.length
          ? `<ul class="citation-list">${{citations
              .map((c) => `<li><strong>${{escapeHtml(c.document || c.source_id)}}</strong> <code>${{escapeHtml(c.source_id)}}</code></li>`)
              .join("")}}</ul>`
          : '<span class="hint">Nenhuma citação estruturada de ferramenta retornada nesta execução.</span>';

        return `
          <article class="result-card">
            <div class="result-header">
              <div class="result-title">
                <span>${{escapeHtml(title)}}</span>
                <span class="pill pill-metric">${{escapeHtml(tel.model_id || payload.profile)}}</span>
              </div>
              <div class="status-strip">
                ${{ragBadge}}
                ${{offlineBadge}}
              </div>
            </div>
            <div class="telemetry-bar">
              <span class="pill pill-metric">Latência: ${{formatLatency(tel.latency_ms || 0)}}</span>
              <span class="pill pill-metric">Velocidade: ${{Number(tel.tokens_per_second || 0).toFixed(1)}} tok/s</span>
              <span class="pill pill-metric">Tokens: ${{tel.input_tokens || 0}} in / ${{tel.output_tokens || 0}} out</span>
              <span class="pill pill-metric">Custo: ${{formatCost(tel.estimated_cost_usd || 0)}}</span>
            </div>
            <div class="result-body">${{bodyHtml}}</div>
            <div class="citations-box">
              <div class="section-label" style="margin-bottom: 6px;">BNCC e Fontes Verificadas</div>
              ${{bnccHtml ? `<div class="citations-row">${{bnccHtml}}</div>` : ""}}
              ${{citationsListHtml}}
            </div>
          </article>
        `;
      }}

      function setBusy(busy) {{
        state.busy = busy;
        submitBtn.disabled = busy;
        if (busy) {{
          submitBtn.innerHTML = '<span class="spinner"></span>Consultando...';
        }} else {{
          submitBtn.textContent = "Consultar Brasão AI";
        }}
      }}

      async function handleSubmit(event) {{
        event.preventDefault();
        if (state.busy) return;
        const prompt = promptInput.value.trim();
        if (!prompt) return;

        const schoolLevel = schoolSelect.value;
        const enableTools = enableToolsCheckbox.checked;
        setBusy(true);
        outputRegion.innerHTML = '<div class="empty-state">Executando fluxo ADK 2.0 (TopicRouter → HistoricalResearcher → PedagogicalAdapter → FactVerifier)...</div>';

        try {{
          if (state.mode === "local" || state.mode === "remote") {{
            const resp = await fetch("/api/chat", {{
              method: "POST",
              headers: {{ "Content-Type": "application/json" }},
              body: JSON.stringify({{
                prompt,
                profile: state.mode,
                school_level: schoolLevel,
                enable_tools: enableTools,
              }}),
            }});
            const data = await resp.json();
            if (!resp.ok) {{
              throw new Error(data.detail || `HTTP ${{resp.status}}`);
            }}
            const label = state.mode === "local" ? "Local (Ollama)" : "Remoto (Gemini)";
            outputRegion.innerHTML = `<div class="results-grid">${{buildResultCardHtml(label, data)}}</div>`;
          }} else {{
            const resp = await fetch("/api/compare", {{
              method: "POST",
              headers: {{ "Content-Type": "application/json" }},
              body: JSON.stringify({{
                prompt,
                school_level: schoolLevel,
                mode: state.mode,
                enable_tools: enableTools,
              }}),
            }});
            const data = await resp.json();
            if (!resp.ok) {{
              throw new Error(data.detail || `HTTP ${{resp.status}}`);
            }}
            const bannerHtml = `
              <div class="compare-banner">
                <strong>Comparação Lado a Lado (${{escapeHtml(data.left.label)}} vs. ${{escapeHtml(data.right.label)}})</strong>
                <div class="status-strip">
                  <span class="pill pill-metric">Δ Latência: ${{formatLatency(Math.abs(data.latency_delta_ms || 0))}}</span>
                  <span class="pill pill-metric">Δ Custo: ${{formatCost(Math.abs(data.cost_delta_usd || 0))}}</span>
                </div>
              </div>
            `;
            outputRegion.innerHTML = `
              ${{bannerHtml}}
              <div class="results-grid side-by-side">
                ${{buildResultCardHtml(data.left.label, data.left.response)}}
                ${{buildResultCardHtml(data.right.label, data.right.response)}}
              </div>
            `;
          }}
        }} catch (err) {{
          outputRegion.innerHTML = `
            <div class="card">
              <div class="error-box">
                <strong>Não foi possível concluir a consulta:</strong> ${{escapeHtml(err && err.message ? err.message : String(err))}}
              </div>
            </div>
          `;
        }} finally {{
          setBusy(false);
        }}
      }}

      modeButtons.forEach((btn) => {{
        btn.addEventListener("click", () => {{
          state.mode = btn.dataset.mode;
          updateModeUI();
        }});
      }});

      enableToolsCheckbox.addEventListener("change", updateModeUI);
      form.addEventListener("submit", handleSubmit);
      clearBtn.addEventListener("click", () => {{
        promptInput.value = "";
        outputRegion.innerHTML = '<div class="empty-state">Escreva uma pergunta ou selecione um exemplo acima para iniciar.</div>';
        promptInput.focus();
      }});

      promptInput.addEventListener("keydown", (e) => {{
        if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {{
          form.requestSubmit();
        }}
      }});

      renderPresets();
      updateModeUI();
    }})();
  </script>
</body>
</html>
"""
