"""Deterministic, profile-agnostic output sanitization and quiz tool guards.

Applied identically across `local` (Ollama) and `remote` (Gemini) profiles to prevent
leaked internal workflow headers, routing metadata, tool status blocks, or raw quiz
blueprint guidance from reaching students, and to stop unprompted quiz tool calls on
non-quiz turns.
"""

from __future__ import annotations

import re
import unicodedata


def _fold_text(text: str) -> str:
    """Lowercase and strip diacritics for accent-tolerant matching."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).lower()


_INTENT_LINE_RE = re.compile(
    r"^\s*(?:[-*•]\s*)?(?:\*\*|__|`)?(?:intent|intencao)(?:\*\*|__|`)?\s*[:=]\s*"
    r"(?:\*\*|__|`)?\s*"
    r"(explanation|timeline|quiz|out_of_scope|explicacao|linha do tempo|fora de escopo)\b",
    re.IGNORECASE,
)

_CANONICAL_INTENT_MAP: dict[str, str] = {
    "explanation": "explanation",
    "explicacao": "explanation",
    "timeline": "timeline",
    "linha do tempo": "timeline",
    "quiz": "quiz",
    "out_of_scope": "out_of_scope",
    "fora de escopo": "out_of_scope",
}

_EXPLICIT_QUIZ_NOUN_RE = re.compile(
    r"\b(?:mini[- ]?quiz|quiz|simulado|exercicios?|gabarito|"
    r"multipla escolha|revisao avaliativa)\b"
)

_EXPLICIT_QUIZ_REQUEST_RE = re.compile(
    r"\b(?:crie|criar|monte|montar|elabore|elaborar|faca|fazer|"
    r"prepare|preparar|gere|gerar|proponha|propor|escreva|escrever|"
    r"quero|preciso de|me de)\b"
    r"[^.!?\n]{0,50}\b"
    r"(?:questao|questoes|pergunta|perguntas|exercicio|exercicios|"
    r"atividade|atividades|quiz|simulado)\b"
)

# Exact internal section labels (folded, punctuation/markdown stripped) that should
# never appear as standalone headings in student-facing output.
_INTERNAL_HEADING_LABELS: frozenset[str] = frozenset(
    {
        "rascunho",
        "rascunho didatico",
        "rascunho pedagogico",
        "rascunho pedagogico a revisar",
        "rascunho provisorio",
        "dossie factual",
        "dossie historico",
        "plano de roteamento",
        "verificacao factual",
        "auditoria factual",
        "auditoria final",
        "checklist de auditoria",
        "resposta final verificada",
        "resposta verificada",
        "format guidelines",
        "diretrizes do blueprint",
        "diretrizes de formatacao",
        "diretrizes de formatacao do quiz",
        "quiz blueprint",
        "roteiro de quiz",
        "topicrouter",
        "historicalresearcher",
        "pedagogicaladapter",
        "factverifier",
    }
)

# Inline internal prefixes that may precede legitimate student prose on the same line.
_INLINE_INTERNAL_PREFIX_RE = re.compile(
    r"^\s*(?:[-*•]\s*)?(?:#{1,6}\s*)?(?:\*\*|__|\[|`)?"
    r"(?:rascunho\s+(?:did[aá]tico|pedag[oó]gico(?:\s+a\s+revisar)?|provis[oó]rio)"
    r"|resposta\s+final\s+verificada|resposta\s+verificada"
    r"|auditoria\s+(?:final|factual)|nota\s+do\s+factverifier"
    r"|factverifier|pedagogicaladapter|historicalresearcher|topicrouter)"
    r"(?:\*\*|__|\]|`)?\s*[:\-\u2013\u2014]\s*(?:\*\*|__|\]|`)?\s*",
    re.IGNORECASE,
)

# Standalone internal routing/tool metadata key-value lines.
_ROUTING_METADATA_KEY_RE = re.compile(
    r"^\s*(?:[-*•]\s*)?(?:\*\*|__|`)?"
    r"(?:intent|intencao|target_level|nivel[-_ ]alvo|historical_topic|"
    r"topico[-_ ]historico|tema[-_ ]historico|time_range|intervalo[-_ ]temporal|"
    r"recorte[-_ ]temporal|student_goal|objetivo[-_ ](?:do[-_ ])?estudante|"
    r"routing_plan|historical_evidence|pedagogical_draft|verified_response|"
    r"format_guidelines|misconceptions_to_test|grounded_facts|"
    r"pedagogical_focus|topic_id|level_label|num_questions)"
    r"(?:\*\*|__|`)?\s*[:=]\s*(?:\*\*|__|`)?",
    re.IGNORECASE,
)

# Internal tool/evidence status lines (restricted to known internal status values
# so legitimate historical prose mentioning the word "status" is never altered).
_INTERNAL_STATUS_LINE_RE = re.compile(
    r"^\s*(?:[-*•]\s*)?(?:\*\*|__|`)?status(?:\*\*|__|`)?\s*[:=]\s*"
    r"(?:\*\*|__|`)?\s*(?:ok|not_found|out_of_scope|error|skipped)\s*(?:\*\*|__|`)?"
    r"(?:\s*[.,;!?]?)?\s*$",
    re.IGNORECASE,
)

# Raw guidance strings from `app/tools/quiz.py` (`format_guidelines`) or tool guards
# that must never leak verbatim into student-facing text.
_BLUEPRINT_BOILERPLATE_SUBSTRINGS: tuple[str, ...] = (
    "elabore questoes em portugues do brasil com enunciados contextualizados",
    "elabore questoes em portugues do brasil",
    "inclua pelo menos uma questao que desafie um mito historico comum",
    "listado em misconceptions_to_test",
    "forneca gabarito comentado explicando por que a alternativa correta esta certa",
    "quiz blueprint skipped because the student requested",
)

_HORIZONTAL_RULE_RE = re.compile(r"^\s*(?:---+|\*\*\*+|___+)\s*$")


def extract_routing_intent(routing_plan: str | None) -> str | None:
    """Extract the canonical intent from a `TopicRouter` state string, if present."""
    if not routing_plan:
        return None
    for line in routing_plan.splitlines():
        folded = _fold_text(line)
        match = _INTENT_LINE_RE.match(folded)
        if match:
            raw_intent = match.group(1).strip().lower()
            return _CANONICAL_INTENT_MAP.get(raw_intent, raw_intent)
    return None


def has_explicit_quiz_request(user_text: str | None) -> bool:
    """Return True when the student message explicitly asks for a quiz or exercises."""
    if not user_text:
        return False
    folded = _fold_text(user_text)
    return bool(
        _EXPLICIT_QUIZ_NOUN_RE.search(folded)
        or _EXPLICIT_QUIZ_REQUEST_RE.search(folded)
    )


def is_quiz_tool_allowed(
    routing_plan: str | None,
    user_text: str | None = None,
) -> bool:
    """Decide whether `PedagogicalAdapter` may invoke `build_quiz_blueprint`.

    Allows the tool when:
    - `TopicRouter` classified `intent: quiz`, OR
    - the student explicitly requested exercises/questions/quiz, OR
    - no recognizable routing intent is present (fail-open for direct invocation).

    Blocks spurious tool calls when `TopicRouter` explicitly classified the turn as
    `explanation`, `timeline`, or `out_of_scope` without an explicit student quiz ask.
    """
    intent = extract_routing_intent(routing_plan)
    if intent is None or intent == "quiz":
        return True
    return has_explicit_quiz_request(user_text)


def _normalize_heading_label(line: str) -> str:
    """Strip markdown heading/emphasis wrappers and trailing punctuation from a line."""
    cleaned = line.strip()
    cleaned = re.sub(r"^#{1,6}\s*", "", cleaned)
    cleaned = re.sub(r"^[-*•]\s*", "", cleaned)
    cleaned = cleaned.strip("*_`[]() ")
    cleaned = re.sub(r"\s*[:\-\u2013\u2014]+$", "", cleaned)
    cleaned = cleaned.strip("*_`[]() ")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return _fold_text(cleaned)


def _is_blueprint_boilerplate_line(line: str) -> bool:
    """Return True when a line leaks raw `build_quiz_blueprint` format guidelines."""
    folded = _fold_text(line)
    return any(fragment in folded for fragment in _BLUEPRINT_BOILERPLATE_SUBSTRINGS)


def sanitize_student_response(text: str) -> str:
    """Strip leaked internal workflow headings, routing metadata, and blueprint text.

    Preserves legitimate Brazilian History prose, Markdown formatting, citations, and
    student-requested quizzes identically across `local` and `remote` profiles.
    """
    if not text:
        return ""

    raw_lines = text.splitlines()
    cleaned_lines: list[str] = []
    removed_any = False

    for line in raw_lines:
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append("")
            continue

        folded = _fold_text(stripped)

        if _normalize_heading_label(stripped) in _INTERNAL_HEADING_LABELS:
            removed_any = True
            continue

        if _INTERNAL_STATUS_LINE_RE.match(folded):
            removed_any = True
            continue

        if _ROUTING_METADATA_KEY_RE.match(folded):
            removed_any = True
            continue

        if _is_blueprint_boilerplate_line(stripped):
            removed_any = True
            continue

        inline_match = _INLINE_INTERNAL_PREFIX_RE.match(line)
        if inline_match:
            remainder = line[inline_match.end() :].strip()
            removed_any = True
            if remainder:
                # Re-check remainder in case it is itself a leaked status/boilerplate line.
                folded_rem = _fold_text(remainder)
                if not (
                    _INTERNAL_STATUS_LINE_RE.match(folded_rem)
                    or _ROUTING_METADATA_KEY_RE.match(folded_rem)
                    or _is_blueprint_boilerplate_line(remainder)
                ):
                    cleaned_lines.append(remainder)
            continue

        cleaned_lines.append(line)

    if removed_any:
        # Trim leading/trailing blank lines and orphaned horizontal rules left behind
        # after removing leaked header/footer blocks.
        while cleaned_lines and (
            not cleaned_lines[0].strip() or _HORIZONTAL_RULE_RE.match(cleaned_lines[0])
        ):
            cleaned_lines.pop(0)
        while cleaned_lines and (
            not cleaned_lines[-1].strip()
            or _HORIZONTAL_RULE_RE.match(cleaned_lines[-1])
        ):
            cleaned_lines.pop()

    joined = "\n".join(cleaned_lines)
    joined = re.sub(r"\n{3,}", "\n\n", joined)
    return joined.strip()
