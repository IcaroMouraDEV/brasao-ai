"""Local LLM-as-judge for `custom_response_quality` (see eval_config.yaml).

Evaluates Brazilian History factual accuracy, hallucination resistance, and UN SDG 4
pedagogical quality (clarity, grade-level adaptation, and critical thinking).
"""

import threading

from google import genai
from google.genai import types
from pydantic import BaseModel

_local = threading.local()


class _Verdict(BaseModel):
    score: int  # 1-5
    explanation: str


def _client() -> genai.Client:
    """One client per grading thread.

    The eval SDK grades cases on its own thread pool; this initialization runs once
    per thread. Avoids creating a new client for each eval case, which would re-do
    ADC and the TLS handshake every time. Each thread gets its own client, because
    google-auth freezes the SSL context after the first connection when a client
    certificate is present.
    """
    client = getattr(_local, "client", None)
    if client is None:
        # AI Studio (GEMINI_API_KEY) or Gemini platform (ADC).
        client = _local.client = genai.Client()
    return client


def evaluate(instance):
    reference = instance.get("reference")
    rubric = (
        "Grade the Brazilian History tutor's final response on a 1-5 scale (1 poor, 5 excellent) "
        "considering: (1) historical accuracy and absence of hallucinations/anachronisms, "
        "(2) UN SDG 4 pedagogical quality (clear Brazilian Portuguese, grade-level adaptation, "
        "critical thinking, and multiple historical perspectives), and (3) grounding in tool outputs."
    )
    if reference:
        rubric += (
            " The response should agree with the expected historical ground truth below; "
            "strongly penalize fabricated dates, historical myths, or factual contradictions."
        )
    prompt = (
        f"You are an expert Brazilian History educator and QA evaluator. {rubric}\n"
        f"User Prompt: {instance.get('prompt', '')}\n"
        f"Final Response: {instance.get('response', '')}\n"
    )
    if reference:
        prompt += f"Expected Answer (ground truth): {reference}\n"
    prompt += f"Full Agent Trace: {instance.get('agent_data', '')}\n"

    response = _client().models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0,  # deterministic grading
            response_mime_type="application/json",
            response_schema=_Verdict,  # guaranteed schema-valid JSON
        ),
    )
    verdict = response.parsed
    if verdict is None:  # model returned nothing usable
        return {"score": 0, "explanation": response.text or ""}
    return {"score": max(1, min(5, verdict.score)), "explanation": verdict.explanation}
