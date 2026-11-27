"""Curated BNCC-aligned Brazilian History corpus loader and hybrid RAG search tool."""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from app.tools.rag import (
    CORPUS_PATH,
    STOPWORDS,
    HybridRagIndex,
    ScoredChunk,
    get_rag_index,
    normalize_text,
    tokenize_query,
    truncate_excerpt,
)

MAX_TOOL_PAYLOAD_BYTES = 8192
MAX_CITATIONS_PRIMARY = 3
MAX_CITATIONS_SECONDARY = 2
MAX_COMPACT_CHUNKS = 4
SECONDARY_SUMMARY_MAX_CHARS = 180
SECONDARY_FACT_MAX_CHARS = 130
SECONDARY_PERSPECTIVE_MAX_CHARS = 140
SECONDARY_MISCONCEPTION_MAX_CHARS = 130
NON_CORPUS_SUMMARY_MAX_CHARS = 280
CORPUS_CHUNK_EXCERPT_MAX_CHARS = 120
EXTERNAL_CHUNK_EXCERPT_MAX_CHARS = 200


@lru_cache(maxsize=1)
def load_corpus() -> tuple[dict[str, Any], ...]:
    """Load the immutable curated Brazilian History corpus from disk."""
    raw = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    return tuple(raw)


@lru_cache(maxsize=1)
def _load_corpus_by_id() -> dict[str, dict[str, Any]]:
    return {str(entry["id"]): entry for entry in load_corpus()}


def _canonical_citation_key(source_id: str) -> str:
    """Collapse fragment variants like `corpus:<id>#perspectivas` into one citation key."""
    return source_id.split("#", 1)[0]


def _collect_entry_citations(
    base_entry: dict[str, Any] | None,
    group_chunks: list[ScoredChunk],
    *,
    max_citations: int,
) -> list[dict[str, Any]]:
    """Build a deduplicated, bounded list of verified citations for an entry."""
    seen_keys: set[str] = set()
    citations: list[dict[str, Any]] = []

    for scored in group_chunks:
        chunk = scored.chunk
        citation_key = _canonical_citation_key(chunk.source_id)
        if citation_key in seen_keys:
            continue
        seen_keys.add(citation_key)
        citations.append(chunk.to_citation_dict())
        if len(citations) >= max_citations:
            break

    if base_entry is not None and len(citations) < max_citations:
        entry_id = str(base_entry["id"])
        corpus_source_id = f"corpus:{entry_id}"
        if corpus_source_id not in seen_keys:
            citations.append(
                {
                    "source_id": corpus_source_id,
                    "document": f"Corpus Curado BNCC — {base_entry['title']}",
                    "bncc_codes": list(base_entry["bncc_codes"]),
                }
            )

    return citations


def _compact_list(items: list[str], *, max_items: int, max_chars: int) -> list[str]:
    """Return a bounded, concisely truncated list of strings."""
    return [truncate_excerpt(str(item), max_chars) for item in items[:max_items]]


def _build_entry_payload(
    base_entry: dict[str, Any] | None,
    group_chunks: list[ScoredChunk],
    *,
    compact: bool = False,
) -> dict[str, Any]:
    """Assemble a deduplicated entry payload with verified citations and no chunk duplication."""
    top_chunk = group_chunks[0].chunk
    max_citations = MAX_CITATIONS_SECONDARY if compact else MAX_CITATIONS_PRIMARY
    citations = _collect_entry_citations(
        base_entry,
        group_chunks,
        max_citations=max_citations,
    )

    if base_entry is not None:
        if not compact:
            return {
                "id": base_entry["id"],
                "title": base_entry["title"],
                "period": base_entry["period"],
                "start_year": base_entry["start_year"],
                "end_year": base_entry["end_year"],
                "bncc_codes": list(base_entry["bncc_codes"]),
                "summary": base_entry["summary"],
                "key_facts": list(base_entry["key_facts"]),
                "primary_actors": list(base_entry["primary_actors"]),
                "critical_perspectives": list(base_entry["critical_perspectives"]),
                "common_misconceptions": list(base_entry["common_misconceptions"]),
                "recommended_sources": list(base_entry["recommended_sources"]),
                "source_id": top_chunk.source_id,
                "document": top_chunk.document,
                "citations": citations,
            }

        return {
            "id": base_entry["id"],
            "title": base_entry["title"],
            "period": base_entry["period"],
            "start_year": base_entry["start_year"],
            "end_year": base_entry["end_year"],
            "bncc_codes": list(base_entry["bncc_codes"]),
            "summary": truncate_excerpt(
                str(base_entry["summary"]), SECONDARY_SUMMARY_MAX_CHARS
            ),
            "key_facts": _compact_list(
                base_entry["key_facts"],
                max_items=2,
                max_chars=SECONDARY_FACT_MAX_CHARS,
            ),
            "primary_actors": list(base_entry["primary_actors"])[:3],
            "critical_perspectives": _compact_list(
                base_entry["critical_perspectives"],
                max_items=1,
                max_chars=SECONDARY_PERSPECTIVE_MAX_CHARS,
            ),
            "common_misconceptions": _compact_list(
                base_entry["common_misconceptions"],
                max_items=1,
                max_chars=SECONDARY_MISCONCEPTION_MAX_CHARS,
            ),
            "recommended_sources": list(base_entry["recommended_sources"])[:1],
            "source_id": top_chunk.source_id,
            "document": top_chunk.document,
            "citations": citations,
        }

    summary_limit = (
        SECONDARY_SUMMARY_MAX_CHARS if compact else NON_CORPUS_SUMMARY_MAX_CHARS
    )
    return {
        "id": top_chunk.source_id,
        "title": top_chunk.title,
        "period": top_chunk.period,
        "bncc_codes": list(top_chunk.bncc_codes),
        "summary": truncate_excerpt(top_chunk.text, summary_limit),
        "source_id": top_chunk.source_id,
        "document": top_chunk.document,
        "citations": citations,
    }


def _select_compact_chunks(
    scored_chunks: list[ScoredChunk],
    *,
    limit: int,
    included_topic_ids: set[str],
) -> list[dict[str, Any]]:
    """Deduplicate ranked chunks by `source_id` and emit concise citation payloads."""
    max_chunks = min(limit + 1, MAX_COMPACT_CHUNKS)
    seen_sources: set[str] = set()
    compact_chunks: list[dict[str, Any]] = []

    for scored in scored_chunks:
        chunk = scored.chunk
        if chunk.source_id in seen_sources:
            continue
        seen_sources.add(chunk.source_id)
        is_duplicated_corpus = (
            chunk.source_type == "corpus" and chunk.topic_id in included_topic_ids
        )
        excerpt_limit = (
            CORPUS_CHUNK_EXCERPT_MAX_CHARS
            if is_duplicated_corpus
            else EXTERNAL_CHUNK_EXCERPT_MAX_CHARS
        )
        compact_chunks.append(
            scored.to_compact_payload(max_excerpt_chars=excerpt_limit)
        )
        if len(compact_chunks) >= max_chunks:
            break

    return compact_chunks


def _serialized_size_bytes(payload: dict[str, Any]) -> int:
    """Measure compact UTF-8 JSON size of a tool payload."""
    return len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))


def _enforce_payload_budget(
    payload: dict[str, Any],
    *,
    max_bytes: int = MAX_TOOL_PAYLOAD_BYTES,
) -> dict[str, Any]:
    """Deterministically trim secondary details if a payload exceeds `max_bytes`."""
    if _serialized_size_bytes(payload) <= max_bytes:
        return payload

    # Step 1: Tighten excerpts in top-level chunks while keeping verified citations intact.
    for chunk in payload.get("chunks", []):
        if isinstance(chunk, dict) and "excerpt" in chunk:
            chunk["excerpt"] = truncate_excerpt(str(chunk["excerpt"]), 120)
    if _serialized_size_bytes(payload) <= max_bytes:
        return payload

    # Step 2: Tighten secondary entries (preserving entries[0] lossless for accuracy & quiz.py).
    entries = payload.get("entries", [])
    for secondary in entries[1:]:
        if not isinstance(secondary, dict):
            continue
        if "summary" in secondary:
            secondary["summary"] = truncate_excerpt(str(secondary["summary"]), 140)
        if "key_facts" in secondary:
            secondary["key_facts"] = _compact_list(
                list(secondary["key_facts"]), max_items=1, max_chars=120
            )
        for optional_key in (
            "primary_actors",
            "critical_perspectives",
            "common_misconceptions",
            "recommended_sources",
        ):
            if optional_key in secondary:
                secondary[optional_key] = []
        if _serialized_size_bytes(payload) <= max_bytes:
            return payload

    # Step 3: If still above budget (e.g., max_results=5), pop lowest-ranked secondary entries.
    while len(entries) > 1 and _serialized_size_bytes(payload) > max_bytes:
        entries.pop()
    payload["count"] = len(entries)
    return payload


def _coerce_int(value: int | str | float | None, default: int) -> int:
    """Coerce stringified numeric tool arguments emitted by smaller local LLMs."""
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _search_with_index(
    query: str,
    *,
    max_results: int = 3,
    index: HybridRagIndex | None = None,
    embedder: Any = None,
    enable_dense: bool | None = None,
) -> dict[str, Any]:
    """Internal helper enabling deterministic unit tests with custom indexes/embedders."""
    cleaned_query = str(query).strip()
    if not cleaned_query:
        return {
            "status": "error",
            "message": "Query must not be empty.",
            "entries": [],
            "chunks": [],
        }

    limit = max(1, min(_coerce_int(max_results, 3), 5))
    active_index = index or get_rag_index()
    tokens = tokenize_query(cleaned_query)
    if not tokens and not active_index.has_embeddings:
        return {
            "status": "not_found",
            "query": cleaned_query,
            "retrieval_mode": "bm25_lexical",
            "message": "No meaningful historical search terms were found in the query.",
            "entries": [],
            "chunks": [],
        }

    search_result = active_index.search(
        cleaned_query,
        top_k=max(limit * 4, 12),
        embedder=embedder,
        enable_dense=enable_dense,
    )
    if not search_result.chunks:
        return {
            "status": "not_found",
            "query": cleaned_query,
            "retrieval_mode": search_result.retrieval_mode,
            "message": "No curated Brazilian History entries matched the query.",
            "entries": [],
            "chunks": [],
        }

    corpus_by_id = _load_corpus_by_id()
    grouped: dict[str, list[ScoredChunk]] = {}
    group_order: list[str] = []

    for scored in search_result.chunks:
        topic_id = scored.chunk.topic_id
        group_key = (
            topic_id
            if topic_id and topic_id in corpus_by_id
            else scored.chunk.source_id
        )
        if group_key not in grouped:
            grouped[group_key] = []
            group_order.append(group_key)
        grouped[group_key].append(scored)

    entries: list[dict[str, Any]] = []
    included_topic_ids: set[str] = set()
    for rank_idx, group_key in enumerate(group_order[:limit]):
        base_entry = corpus_by_id.get(group_key)
        if base_entry is not None:
            included_topic_ids.add(group_key)
        entries.append(
            _build_entry_payload(
                base_entry,
                grouped[group_key],
                compact=rank_idx > 0,
            )
        )

    top_chunks = _select_compact_chunks(
        search_result.chunks,
        limit=limit,
        included_topic_ids=included_topic_ids,
    )
    payload = {
        "status": "ok",
        "query": cleaned_query,
        "retrieval_mode": search_result.retrieval_mode,
        "count": len(entries),
        "entries": entries,
        "chunks": top_chunks,
    }
    return _enforce_payload_budget(payload)


def search_historical_corpus(query: str, max_results: int = 3) -> dict[str, Any]:
    """Search the curated BNCC-aligned Brazilian History knowledge base via hybrid RAG.

    Uses offline-capable hybrid retrieval (BM25 lexical + local Ollama dense embeddings
    fused with Reciprocal Rank Fusion) over the curated corpus, BNCC guidelines,
    Constitution of 1988 excerpts, primary historical sources, and ENEM references,
    with automatic fallback to deterministic BM25 lexical search when embeddings or
    Ollama are unavailable. Both `local` and `remote` profiles query the exact same index.

    Args:
        query: Historical topic, event, actor, year, or BNCC code in Portuguese or English.
        max_results: Maximum number of matching entries to return (1 to 5).

    Returns:
        A compact, deduplicated JSON-serializable dict (<= 8 KB) with `status`, `query`,
        `retrieval_mode`, `count`, matching historical `entries`, and deduplicated `chunks`
        carrying verified citations (`source_id`, `document`, `bncc_codes`) and concise
        `excerpt` text.
    """
    return _search_with_index(query, max_results=max_results)


__all__ = [
    "CORPUS_PATH",
    "MAX_TOOL_PAYLOAD_BYTES",
    "STOPWORDS",
    "load_corpus",
    "normalize_text",
    "search_historical_corpus",
    "tokenize_query",
]
