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
)


@lru_cache(maxsize=1)
def load_corpus() -> tuple[dict[str, Any], ...]:
    """Load the immutable curated Brazilian History corpus from disk."""
    raw = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    return tuple(raw)


@lru_cache(maxsize=1)
def _load_corpus_by_id() -> dict[str, dict[str, Any]]:
    return {str(entry["id"]): entry for entry in load_corpus()}


def _build_entry_payload(
    base_entry: dict[str, Any] | None,
    group_chunks: list[ScoredChunk],
) -> dict[str, Any]:
    """Assemble an entry payload preserving the corpus contract plus verified citations."""
    top_chunk = group_chunks[0].chunk
    seen_citations: set[str] = set()
    citations: list[dict[str, Any]] = []
    matched_chunks: list[dict[str, Any]] = []

    for scored in group_chunks:
        chunk = scored.chunk
        if chunk.source_id not in seen_citations:
            seen_citations.add(chunk.source_id)
            citations.append(chunk.to_citation_dict())
        matched_chunks.append(
            {
                "chunk_id": chunk.chunk_id,
                "source_id": chunk.source_id,
                "document": chunk.document,
                "source_type": chunk.source_type,
                "bncc_codes": list(chunk.bncc_codes),
                "excerpt": chunk.text,
                "rrf_score": round(scored.rrf_score, 6),
                "bm25_score": round(scored.bm25_score, 4),
                "dense_score": (
                    round(scored.dense_score, 4)
                    if scored.dense_score is not None
                    else None
                ),
            }
        )

    if base_entry is not None:
        entry_id = str(base_entry["id"])
        corpus_source_id = f"corpus:{entry_id}"
        corpus_document = f"Corpus Curado BNCC — {base_entry['title']}"
        if corpus_source_id not in seen_citations:
            citations.append(
                {
                    "source_id": corpus_source_id,
                    "document": corpus_document,
                    "bncc_codes": list(base_entry["bncc_codes"]),
                }
            )
        payload = dict(base_entry)
        payload["source_id"] = top_chunk.source_id
        payload["document"] = top_chunk.document
        payload["citations"] = citations
        payload["matched_chunks"] = matched_chunks
        return payload

    return {
        "id": top_chunk.source_id,
        "title": top_chunk.title,
        "period": top_chunk.period,
        "bncc_codes": list(top_chunk.bncc_codes),
        "summary": top_chunk.text,
        "source_id": top_chunk.source_id,
        "document": top_chunk.document,
        "citations": citations,
        "matched_chunks": matched_chunks,
    }


def _search_with_index(
    query: str,
    *,
    max_results: int = 3,
    index: HybridRagIndex | None = None,
    embedder: Any = None,
    enable_dense: bool | None = None,
) -> dict[str, Any]:
    """Internal helper enabling deterministic unit tests with custom indexes/embedders."""
    cleaned_query = query.strip()
    if not cleaned_query:
        return {
            "status": "error",
            "message": "Query must not be empty.",
            "entries": [],
            "chunks": [],
        }

    limit = max(1, min(max_results, 5))
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
    for group_key in group_order[:limit]:
        base_entry = corpus_by_id.get(group_key)
        entries.append(_build_entry_payload(base_entry, grouped[group_key]))

    top_chunks = [
        scored.to_payload() for scored in search_result.chunks[: max(limit * 2, 5)]
    ]
    return {
        "status": "ok",
        "query": cleaned_query,
        "retrieval_mode": search_result.retrieval_mode,
        "count": len(entries),
        "entries": entries,
        "chunks": top_chunks,
    }


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
        A JSON-serializable dict with `status`, `query`, `retrieval_mode`, `count`,
        matching historical `entries`, and ranked `chunks` carrying verified citations
        (`source_id`, `document`, `bncc_codes`).
    """
    return _search_with_index(query, max_results=max_results)


__all__ = [
    "CORPUS_PATH",
    "STOPWORDS",
    "load_corpus",
    "normalize_text",
    "search_historical_corpus",
    "tokenize_query",
]
