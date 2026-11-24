"""Offline-capable hybrid RAG engine (BM25 + local Ollama embeddings + RRF) over SQLite."""

from __future__ import annotations

import json
import logging
import math
import os
import re
import sqlite3
import tempfile
import unicodedata
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Protocol

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CORPUS_PATH = DATA_DIR / "brazil_history_corpus.json"
SOURCES_DIR = DATA_DIR / "sources"
DEFAULT_RAG_INDEX_PATH = DATA_DIR / "rag_index.sqlite3"

DEFAULT_OLLAMA_API_BASE = "http://localhost:11434"
DEFAULT_OLLAMA_EMBED_MODEL = "nomic-embed-text"
DEFAULT_EMBED_TIMEOUT_S = 2.0
DEFAULT_CHUNK_SIZE = 700
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_RRF_K = 60
DEFAULT_MIN_DENSE_SIMILARITY = 0.55
SCHEMA_VERSION = "1"

RetrievalMode = Literal["hybrid", "bm25_lexical"]

STOPWORDS = frozenset(
    {
        "de",
        "da",
        "do",
        "das",
        "dos",
        "em",
        "no",
        "na",
        "nos",
        "nas",
        "os",
        "as",
        "um",
        "uma",
        "uns",
        "umas",
        "por",
        "para",
        "com",
        "sem",
        "que",
        "qual",
        "quais",
        "como",
        "sobre",
        "entre",
        "pelo",
        "pela",
        "pelos",
        "pelas",
        "foi",
        "sao",
        "eram",
        "foram",
        "esta",
        "este",
        "esse",
        "essa",
        "isso",
        "mais",
        "muito",
        "tambem",
        "apos",
        "ate",
        "the",
        "and",
        "for",
        "from",
        "with",
        "about",
    }
)

_BNCC_TOKEN_RE = re.compile(r"^(ef0[6-9]hi\d{2}|em13chs\d{3})$")
_PT_SUFFIX_RULES: tuple[tuple[str, str, int], ...] = (
    ("amentos", "amento", 4),
    ("imentos", "imento", 4),
    ("acoes", "acao", 3),
    ("icoes", "icao", 3),
    ("coes", "cao", 3),
    ("oes", "ao", 3),
    ("istas", "ista", 4),
    ("ismos", "ismo", 4),
    ("idades", "idade", 4),
    ("aveis", "avel", 4),
    ("iveis", "ivel", 4),
    ("ados", "ado", 4),
    ("adas", "ada", 4),
    ("idos", "ido", 4),
    ("idas", "ida", 4),
    ("ais", "al", 4),
    ("eis", "el", 4),
)


class EmbeddingError(RuntimeError):
    """Raised when local embedding generation fails or is unavailable."""


class EmbeddingProvider(Protocol):
    """Protocol for offline/local dense text embedding providers."""

    model: str

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one dense embedding vector per input string."""


def normalize_text(text: str) -> str:
    """Lowercase and strip Portuguese diacritics for accent-insensitive search."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def stem_pt_token(token: str) -> str:
    """Apply lightweight, deterministic Portuguese suffix normalization for BM25."""
    if len(token) < 4 or token.isdigit() or "-" in token or _BNCC_TOKEN_RE.match(token):
        return token

    for suffix, replacement, min_stem in _PT_SUFFIX_RULES:
        if token.endswith(suffix) and (len(token) - len(suffix)) >= min_stem:
            return token[: -len(suffix)] + replacement

    if (
        len(token) >= 5
        and token.endswith("s")
        and not token.endswith(("ss", "us", "is", "os"))
    ):
        return token[:-1]
    if len(token) >= 6 and token.endswith("os") and not token.endswith("ios"):
        return token[:-1]
    return token


def tokenize_query(text: str) -> list[str]:
    """Split normalized text into meaningful alphanumeric search tokens."""
    normalized = normalize_text(text)
    raw_tokens = re.findall(r"[a-z0-9-]+", normalized)
    return [tok for tok in raw_tokens if len(tok) >= 3 and tok not in STOPWORDS]


def tokenize_for_bm25(text: str) -> list[str]:
    """Tokenize and apply light Portuguese suffix normalization for BM25 retrieval."""
    return [stem_pt_token(tok) for tok in tokenize_query(text)]


def chunk_text(
    text: str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping chunks preferring paragraph and sentence boundaries."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must satisfy 0 <= chunk_overlap < chunk_size.")

    cleaned = re.sub(r"\n{3,}", "\n\n", text.strip())
    if not cleaned:
        return []
    if len(cleaned) <= chunk_size:
        return [cleaned]

    chunks: list[str] = []
    start = 0
    length = len(cleaned)

    while start < length:
        end = min(start + chunk_size, length)
        if end < length:
            split_pos = _find_natural_boundary(cleaned, start, end)
            if split_pos > start:
                end = split_pos

        piece = cleaned[start:end].strip()
        if piece:
            chunks.append(piece)

        if end >= length:
            break

        next_start = max(end - chunk_overlap, start + 1)
        if next_start < length and next_start > 0 and not cleaned[next_start].isspace():
            ws_pos = cleaned.find(" ", next_start, end)
            if ws_pos != -1 and ws_pos < end - 1:
                next_start = ws_pos + 1
        start = next_start

    return chunks


def _find_natural_boundary(text: str, start: int, end: int) -> int:
    """Locate the rightmost paragraph, sentence, or whitespace boundary within a window."""
    min_cut = start + max(1, (end - start) // 2)

    para_idx = text.rfind("\n\n", min_cut, end)
    if para_idx != -1:
        return para_idx

    sentence_window = text[min_cut:end]
    sentence_matches = list(re.finditer(r"[.!?](?:\s|$)", sentence_window))
    if sentence_matches:
        return min_cut + sentence_matches[-1].start() + 1

    space_idx = text.rfind(" ", min_cut, end)
    if space_idx != -1:
        return space_idx

    return end


@dataclass(frozen=True)
class RagChunk:
    """A single indexed text chunk with verified source citations and optional embedding."""

    chunk_id: str
    source_id: str
    document: str
    source_type: str
    topic_id: str
    title: str
    period: str
    bncc_codes: tuple[str, ...]
    text: str
    tokens: tuple[str, ...]
    title_tokens: tuple[str, ...]
    embedding: tuple[float, ...] | None = None

    def to_citation_dict(self) -> dict[str, Any]:
        """Return the compact verified source citation payload."""
        return {
            "source_id": self.source_id,
            "document": self.document,
            "bncc_codes": list(self.bncc_codes),
        }


@dataclass(frozen=True)
class ScoredChunk:
    """A retrieved chunk with BM25, dense cosine, and Reciprocal Rank Fusion scores."""

    chunk: RagChunk
    rrf_score: float
    bm25_score: float
    dense_score: float | None
    matched_tokens: int

    def to_payload(self) -> dict[str, Any]:
        """Serialize the scored chunk with citations for tool outputs."""
        return {
            "chunk_id": self.chunk.chunk_id,
            "source_id": self.chunk.source_id,
            "document": self.chunk.document,
            "source_type": self.chunk.source_type,
            "topic_id": self.chunk.topic_id,
            "title": self.chunk.title,
            "period": self.chunk.period,
            "bncc_codes": list(self.chunk.bncc_codes),
            "text": self.chunk.text,
            "rrf_score": round(self.rrf_score, 6),
            "bm25_score": round(self.bm25_score, 4),
            "dense_score": (
                round(self.dense_score, 4) if self.dense_score is not None else None
            ),
        }


@dataclass(frozen=True)
class HybridSearchResult:
    """Result returned by `HybridRagIndex.search`."""

    retrieval_mode: RetrievalMode
    chunks: list[ScoredChunk]


def build_chunks_from_corpus(
    corpus: Sequence[Mapping[str, Any]],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[RagChunk]:
    """Convert structured Brazilian History corpus entries into indexed RAG chunks."""
    chunks: list[RagChunk] = []

    for entry in corpus:
        entry_id = str(entry["id"])
        title = str(entry["title"])
        period = str(entry["period"])
        bncc_codes = tuple(str(code) for code in entry.get("bncc_codes", ()))
        document = f"Corpus Curado BNCC — {title}"
        title_tokens = tuple(tokenize_for_bm25(f"{title} {period}"))

        overview_parts = [
            f"{title} ({period}, {entry['start_year']}-{entry['end_year']}).",
            str(entry["summary"]),
            "Fatos centrais: " + " ".join(str(fact) for fact in entry["key_facts"]),
            "Atores históricos: "
            + ", ".join(str(actor) for actor in entry["primary_actors"])
            + ".",
            "Códigos BNCC: " + ", ".join(bncc_codes) + ".",
        ]
        overview_text = "\n".join(overview_parts)
        for idx, piece in enumerate(
            chunk_text(
                overview_text,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        ):
            chunks.append(
                RagChunk(
                    chunk_id=f"corpus:{entry_id}:overview:{idx}",
                    source_id=f"corpus:{entry_id}",
                    document=document,
                    source_type="corpus",
                    topic_id=entry_id,
                    title=title,
                    period=period,
                    bncc_codes=bncc_codes,
                    text=piece,
                    tokens=tuple(tokenize_for_bm25(f"{title} {period} {piece}")),
                    title_tokens=title_tokens,
                )
            )

        perspectives_parts = [
            f"{title} — Perspectivas críticas e fontes.",
            "Perspectivas críticas: "
            + " ".join(str(item) for item in entry["critical_perspectives"]),
            "Mitos e equívocos comuns: "
            + " ".join(str(item) for item in entry["common_misconceptions"]),
            "Fontes recomendadas: "
            + "; ".join(str(src) for src in entry["recommended_sources"])
            + ".",
        ]
        perspectives_text = "\n".join(perspectives_parts)
        for idx, piece in enumerate(
            chunk_text(
                perspectives_text,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        ):
            chunks.append(
                RagChunk(
                    chunk_id=f"corpus:{entry_id}:perspectives:{idx}",
                    source_id=f"corpus:{entry_id}#perspectivas",
                    document=document,
                    source_type="corpus",
                    topic_id=entry_id,
                    title=title,
                    period=period,
                    bncc_codes=bncc_codes,
                    text=piece,
                    tokens=tuple(tokenize_for_bm25(f"{title} {period} {piece}")),
                    title_tokens=title_tokens,
                )
            )

    return chunks


def build_chunks_from_sources(
    sources_dir: Path = SOURCES_DIR,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[RagChunk]:
    """Load and chunk curated BNCC, Constitution, primary source, and ENEM documents."""
    if not sources_dir.exists():
        return []

    chunks: list[RagChunk] = []
    for path in sorted(sources_dir.iterdir()):
        if not path.is_file():
            continue

        if path.suffix.lower() == ".json":
            raw_items = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw_items, list):
                raise ValueError(
                    f"Curated source JSON '{path}' must contain a list of documents."
                )
            for item in raw_items:
                source_id = str(item["source_id"])
                document = str(item["document"])
                source_type = str(item.get("source_type", "reference"))
                title = str(item.get("title", document))
                period = str(item.get("period", "História do Brasil"))
                topic_ids = [str(tid) for tid in item.get("topic_ids", ())]
                primary_topic_id = (
                    topic_ids[0]
                    if topic_ids
                    and (
                        len(topic_ids) == 1
                        or source_type in {"constitution", "primary_source"}
                    )
                    else ""
                )
                bncc_codes = tuple(str(code) for code in item.get("bncc_codes", ()))
                text = str(item["text"])
                title_tokens = tuple(tokenize_for_bm25(f"{title} {document} {period}"))

                for idx, piece in enumerate(
                    chunk_text(
                        text,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )
                ):
                    searchable = (
                        f"{title} {document} {period} {' '.join(bncc_codes)} {piece}"
                    )
                    chunks.append(
                        RagChunk(
                            chunk_id=f"{source_id}:{idx}",
                            source_id=source_id,
                            document=document,
                            source_type=source_type,
                            topic_id=primary_topic_id,
                            title=title,
                            period=period,
                            bncc_codes=bncc_codes,
                            text=piece,
                            tokens=tuple(tokenize_for_bm25(searchable)),
                            title_tokens=title_tokens,
                        )
                    )
        elif path.suffix.lower() in {".md", ".txt"}:
            raw_text = path.read_text(encoding="utf-8").strip()
            if not raw_text:
                continue
            first_line = raw_text.splitlines()[0].lstrip("#").strip()
            title = first_line or path.stem.replace("_", " ").title()
            source_id = f"doc:{path.stem}"
            document = f"Documento Curado — {title}"
            title_tokens = tuple(tokenize_for_bm25(title))
            for idx, piece in enumerate(
                chunk_text(
                    raw_text,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
            ):
                chunks.append(
                    RagChunk(
                        chunk_id=f"{source_id}:{idx}",
                        source_id=source_id,
                        document=document,
                        source_type="document",
                        topic_id="",
                        title=title,
                        period="História do Brasil",
                        bncc_codes=(),
                        text=piece,
                        tokens=tuple(tokenize_for_bm25(f"{title} {piece}")),
                        title_tokens=title_tokens,
                    )
                )

    return chunks


def collect_all_chunks(
    corpus_path: Path = CORPUS_PATH,
    sources_dir: Path = SOURCES_DIR,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[RagChunk]:
    """Build all RAG chunks from the curated JSON corpus and reference documents."""
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    corpus_chunks = build_chunks_from_corpus(
        corpus,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    source_chunks = build_chunks_from_sources(
        sources_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return [*corpus_chunks, *source_chunks]


def cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """Compute cosine similarity between two dense vectors using pure stdlib math."""
    if len(vec_a) != len(vec_b) or not vec_a:
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for a, b in zip(vec_a, vec_b, strict=True):
        dot += a * b
        norm_a += a * a
        norm_b += b * b
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def reciprocal_rank_fusion(
    bm25_ranks: Mapping[str, int],
    dense_ranks: Mapping[str, int],
    *,
    k: int = DEFAULT_RRF_K,
    bm25_weight: float = 1.0,
    dense_weight: float = 1.0,
) -> dict[str, float]:
    """Fuse lexical BM25 ranks and dense similarity ranks via Reciprocal Rank Fusion."""
    fused: dict[str, float] = {}
    for chunk_id in set(bm25_ranks) | set(dense_ranks):
        score = 0.0
        if chunk_id in bm25_ranks:
            score += bm25_weight / (k + bm25_ranks[chunk_id])
        if chunk_id in dense_ranks:
            score += dense_weight / (k + dense_ranks[chunk_id])
        fused[chunk_id] = score
    return fused


class OllamaEmbeddingClient:
    """Zero-dependency HTTP client for offline local embeddings via Ollama."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        timeout_s: float | None = None,
    ) -> None:
        self.model = (
            model or os.getenv("OLLAMA_EMBED_MODEL", DEFAULT_OLLAMA_EMBED_MODEL)
        ).strip()
        raw_base = (
            base_url or os.getenv("OLLAMA_API_BASE", DEFAULT_OLLAMA_API_BASE)
        ).strip()
        self.base_url = raw_base.rstrip("/")
        self.timeout_s = (
            timeout_s
            if timeout_s is not None
            else float(os.getenv("RAG_EMBED_TIMEOUT_S", str(DEFAULT_EMBED_TIMEOUT_S)))
        )

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Generate embeddings via Ollama `/api/embed` with `/api/embeddings` fallback."""
        if not texts:
            return []

        payload = {"model": self.model, "input": list(texts)}
        try:
            response = self._post_json("/api/embed", payload)
            embeddings = response.get("embeddings")
            if isinstance(embeddings, list) and len(embeddings) == len(texts):
                return [[float(v) for v in vec] for vec in embeddings]
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise EmbeddingError(
                    f"Ollama /api/embed failed with HTTP {exc.code} for model '{self.model}'."
                ) from exc
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            raise EmbeddingError(
                f"Ollama embedding server unreachable at {self.base_url}: {exc}"
            ) from exc

        # Legacy Ollama endpoint fallback (one request per text).
        vectors: list[list[float]] = []
        for text in texts:
            try:
                response = self._post_json(
                    "/api/embeddings", {"model": self.model, "prompt": text}
                )
            except (
                urllib.error.HTTPError,
                urllib.error.URLError,
                TimeoutError,
                OSError,
                ValueError,
            ) as exc:
                raise EmbeddingError(
                    f"Ollama /api/embeddings failed at {self.base_url}: {exc}"
                ) from exc
            vector = response.get("embedding")
            if not isinstance(vector, list) or not vector:
                raise EmbeddingError(
                    f"Ollama returned an empty embedding vector for model '{self.model}'."
                )
            vectors.append([float(v) for v in vector])
        return vectors

    def _post_json(self, path: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_s) as resp:
            decoded = resp.read().decode("utf-8")
        parsed = json.loads(decoded)
        if not isinstance(parsed, dict):
            raise ValueError("Expected JSON object response from Ollama.")
        return parsed


class BM25Index:
    """Pure-Python field-aware Okapi BM25 lexical index over `RagChunk` records."""

    def __init__(
        self,
        chunks: Sequence[RagChunk],
        *,
        k1: float = 1.5,
        b: float = 0.75,
        title_boost: float = 2.0,
    ) -> None:
        self.chunks = tuple(chunks)
        self.k1 = k1
        self.b = b
        self.title_boost = title_boost

        self._doc_term_freqs: list[dict[str, float]] = []
        self._doc_token_sets: list[frozenset[str]] = []
        self._title_token_sets: list[frozenset[str]] = []
        self._doc_lengths: list[float] = []
        doc_freq: Counter[str] = Counter()

        for chunk in self.chunks:
            body_counts = Counter(chunk.tokens)
            title_counts = Counter(chunk.title_tokens)
            weighted_tf: dict[str, float] = {}
            for term, count in body_counts.items():
                weighted_tf[term] = float(count)
            for term, count in title_counts.items():
                weighted_tf[term] = (
                    weighted_tf.get(term, 0.0) + self.title_boost * count
                )

            self._doc_term_freqs.append(weighted_tf)
            self._doc_token_sets.append(frozenset(weighted_tf))
            self._title_token_sets.append(frozenset(title_counts))
            self._doc_lengths.append(sum(weighted_tf.values()))
            for term in weighted_tf:
                doc_freq[term] += 1

        total_docs = len(self.chunks)
        self._avgdl = (
            (sum(self._doc_lengths) / total_docs) if total_docs > 0 else 1.0
        ) or 1.0
        self._idf: dict[str, float] = {
            term: math.log(1.0 + (total_docs - df + 0.5) / (df + 0.5))
            for term, df in doc_freq.items()
        }

    def score_query(
        self,
        query_tokens: Sequence[str],
    ) -> list[tuple[int, float, int]]:
        """Score all chunks against tokenized query and apply anti-false-positive gating."""
        unique_tokens = list(dict.fromkeys(query_tokens))
        if not unique_tokens or not self.chunks:
            return []

        raw_matches: list[tuple[int, float, set[str], int]] = []
        group_matched_tokens: dict[str, set[str]] = {}
        group_title_hits: dict[str, int] = {}

        for idx, chunk in enumerate(self.chunks):
            tf_map = self._doc_term_freqs[idx]
            title_set = self._title_token_sets[idx]
            doc_len = self._doc_lengths[idx]

            score = 0.0
            matched_set: set[str] = set()
            title_hits = 0

            for term in unique_tokens:
                tf = tf_map.get(term, 0.0)
                if tf <= 0.0:
                    continue
                matched_set.add(term)
                if term in title_set:
                    title_hits += 1
                idf = self._idf.get(term, 0.0)
                denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self._avgdl))
                score += idf * ((tf * (self.k1 + 1.0)) / denom)

            if score > 0.0:
                group_key = chunk.topic_id or chunk.source_id
                group_matched_tokens.setdefault(group_key, set()).update(matched_set)
                group_title_hits[group_key] = (
                    group_title_hits.get(group_key, 0) + title_hits
                )
                raw_matches.append((idx, score, matched_set, title_hits))

        results: list[tuple[int, float, int]] = []
        num_query_tokens = len(unique_tokens)

        for idx, score, matched_set, title_hits in raw_matches:
            chunk = self.chunks[idx]
            if num_query_tokens >= 4:
                group_key = chunk.topic_id or chunk.source_id
                group_coverage = len(group_matched_tokens.get(group_key, matched_set))
                has_title_hit = title_hits > 0 or group_title_hits.get(group_key, 0) > 0
                min_required = (
                    max(2, num_query_tokens // 3)
                    if has_title_hit
                    else max(2, (num_query_tokens + 1) // 2)
                )
                if group_coverage < min_required:
                    continue
            results.append((idx, score, len(matched_set)))

        results.sort(
            key=lambda item: (item[1], item[2], self.chunks[item[0]].chunk_id),
            reverse=True,
        )
        return results


class HybridRagIndex:
    """Offline-capable hybrid retriever combining BM25, local dense vectors, and RRF."""

    def __init__(
        self,
        chunks: Sequence[RagChunk],
        *,
        embedding_model: str | None = None,
        embedding_dim: int = 0,
        source_path: Path | None = None,
    ) -> None:
        self.chunks = tuple(chunks)
        self._chunks_by_id = {chunk.chunk_id: chunk for chunk in self.chunks}
        self._bm25 = BM25Index(self.chunks)
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim or self._infer_embedding_dim(self.chunks)
        self.has_embeddings = self.embedding_dim > 0 and any(
            chunk.embedding is not None for chunk in self.chunks
        )
        self.source_path = source_path

    @staticmethod
    def _infer_embedding_dim(chunks: Sequence[RagChunk]) -> int:
        for chunk in chunks:
            if chunk.embedding:
                return len(chunk.embedding)
        return 0

    def search(
        self,
        query: str,
        *,
        top_k: int = 6,
        embedder: EmbeddingProvider | None = None,
        enable_dense: bool | None = None,
        min_dense_similarity: float | None = None,
        rrf_k: int = DEFAULT_RRF_K,
    ) -> HybridSearchResult:
        """Execute hybrid (BM25 + dense RRF) retrieval with graceful BM25 fallback."""
        cleaned = query.strip()
        if not cleaned or not self.chunks:
            return HybridSearchResult(retrieval_mode="bm25_lexical", chunks=[])

        query_tokens = tokenize_for_bm25(cleaned)
        bm25_scored = self._bm25.score_query(query_tokens)

        bm25_ranks: dict[str, int] = {}
        bm25_scores: dict[str, float] = {}
        matched_token_counts: dict[str, int] = {}
        for rank, (chunk_idx, score, matched_count) in enumerate(bm25_scored, start=1):
            chunk_id = self.chunks[chunk_idx].chunk_id
            bm25_ranks[chunk_id] = rank
            bm25_scores[chunk_id] = score
            matched_token_counts[chunk_id] = matched_count

        dense_allowed = (
            enable_dense
            if enable_dense is not None
            else os.getenv("RAG_ENABLE_DENSE", "true").strip().lower()
            not in {"0", "false", "no", "off"}
        )
        dense_threshold = (
            min_dense_similarity
            if min_dense_similarity is not None
            else float(
                os.getenv("RAG_MIN_DENSE_SIMILARITY", str(DEFAULT_MIN_DENSE_SIMILARITY))
            )
        )

        dense_ranks: dict[str, int] = {}
        dense_scores: dict[str, float] = {}
        retrieval_mode: RetrievalMode = "bm25_lexical"

        if dense_allowed and self.has_embeddings:
            query_vector = self._try_embed_query(cleaned, embedder=embedder)
            if query_vector is not None and len(query_vector) == self.embedding_dim:
                retrieval_mode = "hybrid"
                dense_candidates: list[tuple[str, float]] = []
                for chunk in self.chunks:
                    if chunk.embedding is None:
                        continue
                    sim = cosine_similarity(query_vector, chunk.embedding)
                    if sim >= dense_threshold:
                        dense_candidates.append((chunk.chunk_id, sim))
                dense_candidates.sort(key=lambda item: (-item[1], item[0]))
                for rank, (chunk_id, sim) in enumerate(dense_candidates, start=1):
                    dense_ranks[chunk_id] = rank
                    dense_scores[chunk_id] = sim

        if not bm25_ranks and not dense_ranks:
            return HybridSearchResult(retrieval_mode=retrieval_mode, chunks=[])

        rrf_scores = reciprocal_rank_fusion(
            bm25_ranks=bm25_ranks,
            dense_ranks=dense_ranks,
            k=rrf_k,
        )
        ranked_ids = sorted(
            rrf_scores,
            key=lambda cid: (
                -rrf_scores[cid],
                -(dense_scores.get(cid, -1.0)),
                -(bm25_scores.get(cid, 0.0)),
                cid,
            ),
        )

        scored_chunks = [
            ScoredChunk(
                chunk=self._chunks_by_id[cid],
                rrf_score=rrf_scores[cid],
                bm25_score=bm25_scores.get(cid, 0.0),
                dense_score=dense_scores.get(cid),
                matched_tokens=matched_token_counts.get(cid, 0),
            )
            for cid in ranked_ids[:top_k]
        ]
        return HybridSearchResult(retrieval_mode=retrieval_mode, chunks=scored_chunks)

    def _try_embed_query(
        self,
        query: str,
        *,
        embedder: EmbeddingProvider | None = None,
    ) -> list[float] | None:
        provider = embedder or OllamaEmbeddingClient(model=self.embedding_model)
        try:
            vectors = provider.embed_texts([query])
        except EmbeddingError as exc:
            logger.debug("Falling back to BM25 lexical retrieval: %s", exc)
            return None
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.debug("Unexpected embedding error; using BM25 fallback: %s", exc)
            return None
        if not vectors or not vectors[0]:
            return None
        return vectors[0]


def save_rag_index(
    index_path: Path,
    chunks: Sequence[RagChunk],
    *,
    embedding_model: str | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> dict[str, Any]:
    """Persist chunks and optional local dense embeddings to a reproducible SQLite DB."""
    index_path.parent.mkdir(parents=True, exist_ok=True)
    embedding_dim = 0
    for chunk in chunks:
        if chunk.embedding:
            embedding_dim = len(chunk.embedding)
            break
    has_embeddings = embedding_dim > 0 and all(
        chunk.embedding is not None and len(chunk.embedding) == embedding_dim
        for chunk in chunks
    )

    fd, tmp_name = tempfile.mkstemp(
        prefix=".rag_index_",
        suffix=".sqlite3.tmp",
        dir=str(index_path.parent),
    )
    os.close(fd)
    tmp_path = Path(tmp_name)

    try:
        conn = sqlite3.connect(str(tmp_path))
        try:
            with conn:
                conn.execute(
                    """
                    CREATE TABLE metadata (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE chunks (
                        chunk_id TEXT PRIMARY KEY,
                        source_id TEXT NOT NULL,
                        document TEXT NOT NULL,
                        source_type TEXT NOT NULL,
                        topic_id TEXT NOT NULL,
                        title TEXT NOT NULL,
                        period TEXT NOT NULL,
                        bncc_codes TEXT NOT NULL,
                        text TEXT NOT NULL,
                        tokens TEXT NOT NULL,
                        title_tokens TEXT NOT NULL,
                        embedding TEXT
                    )
                    """
                )
                conn.execute("CREATE INDEX idx_chunks_source_id ON chunks (source_id)")
                conn.execute("CREATE INDEX idx_chunks_topic_id ON chunks (topic_id)")
                conn.execute(
                    "CREATE INDEX idx_chunks_source_type ON chunks (source_type)"
                )

                metadata = {
                    "schema_version": SCHEMA_VERSION,
                    "embedding_model": embedding_model or "",
                    "embedding_dim": str(embedding_dim if has_embeddings else 0),
                    "has_embeddings": "true" if has_embeddings else "false",
                    "chunk_size": str(chunk_size),
                    "chunk_overlap": str(chunk_overlap),
                    "chunk_count": str(len(chunks)),
                    "built_at": datetime.now(UTC).isoformat(),
                }
                conn.executemany(
                    "INSERT INTO metadata (key, value) VALUES (?, ?)",
                    list(metadata.items()),
                )

                rows = [
                    (
                        chunk.chunk_id,
                        chunk.source_id,
                        chunk.document,
                        chunk.source_type,
                        chunk.topic_id,
                        chunk.title,
                        chunk.period,
                        json.dumps(list(chunk.bncc_codes), ensure_ascii=False),
                        chunk.text,
                        json.dumps(list(chunk.tokens), ensure_ascii=False),
                        json.dumps(list(chunk.title_tokens), ensure_ascii=False),
                        (
                            json.dumps(list(chunk.embedding))
                            if has_embeddings and chunk.embedding is not None
                            else None
                        ),
                    )
                    for chunk in chunks
                ]
                conn.executemany(
                    """
                    INSERT INTO chunks (
                        chunk_id, source_id, document, source_type, topic_id,
                        title, period, bncc_codes, text, tokens, title_tokens, embedding
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
        finally:
            conn.close()

        os.replace(tmp_path, index_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    return {
        "index_path": str(index_path),
        "chunk_count": len(chunks),
        "has_embeddings": has_embeddings,
        "embedding_model": embedding_model if has_embeddings else None,
        "embedding_dim": embedding_dim if has_embeddings else 0,
    }


def load_rag_index(index_path: Path) -> HybridRagIndex:
    """Load a persisted SQLite hybrid RAG index from disk."""
    conn = sqlite3.connect(str(index_path))
    try:
        meta_rows = conn.execute("SELECT key, value FROM metadata").fetchall()
        metadata = {str(key): str(val) for key, val in meta_rows}
        chunk_rows = conn.execute(
            """
            SELECT
                chunk_id, source_id, document, source_type, topic_id,
                title, period, bncc_codes, text, tokens, title_tokens, embedding
            FROM chunks
            ORDER BY chunk_id ASC
            """
        ).fetchall()
    finally:
        conn.close()

    chunks: list[RagChunk] = []
    for row in chunk_rows:
        raw_embedding = row[11]
        embedding = (
            tuple(float(v) for v in json.loads(raw_embedding))
            if raw_embedding is not None
            else None
        )
        chunks.append(
            RagChunk(
                chunk_id=str(row[0]),
                source_id=str(row[1]),
                document=str(row[2]),
                source_type=str(row[3]),
                topic_id=str(row[4]),
                title=str(row[5]),
                period=str(row[6]),
                bncc_codes=tuple(str(c) for c in json.loads(row[7])),
                text=str(row[8]),
                tokens=tuple(str(t) for t in json.loads(row[9])),
                title_tokens=tuple(str(t) for t in json.loads(row[10])),
                embedding=embedding,
            )
        )

    embedding_model = metadata.get("embedding_model") or None
    embedding_dim = int(metadata.get("embedding_dim", "0") or 0)
    return HybridRagIndex(
        chunks,
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
        source_path=index_path,
    )


def build_and_save_index(
    *,
    corpus_path: Path = CORPUS_PATH,
    sources_dir: Path = SOURCES_DIR,
    output_path: Path = DEFAULT_RAG_INDEX_PATH,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    embedder: EmbeddingProvider | None = None,
    include_embeddings: bool = True,
    allow_lexical_fallback: bool = False,
    batch_size: int = 16,
) -> dict[str, Any]:
    """Chunk curated materials, optionally embed via local Ollama, and write SQLite index."""
    base_chunks = collect_all_chunks(
        corpus_path=corpus_path,
        sources_dir=sources_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    final_chunks = base_chunks
    used_model: str | None = None

    if include_embeddings and base_chunks:
        provider = embedder or OllamaEmbeddingClient()
        try:
            vectors: list[list[float]] = []
            step = max(1, batch_size)
            for start in range(0, len(base_chunks), step):
                batch = base_chunks[start : start + step]
                batch_texts = [
                    f"{chunk.title}\n{chunk.document}\n{chunk.text}" for chunk in batch
                ]
                vectors.extend(provider.embed_texts(batch_texts))

            if len(vectors) != len(base_chunks):
                raise EmbeddingError(
                    f"Expected {len(base_chunks)} embeddings, received {len(vectors)}."
                )

            final_chunks = [
                RagChunk(
                    chunk_id=chunk.chunk_id,
                    source_id=chunk.source_id,
                    document=chunk.document,
                    source_type=chunk.source_type,
                    topic_id=chunk.topic_id,
                    title=chunk.title,
                    period=chunk.period,
                    bncc_codes=chunk.bncc_codes,
                    text=chunk.text,
                    tokens=chunk.tokens,
                    title_tokens=chunk.title_tokens,
                    embedding=tuple(float(v) for v in vector),
                )
                for chunk, vector in zip(base_chunks, vectors, strict=True)
            ]
            used_model = provider.model
        except EmbeddingError:
            if not allow_lexical_fallback:
                raise
            logger.warning(
                "Ollama embeddings unavailable; building BM25-only SQLite index."
            )

    return save_rag_index(
        output_path,
        final_chunks,
        embedding_model=used_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def resolve_rag_index_path() -> Path:
    """Resolve the active SQLite RAG index path shared across local and remote profiles."""
    configured = os.getenv("RAG_INDEX_PATH", "").strip()
    if configured:
        return Path(configured)
    return DEFAULT_RAG_INDEX_PATH


@lru_cache(maxsize=4)
def _load_cached_index(index_path_str: str, mtime_ns: int) -> HybridRagIndex:
    if index_path_str and mtime_ns >= 0:
        path = Path(index_path_str)
        try:
            return load_rag_index(path)
        except (sqlite3.Error, ValueError, KeyError, OSError) as exc:
            logger.warning(
                "Could not load SQLite RAG index '%s' (%s); falling back to in-memory BM25.",
                path,
                exc,
            )

    chunks = collect_all_chunks(corpus_path=CORPUS_PATH, sources_dir=SOURCES_DIR)
    return HybridRagIndex(chunks)


def get_rag_index(index_path: Path | None = None) -> HybridRagIndex:
    """Return the shared RAG index (SQLite if generated, otherwise in-memory BM25)."""
    target = index_path or resolve_rag_index_path()
    if target.exists() and target.is_file():
        stat = target.stat()
        return _load_cached_index(str(target.resolve()), stat.st_mtime_ns)
    return _load_cached_index("", -1)


def clear_rag_index_cache() -> None:
    """Clear the cached RAG index (useful in tests after rebuilding an index)."""
    _load_cached_index.cache_clear()
