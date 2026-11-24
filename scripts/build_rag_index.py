#!/usr/bin/env python3
"""Offline ingestion and indexing CLI for Brasão AI's hybrid RAG SQLite store."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.tools.rag import (  # noqa: E402
    CORPUS_PATH,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OLLAMA_API_BASE,
    DEFAULT_OLLAMA_EMBED_MODEL,
    DEFAULT_RAG_INDEX_PATH,
    SOURCES_DIR,
    EmbeddingError,
    OllamaEmbeddingClient,
    build_and_save_index,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for building the offline SQLite hybrid RAG index."""
    load_dotenv(override=False)
    parser = argparse.ArgumentParser(
        description=(
            "Build Brasão AI's offline SQLite hybrid RAG index (BM25 + local Ollama embeddings) "
            "over curated Brazilian History corpus and BNCC/CF88/ENEM/primary source documents."
        )
    )
    parser.add_argument(
        "--corpus-path",
        type=Path,
        default=CORPUS_PATH,
        help="Path to brazil_history_corpus.json.",
    )
    parser.add_argument(
        "--sources-dir",
        type=Path,
        default=SOURCES_DIR,
        help="Directory containing curated BNCC, Constitution, primary source, and ENEM files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(os.getenv("RAG_INDEX_PATH", str(DEFAULT_RAG_INDEX_PATH))),
        help="Destination SQLite file path (gitignored generated artifact).",
    )
    parser.add_argument(
        "--ollama-base-url",
        default=os.getenv("OLLAMA_API_BASE", DEFAULT_OLLAMA_API_BASE),
        help="Local Ollama server base URL.",
    )
    parser.add_argument(
        "--embed-model",
        default=os.getenv("OLLAMA_EMBED_MODEL", DEFAULT_OLLAMA_EMBED_MODEL),
        help="Local Ollama embedding model (e.g., nomic-embed-text or bge-m3).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Maximum character length per indexed chunk.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help="Character overlap between consecutive chunks.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Number of chunks embedded per Ollama API batch.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP timeout in seconds for Ollama embedding requests during indexing.",
    )
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Skip Ollama dense embeddings and build a 100% offline BM25-only SQLite index.",
    )
    parser.add_argument(
        "--allow-lexical-fallback",
        action="store_true",
        help="Build a BM25-only SQLite index if Ollama is unreachable instead of exiting with error.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run offline ingestion, chunking, optional Ollama embedding, and SQLite persistence."""
    args = parse_args(argv)
    embedder = None
    if not args.no_embeddings:
        embedder = OllamaEmbeddingClient(
            model=args.embed_model,
            base_url=args.ollama_base_url,
            timeout_s=args.timeout,
        )

    try:
        summary = build_and_save_index(
            corpus_path=args.corpus_path,
            sources_dir=args.sources_dir,
            output_path=args.output,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            embedder=embedder,
            include_embeddings=not args.no_embeddings,
            allow_lexical_fallback=args.allow_lexical_fallback,
            batch_size=args.batch_size,
        )
    except EmbeddingError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print(
            f"Hint: start Ollama and run `ollama pull {args.embed_model}`, "
            "or pass `--no-embeddings` / `--allow-lexical-fallback` for lexical-only indexing.",
            file=sys.stderr,
        )
        return 1

    mode = (
        f"hybrid (BM25 + {summary['embedding_model']}, dim={summary['embedding_dim']})"
        if summary["has_embeddings"]
        else "bm25_lexical (no dense embeddings)"
    )
    print(f"Built RAG index: {summary['index_path']}")
    print(f"Indexed chunks:  {summary['chunk_count']}")
    print(f"Retrieval mode:  {mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
