"""Offline deterministic unit tests for hybrid RAG chunking, RRF ranking, and fallback."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from app.tools.corpus import _search_with_index, search_historical_corpus
from app.tools.rag import (
    CORPUS_PATH,
    SOURCES_DIR,
    EmbeddingError,
    HybridRagIndex,
    RagChunk,
    build_and_save_index,
    build_chunks_from_sources,
    chunk_text,
    clear_rag_index_cache,
    collect_all_chunks,
    cosine_similarity,
    load_rag_index,
    reciprocal_rank_fusion,
    tokenize_for_bm25,
)
from scripts.build_rag_index import main as build_rag_index_main


class _ConceptVectorEmbedder:
    """Deterministic offline embedder mapping pt-BR historical themes to orthogonal vectors.

    Simulates local Ollama embeddings (e.g., nomic-embed-text) without network calls so
    unit tests can verify semantic retrieval for pt-BR queries with zero shared keywords.
    """

    model = "fake-nomic-embed-text"

    _CONCEPT_LEXICONS: tuple[tuple[str, ...], ...] = (
        # Dim 0: Abolition / Slavery / Post-abolition
        (
            "abolicao",
            "escravidao",
            "escravizad",
            "lei aurea",
            "libertos",
            "cativeiro",
            "cativo",
            "alforria",
            "princesa isabel",
            "luiz gama",
            "1888",
            "reparacao",
            "quilombo",
        ),
        # Dim 1: Constitution of 1988 / Indigenous & Citizen Rights
        (
            "constituicao",
            "1988",
            "carta magna",
            "art. 231",
            "artigo 231",
            "direitos originarios",
            "indigenas",
            "povos originarios",
            "ulysses",
            "constituinte",
            "cidada",
            "demarcacao",
        ),
        # Dim 2: Military Dictatorship / AI-5 / Authoritarianism
        (
            "ditadura",
            "ai-5",
            "ato institucional",
            "1968",
            "1964",
            "censura",
            "habeas corpus",
            "anos de chumbo",
            "repressao",
            "regime autoritario",
            "golpe militar",
        ),
        # Dim 3: Vargas Era / CLT / Labor Law
        (
            "vargas",
            "clt",
            "1943",
            "trabalh",
            "salario minimo",
            "ferias",
            "sindicat",
            "estado novo",
            "operari",
        ),
        # Dim 4: Modernism / 1922 / Anthropophagy
        (
            "modern",
            "1922",
            "antropofag",
            "oswald",
            "vanguarda",
            "estetica",
            "theatro municipal",
        ),
        # Dim 5: Colonial Arrival / Carta de Caminha
        (
            "caminha",
            "1500",
            "cabral",
            "porto seguro",
            "tupiniquim",
            "eurocentr",
            "coroa portuguesa",
        ),
    )

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            vec = [0.0] * len(self._CONCEPT_LEXICONS)
            for dim, markers in enumerate(self._CONCEPT_LEXICONS):
                hits = sum(1 for marker in markers if marker in lowered)
                vec[dim] = float(hits)
            vectors.append(vec)
        return vectors


class _FailingEmbedder:
    """Offline stub simulating an unreachable Ollama server."""

    model = "nomic-embed-text"

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        raise EmbeddingError("Simulated offline Ollama connection refused.")


@pytest.fixture(autouse=True)
def _reset_rag_cache() -> None:
    clear_rag_index_cache()
    yield
    clear_rag_index_cache()


def test_chunk_text_splits_on_boundaries_with_overlap() -> None:
    text = (
        "Primeiro parágrafo histórico sobre a chegada dos portugueses em 1500 e o encontro "
        "com os povos indígenas no litoral.\n\n"
        "Segundo parágrafo detalhando o sistema de plantation açucareira no Nordeste colonial "
        "com trabalho escravizado africano e indígena.\n\n"
        "Terceiro parágrafo analisando a resistência quilombola em Palmares e a luta por "
        "autonomia política e cultural."
    )
    chunks = chunk_text(text, chunk_size=140, chunk_overlap=30)
    assert len(chunks) >= 3
    assert all(chunk.strip() for chunk in chunks)
    assert all(len(chunk) <= 145 for chunk in chunks)
    assert "Primeiro parágrafo" in chunks[0]
    assert "Palmares" in " ".join(chunks)
    assert "autonomia política e cultural." in chunks[-1]
    # Verify non-empty overlap between adjacent chunks from the same paragraph/text.
    overlap_found = any(
        any(word in chunks[i + 1] for word in chunks[i].split()[-4:])
        for i in range(len(chunks) - 1)
    )
    assert overlap_found


def test_chunk_text_validates_arguments() -> None:
    assert chunk_text("   ") == []
    assert chunk_text("Texto curto.", chunk_size=100, chunk_overlap=10) == [
        "Texto curto."
    ]
    with pytest.raises(ValueError, match="chunk_size"):
        chunk_text("abc", chunk_size=0)
    with pytest.raises(ValueError, match="chunk_overlap"):
        chunk_text("abc", chunk_size=50, chunk_overlap=50)
    with pytest.raises(ValueError, match="chunk_overlap"):
        chunk_text("abc", chunk_size=50, chunk_overlap=-1)


def test_curated_sources_cover_bncc_cf88_primary_and_enem(tmp_path: Path) -> None:
    chunks = collect_all_chunks(corpus_path=CORPUS_PATH, sources_dir=SOURCES_DIR)
    source_types = {chunk.source_type for chunk in chunks}
    assert {"corpus", "bncc", "constitution", "primary_source", "enem"}.issubset(
        source_types
    )
    for chunk in chunks:
        assert chunk.chunk_id
        assert chunk.source_id
        assert chunk.document
        assert chunk.text.strip()
        assert len(chunk.bncc_codes) >= 1

    # Verify standalone Markdown ingestion as well.
    custom_md = tmp_path / "nota_historica.md"
    custom_md.write_text(
        "# Nota sobre a Lei 11.645/2008\n\nA lei torna obrigatório o estudo da história e cultura afro-brasileira e indígena.",
        encoding="utf-8",
    )
    md_chunks = build_chunks_from_sources(tmp_path)
    assert len(md_chunks) == 1
    assert md_chunks[0].source_id == "doc:nota_historica"
    assert "Lei 11.645/2008" in md_chunks[0].document


def test_cosine_similarity_and_rrf_boost_dual_signal_candidates() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0
    assert cosine_similarity([1.0], [1.0, 2.0]) == 0.0

    fused = reciprocal_rank_fusion(
        bm25_ranks={"chunk_both": 2, "chunk_bm25_only": 1},
        dense_ranks={"chunk_both": 2, "chunk_dense_only": 1},
        k=60,
    )
    assert fused["chunk_both"] > fused["chunk_bm25_only"]
    assert fused["chunk_both"] > fused["chunk_dense_only"]


def test_hybrid_rag_retrieves_semantic_pt_br_query_without_exact_keywords(
    tmp_path: Path,
) -> None:
    embedder = _ConceptVectorEmbedder()
    index_path = tmp_path / "hybrid_index.sqlite3"
    summary = build_and_save_index(
        corpus_path=CORPUS_PATH,
        sources_dir=SOURCES_DIR,
        output_path=index_path,
        embedder=embedder,
        include_embeddings=True,
    )
    assert summary["has_embeddings"] is True
    assert summary["embedding_model"] == "fake-nomic-embed-text"
    assert summary["embedding_dim"] == 6

    loaded_index = load_rag_index(index_path)
    assert loaded_index.has_embeddings is True

    # Semantic pt-BR query using synonyms ("trabalho cativo", "cativeiro") rather than
    # exact "Lei Áurea" / "Abolição" keywords: BM25 alone misses, dense RRF retrieves!
    semantic_query = "cativeiro cativo"
    lexical_only = _search_with_index(
        semantic_query,
        max_results=3,
        index=loaded_index,
        embedder=embedder,
        enable_dense=False,
    )
    assert lexical_only["status"] == "not_found"

    hybrid_result = _search_with_index(
        semantic_query,
        max_results=3,
        index=loaded_index,
        embedder=embedder,
        enable_dense=True,
    )
    assert hybrid_result["status"] == "ok"
    assert hybrid_result["retrieval_mode"] == "hybrid"
    assert any(
        entry["id"] == "abolicao_escravidao_1888" for entry in hybrid_result["entries"]
    )
    assert any(
        chunk["source_id"].startswith("corpus:abolicao_escravidao_1888")
        for chunk in hybrid_result["chunks"]
    )
    for chunk in hybrid_result["chunks"]:
        assert chunk["source_id"]
        assert chunk["document"]
        assert isinstance(chunk["bncc_codes"], list)
        assert chunk["dense_score"] is not None


def test_hybrid_rrf_ranks_joint_lexical_and_semantic_chunk_first() -> None:
    chunks = [
        RagChunk(
            chunk_id="c_lex_only",
            source_id="src:lex",
            document="Documento Lexical",
            source_type="reference",
            topic_id="",
            title="Decreto histórico",
            period="Brasil Império",
            bncc_codes=("EF08HI19",),
            text="Decreto imperial com menção pontual a liberdade.",
            tokens=tuple(
                tokenize_for_bm25("Decreto imperial com menção pontual a liberdade.")
            ),
            title_tokens=tuple(tokenize_for_bm25("Decreto histórico")),
            embedding=(0.0, 1.0),
        ),
        RagChunk(
            chunk_id="c_both",
            source_id="src:both",
            document="Fonte Primária — Lei Áurea (1888)",
            source_type="primary_source",
            topic_id="abolicao_escravidao_1888",
            title="Lei Áurea e liberdade",
            period="Brasil Império",
            bncc_codes=("EF08HI19", "EM13CHS502"),
            text="Decreto da Lei Áurea que extinguiu a escravidão e garantiu liberdade formal.",
            tokens=tuple(
                tokenize_for_bm25(
                    "Decreto da Lei Áurea que extinguiu a escravidão e garantiu liberdade formal."
                )
            ),
            title_tokens=tuple(tokenize_for_bm25("Lei Áurea e liberdade")),
            embedding=(1.0, 0.0),
        ),
        RagChunk(
            chunk_id="c_dense_only",
            source_id="src:dense",
            document="Referência Semântica",
            source_type="reference",
            topic_id="",
            title="Pós-abolição",
            period="Brasil República",
            bncc_codes=("EM13CHS502",),
            text="Análise sociológica das consequências do trabalho cativo.",
            tokens=tuple(
                tokenize_for_bm25(
                    "Análise sociológica das consequências do trabalho cativo."
                )
            ),
            title_tokens=tuple(tokenize_for_bm25("Pós-abolição")),
            embedding=(1.0, 0.0),
        ),
    ]

    class _FixedQueryEmbedder:
        model = "unit-test-embedder"

        def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
            return [[1.0, 0.0] for _ in texts]

    index = HybridRagIndex(chunks, embedding_model="unit-test-embedder")
    result = index.search(
        "decreto liberdade",
        top_k=3,
        embedder=_FixedQueryEmbedder(),
        enable_dense=True,
        min_dense_similarity=0.5,
    )
    assert result.retrieval_mode == "hybrid"
    assert result.chunks[0].chunk.chunk_id == "c_both"
    assert result.chunks[0].bm25_score > 0.0
    assert result.chunks[0].dense_score == pytest.approx(1.0)


def test_graceful_fallback_to_bm25_when_ollama_is_unavailable(tmp_path: Path) -> None:
    index_path = tmp_path / "embedded.sqlite3"
    build_and_save_index(
        corpus_path=CORPUS_PATH,
        sources_dir=SOURCES_DIR,
        output_path=index_path,
        embedder=_ConceptVectorEmbedder(),
        include_embeddings=True,
    )
    loaded = load_rag_index(index_path)

    result = _search_with_index(
        "Constituição de 1988 Art. 231 direitos originários indígenas",
        max_results=3,
        index=loaded,
        embedder=_FailingEmbedder(),
        enable_dense=True,
    )
    assert result["status"] == "ok"
    assert result["retrieval_mode"] == "bm25_lexical"
    assert any(
        chunk["source_id"] == "cf88:art231_indigenas" for chunk in result["chunks"]
    )
    assert all(chunk["dense_score"] is None for chunk in result["chunks"])


def test_build_rag_index_cli_builds_offline_sqlite_index_and_preserves_parity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index_path = tmp_path / "cli_rag_index.sqlite3"
    exit_code = build_rag_index_main(
        [
            "--output",
            str(index_path),
            "--no-embeddings",
            "--chunk-size",
            "600",
            "--chunk-overlap",
            "80",
        ]
    )
    assert exit_code == 0
    assert index_path.exists()

    monkeypatch.setenv("RAG_INDEX_PATH", str(index_path))
    monkeypatch.setenv("RAG_ENABLE_DENSE", "false")
    clear_rag_index_cache()

    monkeypatch.setenv("MODEL_PROFILE", "local")
    local_result = search_historical_corpus(
        "Ato Institucional 5 AI-5 censura habeas corpus 1968",
        max_results=3,
    )
    monkeypatch.setenv("MODEL_PROFILE", "remote")
    remote_result = search_historical_corpus(
        "Ato Institucional 5 AI-5 censura habeas corpus 1968",
        max_results=3,
    )

    # Zero divergence between local and remote profiles.
    assert local_result == remote_result
    assert local_result["status"] == "ok"
    assert local_result["retrieval_mode"] == "bm25_lexical"
    assert local_result["entries"][0]["id"] == (
        "ditadura_militar_e_redemocratizacao_1964_1988"
    )
    assert any(
        chunk["source_id"] == "primary:ai5_1968_ditadura"
        for chunk in local_result["chunks"]
    )
    for entry in local_result["entries"]:
        assert entry["source_id"]
        assert entry["document"]
        assert len(entry["citations"]) >= 1
        for citation in entry["citations"]:
            assert citation["source_id"]
            assert citation["document"]
            assert isinstance(citation["bncc_codes"], list)


def test_fictional_query_returns_not_found_in_hybrid_mode(tmp_path: Path) -> None:
    index_path = tmp_path / "fictional_check.sqlite3"
    embedder = _ConceptVectorEmbedder()
    build_and_save_index(
        corpus_path=CORPUS_PATH,
        sources_dir=SOURCES_DIR,
        output_path=index_path,
        embedder=embedder,
        include_embeddings=True,
    )
    loaded = load_rag_index(index_path)

    result = _search_with_index(
        "Tratado de Guanabara 1742 Imperador do Japao",
        max_results=3,
        index=loaded,
        embedder=embedder,
        enable_dense=True,
    )
    assert result["status"] == "not_found"
    assert result["entries"] == []
    assert result["chunks"] == []
