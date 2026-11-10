"""Tests for src/retriever.py and the Qdrant collection it searches.

Three flavors:

1. Configuration drift guards — the retriever runs exactly the strategy
   ``settings`` describes (the Day 2-5 bug: MMR configured, similarity run).

2. Collection layout — both vectors exist and BM25 has the IDF modifier.

3. End-to-end retrieval — sample Documents in a temp embedded Qdrant, real
   MiniLM + BM25 encoders, real queries.
"""
from __future__ import annotations

import pytest
from langchain_core.documents import Document
from qdrant_client import models

from src.config import settings
from src.retriever import get_retriever
from src.vectorstore import (
    DENSE_VECTOR,
    SPARSE_VECTOR,
    add_chunks,
    ensure_collection,
    get_client,
)


# ---------------------------------------------------------------------------
# Configuration drift guards
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["dense", "hybrid"])
def test_retriever_honors_settings(tmp_qdrant, monkeypatch, mode):
    """Every strategy field of the retriever mirrors settings."""
    monkeypatch.setattr(settings, "retrieval_mode", mode)
    r = get_retriever()
    assert r.retrieval_mode == settings.retrieval_mode
    assert r.search_type == settings.search_type
    assert r.top_k == settings.top_k
    assert r.fetch_k == settings.fetch_k
    assert r.mmr_lambda == settings.mmr_lambda
    assert r.language_pref_ratio == settings.language_pref_ratio


def test_hybrid_mode_never_runs_mmr(tmp_qdrant, monkeypatch, mocker):
    """MMR is dense-only in langchain-qdrant; hybrid must use fused search.

    Running MMR under retrieval_mode="hybrid" would silently drop BM25 —
    the same class of drift as the original similarity-vs-MMR bug.
    """
    monkeypatch.setattr(settings, "retrieval_mode", "hybrid")
    monkeypatch.setattr(settings, "search_type", "mmr")
    from src import vectorstore

    store = vectorstore.get_vectorstore("hybrid")
    mmr = mocker.spy(store, "max_marginal_relevance_search")
    sim = mocker.spy(store, "similarity_search")
    get_retriever().invoke("P0420")
    assert mmr.call_count == 0
    assert sim.call_count >= 1


# ---------------------------------------------------------------------------
# Collection layout
# ---------------------------------------------------------------------------


def test_collection_has_dense_and_idf_weighted_bm25_vectors(tmp_qdrant):
    """BM25 without the IDF modifier degrades to term counting."""
    coll = ensure_collection()
    params = get_client().get_collection(coll).config.params
    assert params.vectors[DENSE_VECTOR].size == 384
    assert params.vectors[DENSE_VECTOR].distance == models.Distance.COSINE
    assert params.sparse_vectors[SPARSE_VECTOR].modifier == models.Modifier.IDF


# ---------------------------------------------------------------------------
# End-to-end retrieval with real encoders + temp Qdrant
# ---------------------------------------------------------------------------


def test_retrieval_finds_obd_chunk_for_code_query(tmp_qdrant, sample_docs):
    """A 'P0420' query should surface the OBD CSV chunk near the top."""
    add_chunks(sample_docs)
    hits = get_retriever().invoke("What does P0420 mean?")
    assert len(hits) >= 1
    top_sources = [h.metadata.get("source") for h in hits[:2]]
    assert "obd_codes.csv" in top_sources, (
        f"P0420 query did not return the OBD chunk in top-2; got {top_sources}"
    )


def test_retrieval_finds_manual_chunk_for_engine_oil_query(tmp_qdrant, sample_docs):
    """An 'engine oil' query should surface the PDF chunk near the top."""
    add_chunks(sample_docs)
    hits = get_retriever().invoke("How often should I change the engine oil?")
    assert len(hits) >= 1
    top_sources = [h.metadata.get("source") for h in hits[:2]]
    assert "manuals/sample.pdf" in top_sources, (
        f"engine oil query did not return the manual chunk in top-2; got {top_sources}"
    )


@pytest.fixture
def similar_code_rows() -> list[Document]:
    """Eight OBD rows with near-identical wording — dense vectors can't tell them apart."""
    rows = [
        ("P0440", "Evaporative Emission Control System Malfunction"),
        ("P0441", "Evaporative Emission Control System Incorrect Purge Flow"),
        ("P0442", "Evaporative Emission Control System Leak Detected (small leak)"),
        ("P0443", "Evaporative Emission Control System Purge Control Valve Circuit"),
        ("P0446", "Evaporative Emission Control System Vent Control Circuit"),
        ("P0451", "Evaporative Emission Control System Pressure Sensor Range"),
        ("P0455", "Evaporative Emission Control System Leak Detected (gross leak)"),
        ("P0456", "Evaporative Emission Control System Leak Detected (very small leak)"),
    ]
    return [
        Document(
            page_content=f"code: {code}\ndescription: {desc}",
            metadata={
                "source": "obd_codes.csv",
                "page": -1,
                "doc_type": "obd_code",
                "language": "en",
                "chunk_id": f"row-{code}",
            },
        )
        for code, desc in rows
    ]


def _codes(hits: list[Document]) -> list[str]:
    return [h.page_content.splitlines()[0].removeprefix("code: ") for h in hits]


def test_dense_alone_misses_exact_code(tmp_qdrant, monkeypatch, similar_code_rows):
    """The failure hybrid search fixes: MiniLM sees the eight rows as near-identical.

    With top_k=6 over 8 rows, dense search leaves P0451 out entirely.
    """
    monkeypatch.setattr(settings, "retrieval_mode", "dense")
    monkeypatch.setattr(settings, "search_type", "similarity")
    add_chunks(similar_code_rows)
    assert "P0451" not in _codes(get_retriever().invoke("What does P0451 mean?"))


def test_hybrid_brings_exact_code_to_the_top(tmp_qdrant, monkeypatch, similar_code_rows):
    """BM25 matches the identifier exactly, so hybrid puts it in the top 2.

    Top 2, not top 1: RRF scores each list's #1 equally (1/(1+rank)), so
    BM25's exact match ties with dense search's best guess. Picking the
    single best chunk is the cross-encoder reranker's job.
    """
    monkeypatch.setattr(settings, "retrieval_mode", "hybrid")
    add_chunks(similar_code_rows)
    for code in ("P0455", "P0442", "P0451"):
        top = _codes(get_retriever().invoke(f"What does {code} mean?"))[:2]
        assert code in top, f"{code} not in the top 2; got {top}"
