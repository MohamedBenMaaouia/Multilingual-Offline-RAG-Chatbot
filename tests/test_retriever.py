"""Tests for src/retriever.py.

Two flavors:

1. Configuration drift test (does NOT touch Chroma) — guards against the
   Day 2-Day 5 silent regression where the retriever ran similarity even
   though settings.search_type was "mmr".

2. End-to-end retrieval tests — ingest 3 sample Documents into a temp
   Chroma collection, run real queries, verify the obviously-relevant
   chunk is in top-k. Uses the real MiniLM embedder (CPU, acceptable).
"""
from __future__ import annotations

import uuid

import pytest

from src.config import settings
from src.embeddings import get_embeddings
from src.retriever import get_retriever
from src.vectorstore import get_collection


# ---------------------------------------------------------------------------
# Configuration drift guard
# ---------------------------------------------------------------------------


def test_retriever_honors_settings_search_type(tmp_chroma):
    """The fix for the Day 2-Day 5 retriever drift.

    Assert search_type and search_kwargs reflect Settings exactly. If a
    refactor silently flips search_type or drops fetch_k/lambda_mult, this
    test fails before CI passes.
    """
    r = get_retriever()
    assert r.search_type == settings.search_type
    if settings.search_type == "mmr":
        assert r.search_kwargs.get("fetch_k") == settings.fetch_k
        assert r.search_kwargs.get("lambda_mult") == settings.mmr_lambda
    assert r.search_kwargs["k"] == settings.top_k


# ---------------------------------------------------------------------------
# End-to-end retrieval with real embedder + temp Chroma
# ---------------------------------------------------------------------------


def _ingest_sample_docs(sample_docs):
    """Embed the 3 sample_docs into the (temp) collection."""
    collection = get_collection()
    embedder = get_embeddings()
    texts = [d.page_content for d in sample_docs]
    metadatas = [d.metadata for d in sample_docs]
    ids = [d.metadata["chunk_id"] for d in sample_docs]
    vectors = embedder.embed_documents(texts)
    collection.add(
        ids=ids,
        embeddings=vectors,
        metadatas=metadatas,
        documents=texts,
    )


def test_retrieval_finds_obd_chunk_for_code_query(tmp_chroma, sample_docs):
    """A 'P0420' query should surface the OBD CSV chunk near the top."""
    _ingest_sample_docs(sample_docs)
    r = get_retriever()
    hits = r.invoke("What does P0420 mean?")
    assert len(hits) >= 1
    # OBD chunk should be in the top-2 (MMR may diversify with 3 docs total).
    top_sources = [h.metadata.get("source") for h in hits[:2]]
    assert "obd_codes.csv" in top_sources, (
        f"P0420 query did not return the OBD chunk in top-2; got {top_sources}"
    )


def test_retrieval_finds_manual_chunk_for_engine_oil_query(tmp_chroma, sample_docs):
    """An 'engine oil' query should surface the PDF chunk near the top."""
    _ingest_sample_docs(sample_docs)
    r = get_retriever()
    hits = r.invoke("How often should I change the engine oil?")
    assert len(hits) >= 1
    top_sources = [h.metadata.get("source") for h in hits[:2]]
    assert "manuals/sample.pdf" in top_sources, (
        f"engine oil query did not return the manual chunk in top-2; got {top_sources}"
    )
