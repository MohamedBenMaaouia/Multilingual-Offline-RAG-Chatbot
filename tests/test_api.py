"""Tests for src/api/routes.py.

Uses fastapi.testclient.TestClient against the FastAPI app. The LLM is
mocked via the ``mock_llm`` fixture (which patches langchain_ollama
ChatOllama.invoke/ainvoke at the class level). Chroma is isolated via
``tmp_chroma``.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.embeddings import get_embeddings
from src.vectorstore import get_collection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ingest_sample_docs(sample_docs):
    coll = get_collection()
    embedder = get_embeddings()
    texts = [d.page_content for d in sample_docs]
    metadatas = [d.metadata for d in sample_docs]
    ids = [d.metadata["chunk_id"] for d in sample_docs]
    vectors = embedder.embed_documents(texts)
    coll.add(ids=ids, embeddings=vectors, metadatas=metadatas, documents=texts)


# ---------------------------------------------------------------------------
# /chat with mocked LLM
# ---------------------------------------------------------------------------


def test_chat_returns_mocked_answer_with_sources(tmp_chroma, sample_docs, mock_llm):
    """End-to-end /chat with real embeddings + retrieval but mocked LLM."""
    _ingest_sample_docs(sample_docs)
    client = TestClient(app)
    resp = client.post("/chat", json={"question": "What does P0420 mean?"})
    assert resp.status_code == 200, resp.text
    body = resp.json()

    # ChatResponse schema shape.
    assert set(body.keys()) >= {"answer", "sources", "latency_ms", "retrieved_k"}
    assert body["answer"] == "Mocked answer (sample.pdf, p.5)."
    assert isinstance(body["sources"], list)
    assert body["retrieved_k"] >= 1
    assert isinstance(body["latency_ms"], int)

    # SourceRef shape: source/page/doc_type/snippet.
    for src in body["sources"]:
        assert set(src.keys()) >= {"source", "page", "doc_type", "snippet"}
        # page is int | None (the sentinel -1 is normalized to None at the API).
        assert src["page"] is None or isinstance(src["page"], int)
        assert src["page"] != -1, "page=-1 sentinel leaked past the API boundary"


def test_chat_empty_collection_returns_canned_refusal(tmp_chroma, mock_llm):
    """If the collection is empty the API short-circuits with the canned line."""
    # tmp_chroma already gave us a fresh empty collection.
    # Just touch it once so it exists.
    _ = get_collection()
    client = TestClient(app)
    resp = client.post("/chat", json={"question": "anything at all"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["answer"] == (
        "I don't have enough information in my knowledge base to answer that reliably."
    )
    assert body["sources"] == []
    assert body["retrieved_k"] == 0


def test_chat_short_question_rejected_by_pydantic(tmp_chroma, mock_llm):
    """ChatRequest.question has min_length=3 — shorter inputs → 422."""
    client = TestClient(app)
    resp = client.post("/chat", json={"question": "hi"})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


def test_health_returns_schema_with_chroma_ok(tmp_chroma):
    """/health never raises — exposes per-component flags."""
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) >= {"status", "chroma_ok", "ollama_ok", "collection_size"}
    assert isinstance(body["chroma_ok"], bool)
    assert isinstance(body["ollama_ok"], bool)
    assert isinstance(body["collection_size"], int)
    assert body["status"] in {"ok", "degraded"}
    # Chroma should be reachable since tmp_chroma created the dir.
    assert body["chroma_ok"] is True
