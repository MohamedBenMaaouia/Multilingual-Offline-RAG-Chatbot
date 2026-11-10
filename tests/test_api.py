"""Tests for src/api/routes.py.

Uses fastapi.testclient.TestClient against the FastAPI app. The LLM is
mocked via the ``mock_llm`` fixture (which patches langchain_ollama
ChatOllama.invoke/ainvoke at the class level). Qdrant is isolated via
``tmp_qdrant``.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.vectorstore import add_chunks, ensure_collection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ingest_sample_docs(sample_docs):
    add_chunks(sample_docs)


# ---------------------------------------------------------------------------
# /chat with mocked LLM
# ---------------------------------------------------------------------------


def test_chat_returns_mocked_answer_with_sources(tmp_qdrant, sample_docs, mock_llm):
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


def test_chat_empty_collection_returns_canned_refusal(tmp_qdrant, mock_llm):
    """If the collection is empty the API short-circuits with the canned line."""
    # tmp_qdrant already gave us a fresh empty store; create the collection.
    ensure_collection()
    client = TestClient(app)
    resp = client.post("/chat", json={"question": "anything at all"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["answer"] == (
        "I don't have enough information in my knowledge base to answer that reliably."
    )
    assert body["sources"] == []
    assert body["retrieved_k"] == 0


def test_chat_short_question_rejected_by_pydantic(tmp_qdrant, mock_llm):
    """ChatRequest.question has min_length=3 — shorter inputs → 422."""
    client = TestClient(app)
    resp = client.post("/chat", json={"question": "hi"})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


def test_health_returns_schema_with_vectorstore_ok(tmp_qdrant):
    """/health never raises — exposes per-component flags."""
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) >= {"status", "vectorstore_ok", "ollama_ok", "collection_size"}
    assert isinstance(body["vectorstore_ok"], bool)
    assert isinstance(body["ollama_ok"], bool)
    assert isinstance(body["collection_size"], int)
    assert body["status"] in {"ok", "degraded"}
    # The vector store should be reachable since tmp_qdrant created the dir.
    assert body["vectorstore_ok"] is True
