"""FastAPI routes for the RAG chatbot.

Implements the three endpoints from ARCHITECTURE.md §5 (/chat, /health,
/ingest) with the §5 error-handling table.

The page=-1 → None sentinel normalization happens HERE, at the API
boundary, and nowhere else (per project convention in CLAUDE.md).
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

import httpx
from fastapi import APIRouter, HTTPException

from src.api.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    SourceRef,
)
from src.chain import get_chain
from src.config import settings
from src.ingest import ingest as run_ingest
from src.language import language_name, resolve_language
from src.vectorstore import count_chunks

router = APIRouter()

# Canned refusal — must match the wording in the system prompt verbatim.
NO_CONTEXT_ANSWER = (
    "I don't have enough information in my knowledge base to answer that reliably."
)


def _doc_to_source_ref(doc) -> SourceRef:
    """Build a SourceRef from a retrieved Document, normalizing page=-1 → None."""
    meta = doc.metadata or {}
    raw_page = meta.get("page", -1)
    page = None if raw_page == -1 else raw_page
    return SourceRef(
        source=meta.get("source", ""),
        page=page,
        doc_type=meta.get("doc_type", ""),
        snippet=(doc.page_content or "")[:200],
    )


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Submit a question, get a grounded answer with sources."""
    # Cheap pre-check: empty collection → canned refusal, no LLM call.
    try:
        collection_size = count_chunks()
    except Exception:
        print("Vector store error during /chat pre-check:", file=sys.stderr)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Retrieval error")

    if collection_size == 0:
        return ChatResponse(
            answer=NO_CONTEXT_ANSWER,
            sources=[],
            latency_ms=0,
            retrieved_k=0,
        )

    # Resolve the effective language once: an explicit request.language wins,
    # "auto" (the default) detects from the question. It then drives BOTH the
    # retrieval bias and the prompt's answer-language directive.
    lang = resolve_language(request.question, request.language)

    # Invoke the chain, timing only the LLM/retrieval wall-clock.
    t0 = time.perf_counter()
    try:
        result = get_chain(lang).invoke(
            {
                "input": request.question,
                "answer_language": language_name(lang),
            }
        )
    except (httpx.ConnectError, httpx.TimeoutException) as exc:
        print(f"Ollama unreachable: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise HTTPException(status_code=503, detail="LLM unavailable, retry")
    except Exception as exc:
        # langchain-ollama wraps timeouts in its own exception types; rather
        # than enumerate, match by name/message as a defensive net.
        name = type(exc).__name__.lower()
        msg = str(exc).lower()
        if (
            "timeout" in name
            or "timeout" in msg
            or "connect" in name
            or "connection" in msg
            or "ollama" in msg
        ):
            print(f"Ollama call failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            raise HTTPException(status_code=503, detail="LLM unavailable, retry")
        print("Unexpected /chat failure:", file=sys.stderr)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Retrieval error")
    elapsed = time.perf_counter() - t0

    context_docs = result.get("context", []) or []

    # Empty retrieval → canned refusal per §5 error table.
    if len(context_docs) == 0:
        return ChatResponse(
            answer=NO_CONTEXT_ANSWER,
            sources=[],
            latency_ms=int(elapsed * 1000),
            retrieved_k=0,
        )

    sources = [_doc_to_source_ref(d) for d in context_docs]

    return ChatResponse(
        answer=result.get("answer", ""),
        sources=sources,
        latency_ms=int(elapsed * 1000),
        retrieved_k=len(context_docs),
    )


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness probe. Never raises — degradations become False flags."""
    vectorstore_ok = False
    collection_size = 0
    try:
        collection_size = count_chunks()
        vectorstore_ok = True
    except Exception as exc:
        print(f"/health vector store check failed: {type(exc).__name__}: {exc}", file=sys.stderr)

    ollama_ok = False
    try:
        with httpx.Client(timeout=2.0) as client:
            r = client.get(settings.ollama_url.rstrip("/") + "/api/tags")
            ollama_ok = r.status_code == 200
    except Exception as exc:
        print(f"/health ollama check failed: {type(exc).__name__}: {exc}", file=sys.stderr)

    status = "ok" if (vectorstore_ok and ollama_ok) else "degraded"
    return HealthResponse(
        status=status,
        vectorstore_ok=vectorstore_ok,
        ollama_ok=ollama_ok,
        collection_size=collection_size,
    )


@router.post("/ingest", response_model=IngestResponse)
def ingest_endpoint(request: IngestRequest) -> IngestResponse:
    """Trigger re-indexing. Body controls wipe-first and optional path list."""
    paths: list[Path] | None
    if request.paths is not None:
        paths = [Path(p) for p in request.paths]
    else:
        paths = None
    result = run_ingest(paths=paths, wipe_first=request.wipe_first)
    return IngestResponse(**result)
