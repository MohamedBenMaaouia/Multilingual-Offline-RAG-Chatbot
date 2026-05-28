"""Pydantic v2 request/response schemas for the FastAPI layer.

Verbatim from ARCHITECTURE.md §5. Do not add custom validators or extra
fields without updating the architecture doc first.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    language: Literal["auto", "en", "fr", "ar"] = "auto"


class SourceRef(BaseModel):
    source: str
    page: int | None
    doc_type: str
    snippet: str  # first 200 chars of the chunk


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceRef]
    latency_ms: int
    retrieved_k: int


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    chroma_ok: bool
    ollama_ok: bool
    collection_size: int


class IngestRequest(BaseModel):
    wipe_first: bool = False
    paths: list[str] | None = None  # if None, ingest everything in data/raw/


class IngestResponse(BaseModel):
    documents_loaded: int
    chunks_created: int
    duration_seconds: float
