"""FastAPI app entrypoint.

Wires the §5 router behind CORS with a lifespan that warms the
embedding model + Chroma client and probes Ollama. Never crashes
startup if Ollama is down — /health surfaces it and /chat returns 503
lazily.
"""
from __future__ import annotations

import sys
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router
from src.config import settings
from src.embeddings import get_embeddings
from src.vectorstore import get_collection


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm singletons so first /chat doesn't pay the cold-start tax.
    try:
        get_embeddings()
    except Exception as exc:
        print(
            f"WARN: embedding warmup failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
    try:
        get_collection()
    except Exception as exc:
        print(
            f"WARN: chroma warmup failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
    # Probe Ollama but never crash startup on its account.
    try:
        with httpx.Client(timeout=2.0) as client:
            r = client.get(settings.ollama_url.rstrip("/") + "/api/tags")
            if r.status_code != 200:
                print(
                    f"WARN: Ollama probe returned HTTP {r.status_code} at "
                    f"{settings.ollama_url}",
                    file=sys.stderr,
                )
    except Exception as exc:
        print(
            f"WARN: Ollama probe failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )

    yield

    print("API shutting down.")


app = FastAPI(
    title="RAG Chatbot API",
    description="Multilingual offline RAG over the automotive POC corpus.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(router)
