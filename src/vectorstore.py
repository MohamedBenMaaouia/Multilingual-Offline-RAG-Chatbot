"""ChromaDB client and collection helpers.

Per ARCHITECTURE.md §3:
- PersistentClient at settings.chroma_path
- Cosine similarity via collection metadata hnsw:space=cosine
- Telemetry disabled
- No business logic here — just the singletons.
"""
from __future__ import annotations

from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.config import settings

_client: Optional[chromadb.PersistentClient] = None
_collections: dict[str, "chromadb.Collection"] = {}


def get_client() -> chromadb.PersistentClient:
    """Return a process-wide singleton PersistentClient."""
    global _client
    if _client is None:
        settings.chroma_path.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(
            path=str(settings.chroma_path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    return _client


def get_collection(name: str | None = None) -> "chromadb.Collection":
    """Return (or create) the cosine-similarity collection.

    Cached per collection name so repeated calls don't hit Chroma's metadata
    code path.
    """
    coll_name = name or settings.collection_name
    if coll_name not in _collections:
        client = get_client()
        _collections[coll_name] = client.get_or_create_collection(
            name=coll_name,
            metadata={"hnsw:space": "cosine"},
        )
    return _collections[coll_name]
