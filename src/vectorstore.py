"""Qdrant client, collection and vector-store helpers.

One collection holds two vectors per chunk:

- ``dense``  — the MiniLM embedding (384 dims, cosine). Semantic matching,
  cross-lingual.
- ``bm25``   — a sparse BM25 vector (FastEmbed ``Qdrant/bm25``). Exact-term
  matching, which is what identifiers like ``P0420`` need. The sparse
  vector is created with ``modifier=IDF`` so Qdrant applies inverse document
  frequency at query time; without it BM25 degrades to term counting.

Both vectors are always written, so ``settings.retrieval_mode`` can switch
between dense and hybrid search without re-ingesting.

Connection: embedded local mode (``QdrantClient(path=...)``) unless
``settings.qdrant_url`` is set, in which case a Qdrant server is used. The
code is identical either way. Embedded mode locks its folder: only ONE
process may open it at a time, so stop the API before running ingest or
eval from another terminal (or run a server and set QDRANT_URL).

We create the collection ourselves instead of letting langchain-qdrant do
it, for two reasons: langchain-qdrant omits the IDF modifier, and its
factory methods open a second client, which embedded mode's lock forbids.

The ``page=-1`` "no page" sentinel is kept even though Qdrant accepts null:
the API boundary (src/api/routes.py) normalizes -1 to null, unchanged.
"""
from __future__ import annotations

import uuid
from typing import Optional

from langchain_core.documents import Document
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient, models

from src.config import settings
from src.embeddings import get_embeddings

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "bm25"
EMBEDDING_DIM = 384  # paraphrase-multilingual-MiniLM-L12-v2

# Metadata fields filtered on at query time get a payload index (a no-op in
# embedded mode, real indexes on a server).
_INDEXED_FIELDS = ("metadata.language", "metadata.doc_type")

_client: Optional[QdrantClient] = None
_sparse: Optional[FastEmbedSparse] = None
_stores: dict[str, QdrantVectorStore] = {}


def get_client() -> QdrantClient:
    """Return the process-wide Qdrant client (embedded or server)."""
    global _client
    if _client is None:
        if settings.qdrant_url:
            _client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
        else:
            settings.qdrant_path.mkdir(parents=True, exist_ok=True)
            _client = QdrantClient(path=str(settings.qdrant_path))
    return _client


def get_sparse_embeddings() -> FastEmbedSparse:
    """Return the process-wide BM25 sparse encoder."""
    global _sparse
    if _sparse is None:
        _sparse = FastEmbedSparse(model_name=settings.sparse_model)
    return _sparse


def reset_caches() -> None:
    """Close the client and drop every cached handle (tests, re-pointing paths)."""
    global _client
    if _client is not None:
        try:
            _client.close()
        except Exception:  # noqa: BLE001 — best effort on shutdown
            pass
    _client = None
    _stores.clear()


def ensure_collection(name: str | None = None) -> str:
    """Create the collection (dense + BM25 vectors) if it does not exist."""
    coll = name or settings.collection_name
    client = get_client()
    if not client.collection_exists(coll):
        client.create_collection(
            collection_name=coll,
            vectors_config={
                DENSE_VECTOR: models.VectorParams(
                    size=EMBEDDING_DIM, distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={
                SPARSE_VECTOR: models.SparseVectorParams(modifier=models.Modifier.IDF)
            },
        )
        for field in _INDEXED_FIELDS:
            client.create_payload_index(
                collection_name=coll,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
    return coll


def delete_collection(name: str | None = None) -> None:
    """Drop the collection if it exists and forget cached store handles."""
    coll = name or settings.collection_name
    client = get_client()
    if client.collection_exists(coll):
        client.delete_collection(coll)
    _stores.clear()


def count_chunks(name: str | None = None) -> int:
    """Exact number of chunks in the collection (0 if it does not exist yet)."""
    coll = name or settings.collection_name
    client = get_client()
    if not client.collection_exists(coll):
        return 0
    return client.count(collection_name=coll, exact=True).count


def get_vectorstore(mode: str | None = None) -> QdrantVectorStore:
    """LangChain wrapper over the shared client, in ``dense`` or ``hybrid`` mode."""
    mode = mode or settings.retrieval_mode
    if mode not in _stores:
        coll = ensure_collection()
        _stores[mode] = QdrantVectorStore(
            client=get_client(),
            collection_name=coll,
            embedding=get_embeddings(),
            sparse_embedding=get_sparse_embeddings(),
            retrieval_mode=RetrievalMode.HYBRID if mode == "hybrid" else RetrievalMode.DENSE,
            vector_name=DENSE_VECTOR,
            sparse_vector_name=SPARSE_VECTOR,
            distance=models.Distance.COSINE,
            validate_collection_config=False,
        )
    return _stores[mode]


def add_chunks(chunks: list[Document], batch_size: int = 256) -> None:
    """Embed and upsert chunks, writing both the dense and the BM25 vector.

    Point ids are the chunks' ``chunk_id`` (uuid4), so ids stay stable and
    traceable from an API response back to the stored point.
    """
    if not chunks:
        return
    # The hybrid-mode store writes both vectors whatever mode queries use.
    get_vectorstore("hybrid").add_documents(
        chunks,
        ids=[_point_id(c.metadata["chunk_id"]) for c in chunks],
        batch_size=batch_size,
    )


def _point_id(chunk_id: str) -> str:
    """Qdrant point ids must be UUIDs: keep a uuid4 chunk_id as-is, else derive one.

    Ingested chunks always carry a uuid4. Hand-built Documents (tests,
    scripts) may use readable ids like "csv-1"; those map to a stable uuid5.
    """
    try:
        return str(uuid.UUID(chunk_id))
    except (ValueError, TypeError, AttributeError):
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"rag-chunk:{chunk_id}"))
