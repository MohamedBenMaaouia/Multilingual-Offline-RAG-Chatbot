"""Qdrant client, collection and vector-store helpers.

One collection holds, per chunk, one dense vector and one sparse BM25 vector
per supported language:

- ``dense``    — the MiniLM embedding (384 dims, cosine). Semantic matching,
  cross-lingual.
- ``bm25_en``, ``bm25_fr`` — sparse BM25 vectors (FastEmbed ``Qdrant/bm25``)
  of the SAME text, tokenized with English and French rules respectively
  (stemmer + stopword list). Exact-term matching, which identifiers like
  ``P0420`` need. A query is encoded with its own language's rules and
  matched against that language's vector, so both sides of the comparison
  are always tokenized the same way. Identifiers tokenize identically under
  both. Each sparse vector has ``modifier=IDF`` so Qdrant applies inverse
  document frequency at query time; without it BM25 degrades to term counting.

All vectors are always written, so ``settings.retrieval_mode`` can switch
between dense and hybrid search without re-ingesting. Adding a language is
one entry in ``_STEMMER_LANGUAGE`` plus a re-ingest.

Connection: embedded local mode (``QdrantClient(path=...)``) unless
``settings.qdrant_url`` is set, in which case a Qdrant server is used. The
code is identical either way. Embedded mode locks its folder: only ONE
process may open it at a time, so stop the API before running ingest or
eval from another terminal (or run a server and set QDRANT_URL).

We create the collection and write points ourselves instead of letting
langchain-qdrant do it: it omits the IDF modifier, writes only one sparse
vector per point, and its factory methods open a second client, which
embedded mode's lock forbids. Queries still go through ``QdrantVectorStore``.

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
from src.language import DEFAULT_LANGUAGE

DENSE_VECTOR = "dense"
EMBEDDING_DIM = 384  # paraphrase-multilingual-MiniLM-L12-v2

# Language code -> FastEmbed BM25 language (stemmer + stopwords).
_STEMMER_LANGUAGE = {"en": "english", "fr": "french"}

# Payload keys langchain-qdrant reads documents back from.
_CONTENT_KEY = "page_content"
_METADATA_KEY = "metadata"

# Metadata fields filtered on at query time get a payload index (a no-op in
# embedded mode, real indexes on a server).
_INDEXED_FIELDS = ("metadata.language", "metadata.doc_type")

_client: Optional[QdrantClient] = None
_sparse: dict[str, FastEmbedSparse] = {}
_stores: dict[tuple[str, str], QdrantVectorStore] = {}


def sparse_vector_name(language: str) -> str:
    """Name of the BM25 vector for ``language`` (unsupported -> default language)."""
    lang = language if language in _STEMMER_LANGUAGE else DEFAULT_LANGUAGE
    return f"bm25_{lang}"


def sparse_vector_names() -> list[str]:
    return [sparse_vector_name(lang) for lang in _STEMMER_LANGUAGE]


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


def get_sparse_embeddings(language: str = DEFAULT_LANGUAGE) -> FastEmbedSparse:
    """Return the process-wide BM25 encoder for ``language``."""
    lang = language if language in _STEMMER_LANGUAGE else DEFAULT_LANGUAGE
    if lang not in _sparse:
        _sparse[lang] = FastEmbedSparse(
            model_name=settings.sparse_model, language=_STEMMER_LANGUAGE[lang]
        )
    return _sparse[lang]


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
    """Create the collection (dense + one BM25 vector per language) if missing.

    An existing collection built with an older layout (e.g. a single
    ``bm25`` vector) is rejected with a clear message instead of failing
    later on a missing vector name.
    """
    coll = name or settings.collection_name
    client = get_client()
    if client.collection_exists(coll):
        have = set((client.get_collection(coll).config.params.sparse_vectors or {}).keys())
        missing = set(sparse_vector_names()) - have
        if missing:
            raise RuntimeError(
                f"Collection '{coll}' lacks sparse vectors {sorted(missing)} (built by an "
                "older version). Re-ingest: python -m src.ingest"
            )
        return coll

    client.create_collection(
        collection_name=coll,
        vectors_config={
            DENSE_VECTOR: models.VectorParams(size=EMBEDDING_DIM, distance=models.Distance.COSINE)
        },
        sparse_vectors_config={
            name: models.SparseVectorParams(modifier=models.Modifier.IDF)
            for name in sparse_vector_names()
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


def get_vectorstore(mode: str | None = None, language: str = DEFAULT_LANGUAGE) -> QdrantVectorStore:
    """LangChain wrapper for querying, in ``dense`` or ``hybrid`` mode.

    In hybrid mode ``language`` picks the BM25 vector (and the query
    tokenizer); dense mode ignores it.
    """
    mode = mode or settings.retrieval_mode
    lang = language if language in _STEMMER_LANGUAGE else DEFAULT_LANGUAGE
    key = (mode, lang if mode == "hybrid" else DEFAULT_LANGUAGE)
    if key not in _stores:
        coll = ensure_collection()
        _stores[key] = QdrantVectorStore(
            client=get_client(),
            collection_name=coll,
            embedding=get_embeddings(),
            sparse_embedding=get_sparse_embeddings(key[1]),
            retrieval_mode=RetrievalMode.HYBRID if mode == "hybrid" else RetrievalMode.DENSE,
            vector_name=DENSE_VECTOR,
            sparse_vector_name=sparse_vector_name(key[1]),
            distance=models.Distance.COSINE,
            validate_collection_config=False,
        )
    return _stores[key]


def add_chunks(chunks: list[Document], batch_size: int = 256) -> None:
    """Embed and upsert chunks with their dense vector and every BM25 vector.

    Point ids are the chunks' ``chunk_id`` (uuid4), so ids stay stable and
    traceable from an API response back to the stored point.
    """
    if not chunks:
        return
    coll = ensure_collection()
    client = get_client()
    dense = get_embeddings()
    encoders = {sparse_vector_name(lang): get_sparse_embeddings(lang) for lang in _STEMMER_LANGUAGE}

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        texts = [c.page_content for c in batch]
        dense_vecs = dense.embed_documents(texts)
        sparse_vecs = {name: enc.embed_documents(texts) for name, enc in encoders.items()}
        points = []
        for i, chunk in enumerate(batch):
            vector: dict = {DENSE_VECTOR: dense_vecs[i]}
            for name, vecs in sparse_vecs.items():
                vector[name] = models.SparseVector(indices=vecs[i].indices, values=vecs[i].values)
            points.append(
                models.PointStruct(
                    id=_point_id(chunk.metadata["chunk_id"]),
                    vector=vector,
                    payload={_CONTENT_KEY: chunk.page_content, _METADATA_KEY: chunk.metadata},
                )
            )
        client.upsert(collection_name=coll, points=points)


def _point_id(chunk_id: str) -> str:
    """Qdrant point ids must be UUIDs: keep a uuid4 chunk_id as-is, else derive one.

    Ingested chunks always carry a uuid4. Hand-built Documents (tests,
    scripts) may use readable ids like "csv-1"; those map to a stable uuid5.
    """
    try:
        return str(uuid.UUID(chunk_id))
    except (ValueError, TypeError, AttributeError):
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"rag-chunk:{chunk_id}"))
