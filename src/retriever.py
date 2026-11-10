"""LangChain retriever, wired to the existing Chroma singleton.

Per ARCHITECTURE.md §4 the default is MMR, configured from ``settings`` so
the search strategy and its parameters can be swapped via env vars without
code changes.

- ``settings.search_type`` selects ``"mmr"`` (default) or ``"similarity"``.
- ``settings.top_k`` is always honored.
- ``settings.fetch_k`` and ``settings.mmr_lambda`` are only consulted when
  ``search_type == "mmr"``; under similarity search they are intentionally
  ignored (the parameters have no meaning there).

We bind ``langchain_chroma.Chroma`` to the same PersistentClient and
collection name used by the ingest path, so both paths read/write the
same collection.

Language-aware retrieval
------------------------
``get_retriever(language=...)`` returns a retriever that *prefers* chunks
tagged with that language without excluding the others. A hard metadata
filter was rejected: the embedding model is cross-lingual, and in this
corpus 395 of 470 OBD rows are tagged ``en``, so filtering a French
request to ``language="fr"`` would hide almost the entire code table.

Instead ``settings.language_pref_ratio`` reserves a share of ``top_k`` for
same-language chunks and leaves the rest open to anything. With
``top_k=6`` and ratio 0.67 that is 4 reserved + 2 open.

``get_retriever()`` with no argument returns the plain settings-driven
``VectorStoreRetriever`` exactly as before — the drift-guard test in
``tests/test_retriever.py`` asserts against that object.
"""
from __future__ import annotations

from typing import Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.vectorstores import VectorStoreRetriever

from src.config import settings
from src.embeddings import get_embeddings
from src.vectorstore import get_client

_retriever: Optional[VectorStoreRetriever] = None
_lang_retrievers: dict[str, Runnable] = {}
_vectorstore: Optional[Chroma] = None


def reset_caches() -> None:
    """Drop every cached handle. Used by tests binding a temp Chroma path."""
    global _retriever, _vectorstore
    _retriever = None
    _vectorstore = None
    _lang_retrievers.clear()


def _get_vectorstore() -> Chroma:
    """Return the process-wide Chroma wrapper over the shared client."""
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = Chroma(
            client=get_client(),
            collection_name=settings.collection_name,
            embedding_function=get_embeddings(),
        )
    return _vectorstore


def _search_kwargs() -> dict:
    """Search kwargs matching ``settings.search_type``."""
    if settings.search_type == "mmr":
        return {
            "k": settings.top_k,
            "fetch_k": settings.fetch_k,
            "lambda_mult": settings.mmr_lambda,
        }
    # "similarity" — fetch_k and lambda_mult are not applicable here.
    return {"k": settings.top_k}


def _search(
    query: str,
    k: int,
    metadata_filter: dict | None = None,
) -> list[Document]:
    """Run one search honoring ``settings.search_type``, optionally filtered."""
    vs = _get_vectorstore()
    if settings.search_type == "mmr":
        return vs.max_marginal_relevance_search(
            query,
            k=k,
            fetch_k=settings.fetch_k,
            lambda_mult=settings.mmr_lambda,
            filter=metadata_filter,
        )
    return vs.similarity_search(query, k=k, filter=metadata_filter)


def _merge_unique(*doc_lists: list[Document], limit: int) -> list[Document]:
    """Concatenate document lists in order, de-duplicating, capped at ``limit``.

    Identity is ``chunk_id`` when present (every ingested chunk has one);
    falls back to the page content so ad-hoc Documents still de-duplicate.
    """
    merged: list[Document] = []
    seen: set = set()
    for docs in doc_lists:
        for d in docs:
            key = (d.metadata or {}).get("chunk_id") or d.page_content
            if key in seen:
                continue
            seen.add(key)
            merged.append(d)
            if len(merged) >= limit:
                return merged
    return merged


def _extract_query(payload) -> str:
    """Normalize the two ways this retriever gets called.

    ``create_retrieval_chain`` unwraps ``inputs["input"]`` only for objects
    that are ``BaseRetriever`` instances; anything else (our Runnable) is
    handed the whole input dict. Direct callers and tests pass a plain
    string. Accept both rather than depending on that internal detail.
    """
    if isinstance(payload, dict):
        return payload.get("input") or payload.get("query") or ""
    return payload or ""


def _language_preferring_search(payload, language: str) -> list[Document]:
    """Fill a share of top_k with ``language`` chunks, the rest with anything.

    Two searches, then merge preferring the language-matched list. The
    unfiltered pass is what keeps cross-language content (English-only code
    tables, for example) reachable from a non-English question.
    """
    query = _extract_query(payload)
    k = settings.top_k
    n_pref = max(1, min(k, round(k * settings.language_pref_ratio)))
    preferred = _search(query, k=n_pref, metadata_filter={"language": language})
    general = _search(query, k=k, metadata_filter=None)
    return _merge_unique(preferred, general, limit=k)


def get_retriever(language: str | None = None) -> Runnable:
    """Return a cached retriever, optionally biased toward ``language``.

    ``language=None`` (default) yields the plain settings-driven
    ``VectorStoreRetriever`` — unchanged behavior, and the object the
    configuration drift test inspects.

    A language code yields a ``Runnable`` implementing the preference
    strategy described in the module docstring. Both are cached; the
    expensive pieces (Chroma client, embedding model) are singletons, so a
    per-language entry costs almost nothing.
    """
    global _retriever

    if language is None or settings.language_pref_ratio <= 0:
        if _retriever is None:
            _retriever = _get_vectorstore().as_retriever(
                search_type=settings.search_type,
                search_kwargs=_search_kwargs(),
            )
        return _retriever

    if language not in _lang_retrievers:
        _lang_retrievers[language] = RunnableLambda(
            lambda q, _lang=language: _language_preferring_search(q, _lang),
            name=f"language_preferring_retriever[{language}]",
        )
    return _lang_retrievers[language]
