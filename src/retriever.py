"""LangChain retriever, wired to the existing Chroma singleton.

Per ARCHITECTURE.md §4 the default is MMR with k=4, fetch_k=20,
lambda_mult=0.5. The retriever is configured from ``settings`` so the
search strategy and its parameters can be swapped via env vars without
code changes.

- ``settings.search_type`` selects ``"mmr"`` (default) or ``"similarity"``.
- ``settings.top_k`` is always honored.
- ``settings.fetch_k`` and ``settings.mmr_lambda`` are only consulted when
  ``search_type == "mmr"``; under similarity search they are intentionally
  ignored (the parameters have no meaning there).

We bind ``langchain_chroma.Chroma`` to the same PersistentClient and
collection name used by the ingest path, so both paths read/write the
same collection.
"""
from __future__ import annotations

from typing import Optional

from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever

from src.config import settings
from src.embeddings import get_embeddings
from src.vectorstore import get_client

_retriever: Optional[VectorStoreRetriever] = None


def get_retriever() -> VectorStoreRetriever:
    """Return a process-wide singleton retriever configured from settings.

    Built per ``settings.search_type``:
      - "mmr":        search_kwargs = {k, fetch_k, lambda_mult}
      - "similarity": search_kwargs = {k}  (fetch_k / lambda_mult ignored)
    """
    global _retriever
    if _retriever is None:
        vectorstore = Chroma(
            client=get_client(),
            collection_name=settings.collection_name,
            embedding_function=get_embeddings(),
        )
        if settings.search_type == "mmr":
            search_kwargs = {
                "k": settings.top_k,
                "fetch_k": settings.fetch_k,
                "lambda_mult": settings.mmr_lambda,
            }
        else:
            # "similarity" — fetch_k and lambda_mult are not applicable here.
            search_kwargs = {"k": settings.top_k}

        _retriever = vectorstore.as_retriever(
            search_type=settings.search_type,
            search_kwargs=search_kwargs,
        )
    return _retriever
