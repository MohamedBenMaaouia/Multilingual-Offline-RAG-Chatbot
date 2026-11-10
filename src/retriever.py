"""Settings-driven retriever over the shared Qdrant collection.

``get_retriever(language)`` returns a ``RagRetriever`` whose fields are a
snapshot of ``settings`` taken when it is built. The drift-guard test in
``tests/test_retriever.py`` asserts those fields against ``settings``, so a
refactor cannot silently run a different strategy than the one configured
(the Day 2-5 bug: MMR configured, similarity executed).

Search strategy (``settings.retrieval_mode``)
----------------------------------------------
- ``dense``: MiniLM vectors only, ranked by ``search_type`` — MMR (default)
  or plain similarity.
- ``hybrid``: dense + BM25 searched together and fused with Reciprocal Rank
  Fusion inside Qdrant (one query, two prefetches). BM25 matches exact
  identifiers such as ``P0420`` that the paraphrase-trained embedding model
  cannot tell apart. MMR does not apply here: langchain-qdrant's MMR uses
  the dense vector only, so honoring ``search_type="mmr"`` in hybrid mode
  would quietly drop BM25.

Language preference
-------------------
``get_retriever(language="fr")`` *prefers* chunks tagged with that language
without excluding the others. A hard filter was rejected: the embedding
model is cross-lingual, and 395 of 470 OBD rows are tagged ``en``, so
filtering a French request to ``fr`` would hide almost the whole code
table. ``settings.language_pref_ratio`` reserves that share of ``top_k``
for same-language chunks (6 x 0.67 = 4) and leaves the rest open.
"""
from __future__ import annotations

from typing import Literal, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from qdrant_client import models

from src.config import settings
from src.language import DEFAULT_LANGUAGE
from src.vectorstore import get_vectorstore

_retrievers: dict[Optional[str], "RagRetriever"] = {}


def reset_caches() -> None:
    """Drop every cached retriever. Used by tests binding a temp collection."""
    _retrievers.clear()


def _language_filter(language: str | None) -> models.Filter | None:
    if not language:
        return None
    return models.Filter(
        must=[
            models.FieldCondition(
                key="metadata.language", match=models.MatchValue(value=language)
            )
        ]
    )


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


class RagRetriever(BaseRetriever):
    """Dense or hybrid search, with an optional same-language preference."""

    language: Optional[str] = None
    retrieval_mode: Literal["dense", "hybrid"]
    search_type: Literal["mmr", "similarity"]
    top_k: int
    fetch_k: int
    mmr_lambda: float
    language_pref_ratio: float
    rerank: bool = False
    bm25_per_language: bool = True

    def _bm25_language(self) -> str:
        """Tokenization rules for the BM25 half of hybrid search."""
        return self.language if self.bm25_per_language and self.language else DEFAULT_LANGUAGE

    def _n_preferred(self) -> int:
        """Slots of top_k reserved for the request language (0 = no preference)."""
        if not self.language or self.language_pref_ratio <= 0:
            return 0
        return max(1, min(self.top_k, round(self.top_k * self.language_pref_ratio)))

    def _search(self, query: str, k: int, language: str | None = None) -> list[Document]:
        """One search honoring the configured strategy, optionally language-filtered."""
        vs = get_vectorstore(self.retrieval_mode, self._bm25_language())
        qfilter = _language_filter(language)
        if self.retrieval_mode == "hybrid":
            return vs.similarity_search(query, k=k, filter=qfilter)
        if self.search_type == "mmr":
            return vs.max_marginal_relevance_search(
                query,
                k=k,
                fetch_k=self.fetch_k,
                lambda_mult=self.mmr_lambda,
                filter=qfilter,
            )
        return vs.similarity_search(query, k=k, filter=qfilter)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        if self.rerank:
            return self._reranked(query)
        n_pref = self._n_preferred()
        if not n_pref:
            return self._search(query, self.top_k)
        # Two searches, merged preferring the language-matched list. The
        # unfiltered pass keeps cross-language content (English-only code
        # rows, for example) reachable from a non-English question.
        preferred = self._search(query, n_pref, language=self.language)
        general = self._search(query, self.top_k)
        return _merge_unique(preferred, general, limit=self.top_k)

    def _reranked(self, query: str) -> list[Document]:
        """Wide candidate pool -> cross-encoder order -> language quota -> top_k."""
        from src.reranker import rerank

        candidates = self._search(query, self.fetch_k)
        n_pref = self._n_preferred()
        if n_pref:
            # Same-language candidates searched separately so they are in the
            # pool even when the other language dominates the open search.
            same_lang = self._search(query, self.fetch_k, language=self.language)
            candidates = _merge_unique(same_lang, candidates, limit=2 * self.fetch_k)
        ranked = rerank(query, candidates)
        if not n_pref:
            return ranked[: self.top_k]
        # Same quota semantics as the unreranked path: the best n_pref chunks
        # in the request language first, then the best of the rest.
        preferred = [d for d in ranked if (d.metadata or {}).get("language") == self.language]
        return _merge_unique(preferred[:n_pref], ranked, limit=self.top_k)


def get_retriever(language: str | None = None) -> RagRetriever:
    """Return a cached retriever, optionally biased toward ``language``.

    With no language (or ``language_pref_ratio <= 0``) every caller shares
    the same unbiased retriever. Each language gets its own cached instance;
    the expensive pieces (Qdrant client, embedding models) are singletons,
    so an extra entry costs almost nothing.
    """
    key = language if language and settings.language_pref_ratio > 0 else None
    if key not in _retrievers:
        _retrievers[key] = RagRetriever(
            language=key,
            retrieval_mode=settings.retrieval_mode,
            search_type=settings.search_type,
            top_k=settings.top_k,
            fetch_k=settings.fetch_k,
            mmr_lambda=settings.mmr_lambda,
            language_pref_ratio=settings.language_pref_ratio,
            rerank=settings.reranker_enabled,
            bm25_per_language=settings.bm25_per_language,
        )
    return _retrievers[key]
