"""Cross-encoder reranker: picks the best chunks out of the retrieved candidates.

The embedding model is a *bi-encoder*: it encodes the question and each
chunk separately and compares the vectors — fast enough to search the whole
collection, but coarse. A *cross-encoder* reads the question and one chunk
together and outputs a single relevance score — much more precise, far too
slow to run over 4,556 chunks. So retrieval casts a wide net (fetch_k
candidates from hybrid search) and the cross-encoder ranks that short list.

Model: ``settings.reranker_model``, by default
``cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`` — multilingual (trained on
mMARCO, 14 languages incl. EN/FR/AR), ~470 MB, CPU-friendly.
"""
from __future__ import annotations

from typing import Optional

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from src.config import settings

_model: Optional[CrossEncoder] = None


def get_reranker() -> CrossEncoder:
    """Return the process-wide cross-encoder (downloaded on first use)."""
    global _model
    if _model is None:
        _model = CrossEncoder(settings.reranker_model, device="cpu", max_length=512)
    return _model


def rerank(query: str, docs: list[Document]) -> list[Document]:
    """Return ``docs`` sorted by cross-encoder relevance to ``query``, best first.

    Each returned Document carries its score in ``metadata["rerank_score"]``
    (useful when inspecting eval rows; the API response does not expose it).
    """
    if not docs:
        return []
    scores = get_reranker().predict(
        [(query, d.page_content) for d in docs],
        batch_size=16,
        show_progress_bar=False,
    )
    for d, s in zip(docs, scores):
        d.metadata = {**(d.metadata or {}), "rerank_score": float(s)}
    return [d for d, _ in sorted(zip(docs, scores), key=lambda p: p[1], reverse=True)]
