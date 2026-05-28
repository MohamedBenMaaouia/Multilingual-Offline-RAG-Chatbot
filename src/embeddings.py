"""HuggingFace embeddings singleton.

Per ARCHITECTURE.md §3 (embedding model section):
- Model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (384-dim)
- Cosine similarity is set on the Chroma collection — so L2-normalize the
  vectors via encode_kwargs={"normalize_embeddings": True} to keep cosine
  distance well-behaved.
- CPU device by default (laptop-targeted POC).
"""
from __future__ import annotations

from typing import Optional

from langchain_huggingface import HuggingFaceEmbeddings

from src.config import settings

_embeddings: Optional[HuggingFaceEmbeddings] = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """Return a process-wide singleton HuggingFaceEmbeddings instance.

    The MiniLM model (~470 MB) is downloaded on first call and cached under
    the user's HuggingFace cache dir. Subsequent calls reuse the loaded model.
    """
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings
