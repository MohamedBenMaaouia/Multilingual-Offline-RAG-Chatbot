"""Chunking strategy.

Per ARCHITECTURE.md §3:
- CSV rows (doc_type='obd_code') are atomic — passed through verbatim,
  only their chunk_id is regenerated to mark them as "chunks".
- Everything else (manual/faq) goes through RecursiveCharacterTextSplitter
  with chunk_size=800, chunk_overlap=150, separators=["\\n\\n", "\\n", ". ", " ", ""].

Every output chunk carries a FRESH chunk_id (uuid4); all other metadata
fields propagate unchanged from the parent document.
"""
from __future__ import annotations

import uuid

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings


_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def chunk_documents(docs: list[Document]) -> list[Document]:
    """Apply the dual chunking strategy and return a flat list of chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=_SEPARATORS,
    )

    out: list[Document] = []
    to_split: list[Document] = []

    for d in docs:
        if d.metadata.get("doc_type") == "obd_code":
            # Atomic — keep verbatim, just give it a fresh chunk_id.
            md = dict(d.metadata)
            md["chunk_id"] = str(uuid.uuid4())
            out.append(Document(page_content=d.page_content, metadata=md))
        else:
            to_split.append(d)

    if to_split:
        split_chunks = splitter.split_documents(to_split)
        for c in split_chunks:
            md = dict(c.metadata)
            md["chunk_id"] = str(uuid.uuid4())
            out.append(Document(page_content=c.page_content, metadata=md))

    return out
