"""Ingestion orchestrator: loaders -> chunker -> embeddings -> Qdrant.

Per ARCHITECTURE.md §3. Uses the existing singletons in src/vectorstore.py
— does NOT instantiate a second client or embedder. Each chunk is stored
with both its dense (MiniLM) and sparse (BM25) vector.
"""
from __future__ import annotations

import time
from pathlib import Path

from langchain_core.documents import Document

from src import cache as answer_cache
from src import chain as _chain
from src import retriever as _retriever
from src.chunker import chunk_documents
from src.config import settings
from src.loaders import load_csv, load_markdown, load_pdfs
from src.vectorstore import add_chunks, count_chunks, delete_collection, ensure_collection


def _load_all(data_raw: Path) -> list[Document]:
    """Load everything sitting under data/raw/ using the convention from the brief."""
    docs: list[Document] = []
    if data_raw.exists():
        # PDFs anywhere under data/raw/ (recursive)
        docs.extend(load_pdfs(data_raw))

        # Single canonical CSV at data/raw/obd_codes.csv (skip if absent)
        csv_path = data_raw / "obd_codes.csv"
        if csv_path.exists():
            docs.extend(load_csv(csv_path))

        # FAQ markdown under data/raw/faq/ (recursive); harmless if missing
        faq_dir = data_raw / "faq"
        if faq_dir.exists():
            docs.extend(load_markdown(faq_dir))
    return docs


def _load_paths(paths: list[Path]) -> list[Document]:
    """Dispatch a list of explicit paths to the right loader by extension."""
    docs: list[Document] = []
    for p in paths:
        p = Path(p)
        suffix = p.suffix.lower()
        if suffix == ".pdf":
            # load_pdfs expects a directory — use parent + filter is overkill;
            # PyPDFLoader directly handles a single file.
            from langchain_community.document_loaders import PyPDFLoader
            from src.loaders import _rel_source
            import uuid as _uuid
            rel = _rel_source(p)
            for d in PyPDFLoader(str(p)).load():
                page = d.metadata.get("page", -1)
                if not isinstance(page, int):
                    try:
                        page = int(page)
                    except (TypeError, ValueError):
                        page = -1
                docs.append(
                    Document(
                        page_content=d.page_content,
                        metadata={
                            "source": rel,
                            "page": page,
                            "doc_type": "manual",
                            "language": "en",
                            "chunk_id": str(_uuid.uuid4()),
                        },
                    )
                )
        elif suffix == ".csv":
            docs.extend(load_csv(p))
        elif suffix == ".md":
            # Reuse load_markdown by pointing at the file's parent; filter afterwards.
            for d in load_markdown(p.parent):
                if d.metadata.get("source", "").endswith(p.name):
                    docs.append(d)
        # silently skip unknown extensions
    return docs


def ingest(paths: list[Path] | None = None, wipe_first: bool = False) -> dict:
    """Run the full offline pipeline. See module docstring for shape."""
    t0 = time.perf_counter()

    if wipe_first:
        delete_collection()
        # Retrievers and chains hold vector-store handles; rebuild them so a
        # running API keeps working after POST /ingest {"wipe_first": true}.
        _retriever.reset_caches()
        _chain.reset_caches()
    ensure_collection()

    if paths is None:
        docs = _load_all(settings.data_raw_path)
    else:
        docs = _load_paths([Path(p) for p in paths])

    documents_loaded = len(docs)

    chunks = chunk_documents(docs)
    chunks_created = len(chunks)

    if chunks_created == 0:
        return {
            "documents_loaded": documents_loaded,
            "chunks_created": 0,
            "duration_seconds": round(time.perf_counter() - t0, 3),
        }

    add_chunks(chunks)
    # Cached answers were built from the previous corpus.
    answer_cache.invalidate()

    if wipe_first:
        # A wipe must leave exactly this ingest's chunks. A mismatch means
        # the delete did not take (see vectorstore.delete_collection) and
        # every retrieval result would be silently skewed by duplicates.
        stored = count_chunks()
        if stored != chunks_created:
            raise RuntimeError(
                f"Collection holds {stored} chunks after a wipe + ingest of "
                f"{chunks_created}; the old collection was not fully deleted."
            )

    return {
        "documents_loaded": documents_loaded,
        "chunks_created": chunks_created,
        "duration_seconds": round(time.perf_counter() - t0, 3),
    }


if __name__ == "__main__":
    result = ingest(wipe_first=True)
    print(result)
