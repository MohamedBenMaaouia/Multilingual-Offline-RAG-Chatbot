"""Document loaders for the three POC source types.

Per ARCHITECTURE.md §3:
- PDFs → PyPDFLoader, one Document per page (page metadata preserved).
- OBD CSV → CSVLoader, one Document per row (atomic, will bypass the chunker).
- FAQ MD → UnstructuredMarkdownLoader.

All loaders normalize metadata to the project schema:
    {source, page, doc_type, language, chunk_id}
where `source` is RELATIVE to settings.data_raw_path and `page` is `-1`
sentinel when the source type has no page concept (CSV, MD). `None` is
NEVER written into metadata — Chroma rejects it.

Language is auto-detected per-Document via `_detect_language` (langdetect)
on the first 500 chars of page_content. Defaults to 'en' on any error.
"""
from __future__ import annotations

import uuid
from pathlib import Path

from langchain_community.document_loaders import (
    CSVLoader,
    PyPDFLoader,
    UnstructuredMarkdownLoader,
)
from langchain_core.documents import Document

from src.config import settings
from src.language import INGEST_CONFIDENCE, detect_language


def _detect_language(
    text: str,
    default: str = "en",
    confidence_threshold: float = INGEST_CONFIDENCE,
) -> str:
    """Thin wrapper over ``src.language.detect_language`` for the ingest path.

    The implementation lives in ``src/language.py`` so the ingestion and
    query paths cannot drift apart on what counts as 'French'.
    """
    return detect_language(
        text, default=default, confidence_threshold=confidence_threshold
    )


def _rel_source(path: Path) -> str:
    """Return path relative to settings.data_raw_path (POSIX-style)."""
    try:
        rel = path.resolve().relative_to(settings.data_raw_path.resolve())
    except ValueError:
        # Path is outside data_raw_path — fall back to its name only.
        rel = Path(path.name)
    return rel.as_posix()


def load_pdfs(dir: Path) -> list[Document]:
    """Recursively glob ``**/*.pdf`` under ``dir`` and load with PyPDFLoader.

    PyPDFLoader yields one Document per page; we overwrite its metadata with
    the project schema, preserving the page index it reported. Language is
    auto-detected per page.

    Missing or empty ``dir`` returns ``[]`` (not an error).
    """
    docs: list[Document] = []
    dir_path = Path(dir)
    if not dir_path.exists():
        return docs

    for pdf_path in sorted(dir_path.rglob("*.pdf")):
        rel = _rel_source(pdf_path)
        loader = PyPDFLoader(str(pdf_path))
        for d in loader.load():
            page = d.metadata.get("page", -1)
            if not isinstance(page, int):
                try:
                    page = int(page)
                except (TypeError, ValueError):
                    page = -1
            lang = _detect_language(d.page_content)
            docs.append(
                Document(
                    page_content=d.page_content,
                    metadata={
                        "source": rel,
                        "page": page,
                        "doc_type": "manual",
                        "language": lang,
                        "chunk_id": str(uuid.uuid4()),
                    },
                )
            )
    return docs


def load_csv(
    path: Path,
    fieldnames: list[str] | None = None,
) -> list[Document]:
    """Load a CSV with CSVLoader, one Document per row.

    The project's OBD CSV is HEADERLESS — passing it through CSVLoader's
    default behavior would treat the first row as column names and lose it.
    We default ``fieldnames=["code", "description"]`` to match the headerless
    OBD layout; callers may pass alternative fieldnames for other CSVs.

    Missing path returns ``[]`` (not an error).
    """
    path = Path(path)
    if not path.exists():
        return []
    rel = _rel_source(path)
    if fieldnames is None:
        fieldnames = ["code", "description"]
    loader = CSVLoader(
        file_path=str(path),
        csv_args={"fieldnames": fieldnames},
    )
    docs: list[Document] = []
    for d in loader.load():
        lang = _detect_language(d.page_content)
        docs.append(
            Document(
                page_content=d.page_content,
                metadata={
                    "source": rel,
                    "page": -1,
                    "doc_type": "obd_code",
                    "language": lang,
                    "chunk_id": str(uuid.uuid4()),
                },
            )
        )
    return docs


def load_markdown(dir: Path) -> list[Document]:
    """Recursively glob ``**/*.md`` under ``dir`` and load via Unstructured.

    Missing or empty ``dir`` returns ``[]`` (not an error).
    """
    docs: list[Document] = []
    dir_path = Path(dir)
    if not dir_path.exists():
        return docs

    for md_path in sorted(dir_path.rglob("*.md")):
        rel = _rel_source(md_path)
        loader = UnstructuredMarkdownLoader(str(md_path))
        for d in loader.load():
            lang = _detect_language(d.page_content)
            docs.append(
                Document(
                    page_content=d.page_content,
                    metadata={
                        "source": rel,
                        "page": -1,
                        "doc_type": "faq",
                        "language": lang,
                        "chunk_id": str(uuid.uuid4()),
                    },
                )
            )
    return docs
