"""Unit tests for src/loaders.py.

Covers load_pdfs / load_csv / load_markdown against fixture files in
tests/fixtures/. Verifies:

- Each returns list[Document].
- Every Document has all 5 metadata keys.
- `source` is relative (not anchored at data/raw/).
- `page` is always an int (-1 sentinel for pageless docs).
- `doc_type` matches the loader.
- load_csv treats headerless input correctly (fieldnames honored).
- Missing input dir / file returns [] gracefully (no exception).

These tests are deterministic — langdetect is seeded in src/loaders.py.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.documents import Document

from src.loaders import load_csv, load_markdown, load_pdfs

FIXTURES = Path(__file__).parent / "fixtures"

REQUIRED_META_KEYS = {"source", "page", "doc_type", "language", "chunk_id"}


# ---------------------------------------------------------------------------
# PDF loader
# ---------------------------------------------------------------------------


def test_load_pdfs_returns_list_of_documents():
    docs = load_pdfs(FIXTURES)
    assert isinstance(docs, list)
    assert len(docs) >= 1
    assert all(isinstance(d, Document) for d in docs)


def test_load_pdfs_emits_two_pages_for_sample_pdf():
    """The sample.pdf fixture is 2 pages; loader should emit 2 Documents."""
    docs = load_pdfs(FIXTURES)
    pdf_docs = [d for d in docs if d.metadata["source"].endswith("sample.pdf")]
    assert len(pdf_docs) == 2, f"expected 2 pages, got {len(pdf_docs)}"


def test_load_pdfs_metadata_complete_and_typed():
    docs = load_pdfs(FIXTURES)
    for d in docs:
        assert REQUIRED_META_KEYS.issubset(d.metadata.keys()), (
            f"missing keys: {REQUIRED_META_KEYS - set(d.metadata.keys())}"
        )
        assert d.metadata["doc_type"] == "manual"
        assert isinstance(d.metadata["page"], int)
        # PDFs DO have a page concept — must NOT be the -1 sentinel.
        assert d.metadata["page"] >= 0
        # Source must be relative — no leading data/raw/ or absolute path.
        assert not d.metadata["source"].startswith("data/raw")
        assert not Path(d.metadata["source"]).is_absolute()


def test_load_pdfs_missing_dir_returns_empty_list():
    docs = load_pdfs(Path("./this_directory_does_not_exist_xyz"))
    assert docs == []


# ---------------------------------------------------------------------------
# CSV loader
# ---------------------------------------------------------------------------


def test_load_csv_returns_list_of_documents():
    docs = load_csv(FIXTURES / "sample.csv")
    assert isinstance(docs, list)
    assert len(docs) == 3  # sample.csv has 3 rows, no header
    assert all(isinstance(d, Document) for d in docs)


def test_load_csv_metadata_complete_and_pageless():
    docs = load_csv(FIXTURES / "sample.csv")
    for d in docs:
        assert REQUIRED_META_KEYS.issubset(d.metadata.keys())
        assert d.metadata["doc_type"] == "obd_code"
        assert d.metadata["page"] == -1  # sentinel: pageless
        assert isinstance(d.metadata["page"], int)
        assert not d.metadata["source"].startswith("data/raw")


def test_load_csv_headerless_input_does_not_lose_first_row():
    """sample.csv is headerless: the first row P0420 must be present as content."""
    docs = load_csv(FIXTURES / "sample.csv")
    contents = "\n".join(d.page_content for d in docs)
    assert "P0420" in contents
    assert "P0301" in contents
    assert "P0171" in contents
    # The fieldnames ["code", "description"] should appear as labels in each row.
    assert "code:" in docs[0].page_content
    assert "description:" in docs[0].page_content


def test_load_csv_missing_file_returns_empty_list():
    docs = load_csv(Path("./this_csv_does_not_exist.csv"))
    assert docs == []


# ---------------------------------------------------------------------------
# Markdown loader
# ---------------------------------------------------------------------------


def test_load_markdown_returns_list_of_documents():
    docs = load_markdown(FIXTURES)
    assert isinstance(docs, list)
    assert len(docs) >= 1
    assert all(isinstance(d, Document) for d in docs)


def test_load_markdown_metadata_complete_and_pageless():
    docs = load_markdown(FIXTURES)
    for d in docs:
        assert REQUIRED_META_KEYS.issubset(d.metadata.keys())
        assert d.metadata["doc_type"] == "faq"
        assert d.metadata["page"] == -1
        assert isinstance(d.metadata["page"], int)
        assert not d.metadata["source"].startswith("data/raw")


def test_load_markdown_missing_dir_returns_empty_list():
    docs = load_markdown(Path("./this_md_dir_does_not_exist"))
    assert docs == []
