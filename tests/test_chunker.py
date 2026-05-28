"""Unit tests for src/chunker.py.

Verifies the dual chunking strategy:
- manual / faq pages → RecursiveCharacterTextSplitter (chunk_size=800,
  chunk_overlap=150).
- obd_code rows → atomic, never split.

Also verifies metadata propagation and chunk_id uniqueness.
"""
from __future__ import annotations

import uuid

import pytest
from langchain_core.documents import Document

from src.chunker import chunk_documents
from src.config import settings


# A blob clearly longer than chunk_size so the splitter must produce >1 chunk.
LONG_MANUAL_TEXT = (
    "Engine oil maintenance is one of the most important services. "
    "Change the oil every 10,000 km. Use the viscosity grade specified. "
) * 30  # ~ 30 * 110 = 3,300 chars


def _make_manual_doc(text: str = LONG_MANUAL_TEXT) -> Document:
    return Document(
        page_content=text,
        metadata={
            "source": "manuals/long.pdf",
            "page": 7,
            "doc_type": "manual",
            "language": "en",
            "chunk_id": "parent-uuid",
        },
    )


def _make_obd_doc(text: str) -> Document:
    return Document(
        page_content=text,
        metadata={
            "source": "obd_codes.csv",
            "page": -1,
            "doc_type": "obd_code",
            "language": "en",
            "chunk_id": "parent-uuid",
        },
    )


# ---------------------------------------------------------------------------
# Manual / PDF chunks: splitter applied
# ---------------------------------------------------------------------------


def test_manual_long_text_produces_multiple_chunks():
    out = chunk_documents([_make_manual_doc()])
    assert len(out) > 1, f"expected splitter to produce >1 chunk, got {len(out)}"


def test_manual_chunks_respect_chunk_size_with_tolerance():
    out = chunk_documents([_make_manual_doc()])
    # Allow a small tolerance — separator boundaries can push a chunk a few
    # chars past chunk_size in practice.
    tolerance = 50
    for c in out:
        assert len(c.page_content) <= settings.chunk_size + tolerance, (
            f"chunk len {len(c.page_content)} exceeds {settings.chunk_size + tolerance}"
        )


def test_manual_chunks_have_nonzero_overlap_between_adjacent_chunks():
    """Adjacent chunks of the same source must share some characters.

    Exact-byte overlap is brittle (separator boundaries shift it). We just
    confirm that the tail of chunk N and the head of chunk N+1 share at
    least a short substring — confirming the splitter actually overlapped.
    """
    out = chunk_documents([_make_manual_doc()])
    if len(out) < 2:
        pytest.skip("not enough chunks to test overlap")

    a, b = out[0].page_content, out[1].page_content
    tail = a[-100:]  # last 100 chars of chunk N
    # Try to find any 20-char window of the tail inside chunk N+1's first 200 chars.
    head = b[:200]
    overlap_window = 20
    found = any(
        tail[i : i + overlap_window] and tail[i : i + overlap_window] in head
        for i in range(len(tail) - overlap_window)
    )
    assert found, "no overlap detected between adjacent manual chunks"


def test_manual_metadata_propagates_to_chunks():
    out = chunk_documents([_make_manual_doc()])
    for c in out:
        assert c.metadata["source"] == "manuals/long.pdf"
        assert c.metadata["page"] == 7
        assert c.metadata["doc_type"] == "manual"
        assert c.metadata["language"] == "en"
        # chunk_id must NOT be the parent's — it should be regenerated.
        assert c.metadata["chunk_id"] != "parent-uuid"


# ---------------------------------------------------------------------------
# OBD CSV rows: atomic, never split
# ---------------------------------------------------------------------------


def test_obd_row_is_not_split_even_when_artificially_long():
    """obd_code docs bypass the splitter — verify even an oversize row stays 1 chunk."""
    long_row = "code: P0420\ndescription: " + ("x " * 600)  # > 1200 chars
    out = chunk_documents([_make_obd_doc(long_row)])
    assert len(out) == 1, f"expected exactly 1 chunk for obd row, got {len(out)}"
    assert out[0].page_content == long_row


def test_obd_row_metadata_propagates_and_chunk_id_regenerates():
    doc = _make_obd_doc("code: P0420\ndescription: Catalyst System Below Threshold")
    out = chunk_documents([doc])
    assert len(out) == 1
    c = out[0]
    assert c.metadata["source"] == "obd_codes.csv"
    assert c.metadata["page"] == -1
    assert c.metadata["doc_type"] == "obd_code"
    assert c.metadata["language"] == "en"
    assert c.metadata["chunk_id"] != "parent-uuid"


# ---------------------------------------------------------------------------
# chunk_id uniqueness
# ---------------------------------------------------------------------------


def test_every_chunk_has_unique_chunk_id():
    # Mix of manual (split into many) and obd (atomic) — chunk_ids must all differ.
    docs = [
        _make_manual_doc(),
        _make_manual_doc(text=LONG_MANUAL_TEXT + " extra"),
        _make_obd_doc("code: P0420\ndescription: foo"),
        _make_obd_doc("code: P0301\ndescription: bar"),
    ]
    out = chunk_documents(docs)
    ids = [c.metadata["chunk_id"] for c in out]
    assert len(ids) == len(set(ids)), "chunk_ids are not unique"
    # And they should all parse as UUIDs.
    for cid in ids:
        uuid.UUID(cid)  # raises ValueError if not a valid UUID


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------


def test_empty_input_returns_empty_list():
    assert chunk_documents([]) == []
