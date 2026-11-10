"""Shared pytest fixtures for the RAG chatbot test suite.

Three core fixtures:

- ``tmp_qdrant``  — function-scoped embedded-Qdrant directory bound to
  ``settings.qdrant_path``; closes the client and resets the vectorstore,
  retriever and chain caches so the test actually uses the temp dir.
- ``sample_docs`` — three inline LangChain ``Document`` objects covering
  the three doc_types (manual / obd_code / faq) with full metadata.
- ``mock_llm``   — patches ``ChatOllama.invoke`` (and ``ainvoke``) where
  the chain actually resolves it, returning a canned ``AIMessage``.

Static fixture files live under ``tests/fixtures/``. ``sample.pdf`` is
regenerated on demand if missing (the test suite must not rely on
checked-in binary state).
"""
from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from src import vectorstore
from src.config import settings


# --------------------------------------------------------------------------
# Fixture file management
# --------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _ensure_sample_pdf() -> Path:
    """Regenerate tests/fixtures/sample.pdf if absent.

    Uses reportlab (already in pyproject deps). Two pages so multi-page
    handling in load_pdfs is exercised by the loader tests.
    """
    pdf_path = FIXTURES_DIR / "sample.pdf"
    if pdf_path.exists():
        return pdf_path

    from reportlab.lib.pagesizes import LETTER
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import (
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
    )

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    body = styles["BodyText"]
    title = styles["Title"]

    page1 = (
        "Engine oil is the lifeblood of a passenger vehicle. For most "
        "gasoline engines manufactured after 2010, the manufacturer "
        "recommends changing the oil and filter every 10,000 km or every "
        "six months, whichever comes first. Use only the viscosity grade "
        "specified in the owner manual, typically 5W-30 or 0W-20 synthetic "
        "for modern engines."
    )
    page2 = (
        "Brake pad inspection should be performed at every routine service "
        "interval and any time the driver notices unusual noises, vibration, "
        "or longer pedal travel. New pads start at approximately 10 to 12 "
        "millimeters of friction material; the manufacturer wear limit is "
        "typically 3 millimeters."
    )

    doc = SimpleDocTemplate(str(pdf_path), pagesize=LETTER, title="Sample Manual")
    doc.build([
        Paragraph("Engine Oil", title),
        Spacer(1, 12),
        Paragraph(page1, body),
        PageBreak(),
        Paragraph("Brake Inspection", title),
        Spacer(1, 12),
        Paragraph(page2, body),
    ])
    return pdf_path


@pytest.fixture(scope="session", autouse=True)
def _materialize_fixtures():
    """Make sure the binary fixtures exist before any test runs."""
    _ensure_sample_pdf()
    yield


# --------------------------------------------------------------------------
# tmp_qdrant — isolated embedded Qdrant per test
# --------------------------------------------------------------------------


def _reset_store_caches() -> None:
    """Close the Qdrant client and drop every handle built on top of it."""
    from src import chain as chain_mod
    from src import retriever as retriever_mod

    vectorstore.reset_caches()
    retriever_mod.reset_caches()
    chain_mod.reset_caches()


@pytest.fixture
def tmp_qdrant(tmp_path_factory, monkeypatch):
    """Yield a fresh embedded-Qdrant directory bound to settings.

    Forces embedded mode (``qdrant_url=None``) and closes any open client so
    ``get_client()`` / ``get_retriever()`` / ``get_chain()`` re-open under the
    temp path. Teardown closes the client again, releasing the folder lock.
    """
    tmp_dir = tmp_path_factory.mktemp("qdrant")
    monkeypatch.setattr(settings, "qdrant_path", tmp_dir)
    monkeypatch.setattr(settings, "qdrant_url", None)
    _reset_store_caches()

    yield tmp_dir

    _reset_store_caches()


# --------------------------------------------------------------------------
# sample_docs — inline Document objects, no disk I/O
# --------------------------------------------------------------------------


@pytest.fixture
def sample_docs() -> list[Document]:
    """Three Documents covering the three doc_types with full metadata."""
    return [
        Document(
            page_content=(
                "Regular engine oil changes are essential for engine longevity. "
                "Most modern gasoline engines call for 5W-30 synthetic oil and "
                "an interval of 10,000 km or six months."
            ),
            metadata={
                "source": "manuals/sample.pdf",
                "page": 5,
                "doc_type": "manual",
                "language": "en",
                "chunk_id": "test-uuid-1",
            },
        ),
        Document(
            page_content=(
                "code: P0420\n"
                "description: Catalyst System Efficiency Below Threshold (Bank 1)"
            ),
            metadata={
                "source": "obd_codes.csv",
                "page": -1,
                "doc_type": "obd_code",
                "language": "en",
                "chunk_id": "test-uuid-2",
            },
        ),
        Document(
            page_content=(
                "Q: How often should the cabin air filter be replaced?\n"
                "A: Typically every 25,000 km or once a year, more often in "
                "dusty environments."
            ),
            metadata={
                "source": "faq/sample.md",
                "page": -1,
                "doc_type": "faq",
                "language": "en",
                "chunk_id": "test-uuid-3",
            },
        ),
    ]


# --------------------------------------------------------------------------
# mock_llm — patch ChatOllama at the chain's resolution point
# --------------------------------------------------------------------------


@pytest.fixture
def mock_llm(mocker):
    """Patch ChatOllama.invoke / ainvoke at langchain_ollama.

    Returns the invoke mock. The chain calls the LLM through LangChain's
    Runnable protocol which dispatches to ``invoke`` (sync) or ``ainvoke``
    (async). We also reset the ``src.llm._llm`` singleton so the patched
    class is reinstantiated even if a previous test materialized it.
    """
    canned = AIMessage(content="Mocked answer (sample.pdf, p.5).")

    invoke_mock = mocker.patch(
        "langchain_ollama.ChatOllama.invoke",
        return_value=canned,
    )
    # ainvoke is async — return an awaitable wrapped result.
    async def _ainvoke(*args, **kwargs):
        return canned
    mocker.patch(
        "langchain_ollama.ChatOllama.ainvoke",
        side_effect=_ainvoke,
    )

    # Reset LLM and chain caches so subsequent get_llm/get_chain calls
    # build against the patched ChatOllama.
    from src import llm as llm_mod
    from src import chain as chain_mod
    llm_mod._llm = None
    chain_mod.reset_caches()

    yield invoke_mock

    # Reset on teardown so the next test starts clean.
    llm_mod._llm = None
    chain_mod.reset_caches()
