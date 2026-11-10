"""Tests for language resolution and language-aware retrieval.

Guards the bug found in manual testing: ``ChatRequest.language`` was
declared in the schema, validated by Pydantic, and then never read by any
downstream code — setting it had no effect whatsoever. The wiring test at
the bottom fails if that regresses.
"""
from __future__ import annotations

import pytest
from langchain_core.documents import Document

from src.config import settings
from src.language import (
    DEFAULT_LANGUAGE,
    detect_language,
    language_name,
    resolve_language,
)
from src.retriever import get_retriever
from src.vectorstore import add_chunks


# ---------------------------------------------------------------------------
# Detection / resolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "question,expected",
    [
        ("How do I check the engine oil level on my F-150?", "en"),
        ("What is the boiling point of nitrogen?", "en"),
        ("Comment vérifier le niveau d'huile moteur ?", "fr"),
        ("Quels sont les signes qu'un pneu est sous-gonflé ?", "fr"),
    ],
)
def test_auto_detects_language_of_real_questions(question, expected):
    assert resolve_language(question, "auto") == expected


@pytest.mark.parametrize("question", ["P0420", "bonjour", "merci", "", "   "])
def test_ambiguous_or_short_input_falls_back_to_default(question):
    """Single tokens are unreliable for langdetect — must not crash or guess wild.

    Either the detector is under the confidence threshold or it picks a
    language outside SUPPORTED_LANGUAGES; both collapse to the default.
    """
    assert resolve_language(question, "auto") == DEFAULT_LANGUAGE


def test_explicit_language_overrides_detection():
    """An explicit request language beats whatever the detector thinks."""
    english_question = "How do I check the engine oil level?"
    assert resolve_language(english_question, "fr") == "fr"

    french_question = "Comment vérifier le niveau d'huile moteur ?"
    assert resolve_language(french_question, "en") == "en"


def test_unsupported_explicit_language_falls_back():
    assert resolve_language("anything at all", "de") == DEFAULT_LANGUAGE


def test_language_name_is_a_word_not_a_code():
    """The prompt interpolates a name — 'French' steers the LLM, 'fr' does not."""
    assert language_name("fr") == "French"
    assert language_name("en") == "English"
    assert language_name("zz") == "English"  # unknown → default


def test_detect_language_respects_confidence_threshold():
    """A threshold above any achievable probability always yields the default."""
    assert detect_language("Comment vérifier le niveau d'huile ?", confidence_threshold=1.1) == "en"


# ---------------------------------------------------------------------------
# Language-aware retrieval
# ---------------------------------------------------------------------------


@pytest.fixture
def bilingual_docs() -> list[Document]:
    """Six chunks: 3 English + 2 French manual pages, 1 English-only code row.

    Mirrors the real corpus shape, where the code table is tagged ``en`` even
    though it must stay reachable from a French question.
    """
    def _doc(text, lang, source, cid, doc_type="manual"):
        return Document(
            page_content=text,
            metadata={
                "source": source,
                "page": 1,
                "doc_type": doc_type,
                "language": lang,
                "chunk_id": cid,
            },
        )

    return [
        _doc("To check the engine oil level, park on level ground and use the dipstick.", "en", "manuals/en.pdf", "en-1"),
        _doc("Change the engine oil and filter at the scheduled maintenance interval.", "en", "manuals/en.pdf", "en-2"),
        _doc("Inspect the brake pads for wear at every service interval.", "en", "manuals/en.pdf", "en-3"),
        _doc("Pour vérifier le niveau d'huile moteur, garez le véhicule sur un sol plat.", "fr", "manuals/fr.pdf", "fr-1"),
        _doc("Changez l'huile moteur et le filtre selon le calendrier d'entretien.", "fr", "manuals/fr.pdf", "fr-2"),
        _doc("code: P0420\ndescription: Catalyst System Efficiency Below Threshold (Bank 1)", "en", "obd_codes.csv", "csv-1", "obd_code"),
    ]


def _ingest(docs):
    add_chunks(docs)


def test_french_request_prefers_french_chunks(tmp_qdrant, bilingual_docs):
    """A French-biased retriever must surface more French chunks than the default one."""
    _ingest(bilingual_docs)
    question = "Comment vérifier le niveau d'huile moteur ?"

    default_hits = get_retriever().invoke(question)
    french_hits = get_retriever("fr").invoke(question)

    n_fr_default = sum(1 for d in default_hits if d.metadata["language"] == "fr")
    n_fr_biased = sum(1 for d in french_hits if d.metadata["language"] == "fr")

    assert n_fr_biased >= n_fr_default, (
        f"language bias did not increase French chunks: {n_fr_default} → {n_fr_biased}"
    )
    assert n_fr_biased >= 2, f"expected both French chunks, got {n_fr_biased}"


def test_language_bias_still_reaches_other_language_content(tmp_qdrant, bilingual_docs):
    """The reserved share is partial — an English-only code row stays reachable in FR.

    This is why a hard metadata filter was rejected: 395 of 470 real OBD rows
    are tagged ``en``, so filtering a French request to fr would hide them.
    """
    _ingest(bilingual_docs)
    hits = get_retriever("fr").invoke("Que signifie le code P0420 ?")
    sources = [d.metadata["source"] for d in hits]
    assert "obd_codes.csv" in sources, (
        f"English-tagged code row unreachable from a French query; got {sources}"
    )


def test_language_retriever_respects_top_k(tmp_qdrant, bilingual_docs):
    _ingest(bilingual_docs)
    hits = get_retriever("fr").invoke("huile moteur")
    assert len(hits) <= settings.top_k


def test_language_retrievers_are_cached_per_language(tmp_qdrant):
    assert get_retriever("fr") is get_retriever("fr")
    assert get_retriever("fr") is not get_retriever("en")
    assert get_retriever("fr") is not get_retriever()


def test_ratio_zero_disables_language_bias(tmp_qdrant, monkeypatch):
    """language_pref_ratio=0 must fall back to the plain settings retriever."""
    monkeypatch.setattr(settings, "language_pref_ratio", 0.0)
    assert get_retriever("fr") is get_retriever()


# ---------------------------------------------------------------------------
# End-to-end wiring guard
# ---------------------------------------------------------------------------


def _capture_chain_call(mocker, sample_docs):
    """Patch routes.get_chain with a stub, returning (get_chain_mock, chain_mock)."""
    import src.api.routes as routes_mod

    chain_mock = mocker.MagicMock()
    chain_mock.invoke.return_value = {
        "input": "q",
        "context": list(sample_docs),
        "answer": "stubbed answer",
    }
    get_chain_mock = mocker.patch.object(
        routes_mod, "get_chain", return_value=chain_mock
    )
    return get_chain_mock, chain_mock


def test_chat_request_language_actually_reaches_the_chain(tmp_qdrant, sample_docs, mocker):
    """The dead-parameter guard.

    ``ChatRequest.language`` was accepted and silently ignored for the whole
    POC. Assert an explicit language reaches BOTH the retriever selection
    (the get_chain argument) and the prompt's answer_language slot.
    """
    from fastapi.testclient import TestClient

    from src.api.main import app

    _ingest(sample_docs)
    get_chain_mock, chain_mock = _capture_chain_call(mocker, sample_docs)

    with TestClient(app) as client:
        resp = client.post(
            "/chat",
            json={"question": "How do I check the engine oil?", "language": "fr"},
        )

    assert resp.status_code == 200
    # Retrieval bias: explicit "fr" wins over the English question text.
    get_chain_mock.assert_called_once_with("fr")
    # Answer directive: the display name, not the raw code.
    assert chain_mock.invoke.call_args[0][0]["answer_language"] == "French"


def test_auto_language_reaches_the_chain_from_question_text(tmp_qdrant, sample_docs, mocker):
    """With the default "auto", the detected language must drive both knobs."""
    from fastapi.testclient import TestClient

    from src.api.main import app

    _ingest(sample_docs)
    get_chain_mock, chain_mock = _capture_chain_call(mocker, sample_docs)

    with TestClient(app) as client:
        resp = client.post(
            "/chat", json={"question": "Comment vérifier le niveau d'huile moteur ?"}
        )

    assert resp.status_code == 200
    get_chain_mock.assert_called_once_with("fr")
    assert chain_mock.invoke.call_args[0][0]["answer_language"] == "French"
