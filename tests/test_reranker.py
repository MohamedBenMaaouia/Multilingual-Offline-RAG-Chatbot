"""Tests for src/reranker.py and the reranked retrieval path.

Unit tests patch the cross-encoder's scores (no model needed). The last
test uses the real multilingual cross-encoder (~470 MB, cached after the
first download) to check the promise hybrid search alone cannot keep: an
exact identifier ranked first among near-identical rows.
"""
from __future__ import annotations

import pytest
from langchain_core.documents import Document

from src.config import settings
from src.reranker import rerank
from src.retriever import get_retriever
from src.vectorstore import add_chunks


def _doc(text: str, cid: str, lang: str = "en") -> Document:
    return Document(
        page_content=text,
        metadata={"source": "x", "page": -1, "doc_type": "manual", "language": lang, "chunk_id": cid},
    )


@pytest.fixture
def fake_scores(mocker):
    """Patch the cross-encoder: score = the number at the start of the chunk text."""
    model = mocker.MagicMock()
    model.predict.side_effect = lambda pairs, **kw: [float(text.split()[0]) for _, text in pairs]
    mocker.patch("src.reranker.get_reranker", return_value=model)
    return model


def test_rerank_sorts_by_score_and_records_it(fake_scores):
    docs = [_doc("1 low", "a"), _doc("9 high", "b"), _doc("5 mid", "c")]
    out = rerank("q", docs)
    assert [d.metadata["chunk_id"] for d in out] == ["b", "c", "a"]
    assert out[0].metadata["rerank_score"] == 9.0


def test_rerank_empty_input_skips_the_model(fake_scores):
    assert rerank("q", []) == []
    fake_scores.predict.assert_not_called()


def test_retriever_honors_reranker_setting(tmp_qdrant, monkeypatch):
    """Drift guard: the retriever reranks exactly when settings say so."""
    for enabled in (True, False):
        monkeypatch.setattr(settings, "reranker_enabled", enabled)
        from src import retriever as retriever_mod

        retriever_mod.reset_caches()
        assert get_retriever().rerank is enabled
        assert get_retriever().rerank_relative_margin == settings.rerank_relative_margin


def test_reranked_path_keeps_top_k_and_language_quota(tmp_qdrant, monkeypatch, fake_scores):
    """English scores highest, yet the French quota (4 of 6) is still honored."""
    monkeypatch.setattr(settings, "reranker_enabled", True)
    monkeypatch.setattr(settings, "rerank_relative_margin", None)  # quota only
    docs = [_doc(f"{90 + i} english chunk {i}", f"en-{i}", "en") for i in range(6)]
    docs += [_doc(f"{10 + i} chunk français {i}", f"fr-{i}", "fr") for i in range(6)]
    add_chunks(docs)

    hits = get_retriever("fr").invoke("question")
    assert len(hits) == settings.top_k
    langs = [d.metadata["language"] for d in hits]
    assert langs[:4] == ["fr"] * 4, langs
    # Within the quota, cross-encoder order: best French score first.
    assert hits[0].metadata["chunk_id"] == "fr-5"


def test_relevance_cutoff_drops_chunks_far_below_the_best(tmp_qdrant, monkeypatch, fake_scores):
    """Margin 4: scores 9 and 6 survive (within 4 of 9), 4 and 1 are dropped."""
    monkeypatch.setattr(settings, "reranker_enabled", True)
    monkeypatch.setattr(settings, "rerank_relative_margin", 4.0)
    add_chunks([_doc(f"{s} chunk", f"c{s}") for s in (9, 6, 4, 1)])
    hits = get_retriever().invoke("question")
    assert [d.metadata["chunk_id"] for d in hits] == ["c9", "c6"]


def test_relevance_cutoff_always_keeps_the_best_chunk(tmp_qdrant, monkeypatch, fake_scores):
    monkeypatch.setattr(settings, "reranker_enabled", True)
    monkeypatch.setattr(settings, "rerank_relative_margin", 0.0)
    add_chunks([_doc(f"{s} chunk", f"c{s}") for s in (3, 2, 1)])
    hits = get_retriever().invoke("question")
    assert [d.metadata["chunk_id"] for d in hits] == ["c3"]


def test_relevance_cutoff_disabled_keeps_top_k(tmp_qdrant, monkeypatch, fake_scores):
    monkeypatch.setattr(settings, "reranker_enabled", True)
    monkeypatch.setattr(settings, "rerank_relative_margin", None)
    add_chunks([_doc(f"{s} chunk", f"c{s}") for s in (90, 9, 8, 7, 6, 5, 4)])
    assert len(get_retriever().invoke("question")) == settings.top_k


def test_cutoff_applies_before_the_language_quota(tmp_qdrant, monkeypatch, fake_scores):
    """A weak same-language chunk is not forced back in by the quota."""
    monkeypatch.setattr(settings, "reranker_enabled", True)
    monkeypatch.setattr(settings, "rerank_relative_margin", 4.0)
    add_chunks([
        _doc("9 english best", "en-9", "en"),
        _doc("8 english close", "en-8", "en"),
        _doc("1 chunk français faible", "fr-1", "fr"),
    ])
    hits = get_retriever("fr").invoke("question")
    assert "fr-1" not in [d.metadata["chunk_id"] for d in hits]


def test_reranker_ranks_exact_code_first(tmp_qdrant, monkeypatch):
    """Hybrid ties BM25's exact match with dense search's best guess (RRF);
    the cross-encoder reads question and chunk together and breaks the tie."""
    monkeypatch.setattr(settings, "retrieval_mode", "hybrid")
    monkeypatch.setattr(settings, "reranker_enabled", True)
    rows = [
        ("P0440", "Evaporative Emission Control System Malfunction"),
        ("P0442", "Evaporative Emission Control System Leak Detected (small leak)"),
        ("P0446", "Evaporative Emission Control System Vent Control Circuit"),
        ("P0451", "Evaporative Emission Control System Pressure Sensor Range"),
        ("P0455", "Evaporative Emission Control System Leak Detected (gross leak)"),
        ("P0456", "Evaporative Emission Control System Leak Detected (very small leak)"),
    ]
    add_chunks([
        Document(
            page_content=f"code: {c}\ndescription: {d}",
            metadata={"source": "obd_codes.csv", "page": -1, "doc_type": "obd_code",
                      "language": "en", "chunk_id": f"row-{c}"},
        )
        for c, d in rows
    ])
    for code in ("P0451", "P0442", "P0455"):
        top = get_retriever().invoke(f"What does {code} mean?")[0]
        assert f"code: {code}" in top.page_content, top.page_content
