"""Tests for the Redis answer cache (src/cache.py).

Pure helpers run everywhere. The live tests need Redis Stack on
settings.redis_url and are skipped when it is unreachable. They use a
unique collection name, which changes the cache namespace, so they never
see or touch real cached answers; their keys are deleted afterwards.
"""
from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from src import cache
from src.config import settings

ANSWER = {"answer": "Catalyst efficiency below threshold (obd_codes.csv).", "sources": [], "retrieved_k": 1}


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_normalize_ignores_case_spacing_apostrophes_and_trailing_punctuation():
    a = cache.normalize_question("  How do I check the  OIL level?? ")
    b = cache.normalize_question("how do i check the oil level")
    assert a == b
    assert cache.normalize_question("l’huile") == cache.normalize_question("l'huile")


@pytest.mark.parametrize(
    "question,expected",
    [
        ("What does P0420 mean?", "P0420"),
        ("What does p0420 mean?", "P0420"),
        ("Compare P0455 and P0442.", "P0442,P0455"),
        ("NF DTU 13.3 article R.111-4", "13.3,R.111-4"),
        ("How often should I change the oil?", ""),
    ],
)
def test_identifiers(question, expected):
    assert cache.identifiers(question) == expected


def test_fingerprint_changes_with_answer_shaping_settings(monkeypatch):
    before = cache.config_fingerprint()
    monkeypatch.setattr(settings, "top_k", settings.top_k + 1)
    assert cache.config_fingerprint() != before


def test_disabled_cache_never_touches_redis():
    # conftest switches the cache off for every test
    assert cache.lookup("anything", "en") is None
    cache.store("anything", "en", ANSWER)  # must not raise


# ---------------------------------------------------------------------------
# Live Redis
# ---------------------------------------------------------------------------


@pytest.fixture
def live_cache(monkeypatch):
    monkeypatch.setattr(settings, "cache_enabled", True)
    monkeypatch.setattr(settings, "collection_name", f"cachetest_{uuid.uuid4().hex[:8]}")
    cache.reset()
    r = cache._redis()
    try:
        r.ping()
        r.execute_command("FT._LIST")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Redis Stack not reachable at {settings.redis_url}: {exc}")
    ns_marker = settings.collection_name
    yield r
    fp = cache.config_fingerprint()
    for key in r.scan_iter(match=f"rag:cache:*{fp}*"):
        r.delete(key)
    assert ns_marker  # namespace was test-only


def test_exact_hit(live_cache):
    cache.store("What does P0420 mean?", "en", ANSWER)
    hit = cache.lookup("what does p0420 mean", "en")
    assert hit == (ANSWER, "exact")


def test_semantic_hit_for_a_paraphrase(live_cache, monkeypatch):
    monkeypatch.setattr(settings, "cache_similarity_threshold", 0.80)
    cache.store("How often should I change the engine oil?", "en", ANSWER)
    hit = cache.lookup("How frequently do I need to change my engine oil?", "en")
    assert hit is not None and hit[1] == "semantic"


def test_semantic_layer_never_mixes_identifiers(live_cache, monkeypatch):
    """MiniLM sees P0420 and P0455 as near-identical; the identifier guard must not."""
    monkeypatch.setattr(settings, "cache_similarity_threshold", 0.50)
    cache.store("What does P0420 mean?", "en", ANSWER)
    assert cache.lookup("What does P0455 mean?", "en") is None


def test_language_is_part_of_the_key(live_cache, monkeypatch):
    monkeypatch.setattr(settings, "cache_similarity_threshold", 0.50)
    cache.store("How often should I change the engine oil?", "en", ANSWER)
    assert cache.lookup("How often should I change the engine oil?", "fr") is None


def test_invalidate_makes_old_answers_unreachable(live_cache):
    cache.store("What does P0420 mean?", "en", ANSWER)
    cache.invalidate()
    assert cache.lookup("What does P0420 mean?", "en") is None


def test_unreachable_redis_degrades_to_a_miss(monkeypatch):
    monkeypatch.setattr(settings, "cache_enabled", True)
    monkeypatch.setattr(settings, "redis_url", "redis://127.0.0.1:1/0")
    cache.reset()
    assert cache.lookup("What does P0420 mean?", "en") is None
    cache.store("What does P0420 mean?", "en", ANSWER)  # must not raise


# ---------------------------------------------------------------------------
# API wiring
# ---------------------------------------------------------------------------


def test_api_serves_cache_hit_without_running_the_chain(tmp_qdrant, sample_docs, mocker):
    import src.api.routes as routes
    from src.api.main import app
    from src.vectorstore import add_chunks

    add_chunks(sample_docs)
    mocker.patch.object(routes.answer_cache, "lookup", return_value=(ANSWER, "semantic"))
    get_chain = mocker.patch.object(routes, "get_chain")

    body = TestClient(app).post("/chat", json={"question": "What does P0420 mean?"}).json()
    assert body["answer"] == ANSWER["answer"]
    assert body["cache_hit"] == "semantic"
    get_chain.assert_not_called()


def test_api_stores_fresh_answers(tmp_qdrant, sample_docs, mock_llm, mocker):
    import src.api.routes as routes
    from src.api.main import app
    from src.vectorstore import add_chunks

    add_chunks(sample_docs)
    store = mocker.patch.object(routes.answer_cache, "store")
    body = TestClient(app).post("/chat", json={"question": "What does P0420 mean?"}).json()
    assert body["cache_hit"] is None
    store.assert_called_once()
    _, lang, payload = store.call_args[0]
    assert lang == "en"
    assert payload["answer"] == body["answer"]
