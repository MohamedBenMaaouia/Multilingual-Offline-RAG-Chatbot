"""Answer cache in Redis: repeated questions are answered in milliseconds.

Two layers, checked in order:

1. Exact — the same question (normalized: case, spaces, apostrophes) in the
   same language returns the stored answer. Plain Redis key with a TTL.
2. Semantic — a question that MEANS the same ("how often change oil?" vs
   "oil change interval?") reuses the answer. Uses Redis vector search over
   the MiniLM embedding of past questions, cosine similarity >=
   ``settings.cache_similarity_threshold``.

Guards on the semantic layer, because a wrong cache hit is worse than a miss:

- Identifiers must match exactly. MiniLM sees "What does P0420 mean?" and
  "What does P0455 mean?" as near-identical (the reason hybrid search
  exists), so a hit additionally requires the same set of digit-bearing
  tokens (codes, norm numbers, years, quantities).
- Same language only — a French question never gets an English answer.
- Namespace = re-ingest generation + a hash of everything that shapes an
  answer (models, retrieval settings, prompt text). Re-ingesting or changing
  the config makes every old entry unreachable; old keys age out by TTL.

Redis being down never breaks /chat: the cache disables itself for 30 s
after a failure and requests go through the normal pipeline.

Domain-agnostic: nothing here knows about the corpus.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from typing import Any, Optional

import numpy as np
from redis import Redis
from redis.commands.search.field import TagField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query

from src.config import settings
from src.embeddings import get_embeddings

INDEX = "rag_answer_cache"
EXACT_PREFIX = "rag:cache:exact:"
SEM_PREFIX = "rag:cache:sem:"
GENERATION_KEY = "rag:cache:generation"
_BACKOFF_S = 30.0

_client: Optional[Redis] = None
_index_ready = False
_disabled_until = 0.0


# ---------------------------------------------------------------------------
# Pure helpers (no Redis)
# ---------------------------------------------------------------------------


def normalize_question(question: str) -> str:
    """Case, whitespace, apostrophe and trailing-punctuation insensitive form."""
    q = (question or "").replace("’", "'").replace("‘", "'").lower()
    q = re.sub(r"\s+", " ", q).strip()
    return q.rstrip(" ?!.;:")


def identifiers(question: str) -> str:
    """Canonical set of digit-bearing tokens: codes, norm refs, years, quantities.

    "What does P0420 mean?" -> "P0420"; "DTU 13.3 and NF P01-012" -> "13.3,P01-012".
    Two questions can share a semantic cache entry only if these are equal.
    """
    tokens = re.findall(r"\w[\w.\-/]*", question or "")
    ids = {t.upper().rstrip(".-/") for t in tokens if any(ch.isdigit() for ch in t)}
    return ",".join(sorted(ids))


def config_fingerprint() -> str:
    """Hash of every setting and prompt text that changes what an answer would be."""
    from src.prompt import DOCUMENT_PROMPT_TEMPLATE, SYSTEM_PROMPT_TEMPLATE, USER_PROMPT_TEMPLATE

    shape = {
        "collection": settings.collection_name,
        "llm": [settings.llm_model, settings.temperature, settings.max_tokens],
        "retrieval": [
            settings.retrieval_mode, settings.search_type, settings.top_k, settings.fetch_k,
            settings.mmr_lambda, settings.language_pref_ratio, settings.bm25_per_language,
            settings.reranker_enabled, settings.reranker_model, settings.embedding_model,
        ],
        "prompt": [SYSTEM_PROMPT_TEMPLATE, USER_PROMPT_TEMPLATE, DOCUMENT_PROMPT_TEMPLATE,
                   settings.assistant_persona],
    }
    return hashlib.sha256(json.dumps(shape, sort_keys=True).encode()).hexdigest()[:12]


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


def _tag(value: str) -> str:
    """Escape a value for a RediSearch TAG filter."""
    return re.sub(r"([^A-Za-z0-9_])", r"\\\1", value)


# ---------------------------------------------------------------------------
# Redis plumbing
# ---------------------------------------------------------------------------


def _redis() -> Optional[Redis]:
    """The shared client, or None when caching is off or Redis recently failed."""
    global _client
    if not settings.cache_enabled or time.monotonic() < _disabled_until:
        return None
    if _client is None:
        _client = Redis.from_url(
            settings.redis_url, socket_timeout=1.0, socket_connect_timeout=1.0
        )
    return _client


def _failed(exc: Exception) -> None:
    global _disabled_until
    _disabled_until = time.monotonic() + _BACKOFF_S
    print(
        f"cache: Redis unavailable ({type(exc).__name__}: {exc}); "
        f"bypassing the cache for {_BACKOFF_S:.0f}s",
        file=sys.stderr,
    )


def reset() -> None:
    """Forget the client and back-off state (tests, re-pointing REDIS_URL)."""
    global _client, _index_ready, _disabled_until
    _client = None
    _index_ready = False
    _disabled_until = 0.0


def _ensure_index(r: Redis) -> None:
    global _index_ready
    if _index_ready:
        return
    try:
        r.ft(INDEX).info()
    except Exception:  # noqa: BLE001 — "Unknown index name": create it
        r.ft(INDEX).create_index(
            [
                TagField("ns"),
                TagField("lang"),
                VectorField(
                    "embedding",
                    "HNSW",
                    {"TYPE": "FLOAT32", "DIM": 384, "DISTANCE_METRIC": "COSINE"},
                ),
            ],
            definition=IndexDefinition(prefix=[SEM_PREFIX], index_type=IndexType.HASH),
        )
    _index_ready = True


def _namespace(r: Redis) -> str:
    generation = (r.get(GENERATION_KEY) or b"0").decode()
    return f"g{generation}-{config_fingerprint()}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def lookup(question: str, language: str) -> Optional[tuple[dict[str, Any], str]]:
    """Return ``(cached_response, "exact" | "semantic")`` or None on a miss."""
    r = _redis()
    if r is None:
        return None
    try:
        ns = _namespace(r)
        norm = normalize_question(question)
        raw = r.get(f"{EXACT_PREFIX}{ns}:{language}:{_digest(norm)}")
        if raw:
            return json.loads(raw), "exact"

        _ensure_index(r)
        vec = np.asarray(get_embeddings().embed_query(question), dtype=np.float32)
        query = (
            Query(f"(@ns:{{{_tag(ns)}}} @lang:{{{_tag(language)}}})=>[KNN 3 @embedding $vec AS dist]")
            .sort_by("dist")
            .return_fields("dist", "ids", "response")
            .dialect(2)
        )
        result = r.ft(INDEX).search(query, query_params={"vec": vec.tobytes()})
        wanted_ids = identifiers(question)
        for doc in result.docs:
            similarity = 1.0 - float(doc.dist)
            if similarity < settings.cache_similarity_threshold:
                break  # sorted by distance: the rest are further away
            if doc.ids == wanted_ids:
                return json.loads(doc.response), "semantic"
        return None
    except Exception as exc:  # noqa: BLE001 — a cache must never break /chat
        _failed(exc)
        return None


def store(question: str, language: str, response: dict[str, Any]) -> None:
    """Cache a response under both layers with the configured TTL."""
    r = _redis()
    if r is None:
        return
    try:
        ns = _namespace(r)
        norm = normalize_question(question)
        payload = json.dumps(response, ensure_ascii=False)
        ttl = settings.cache_ttl_s
        r.set(f"{EXACT_PREFIX}{ns}:{language}:{_digest(norm)}", payload, ex=ttl)

        _ensure_index(r)
        vec = np.asarray(get_embeddings().embed_query(question), dtype=np.float32)
        key = f"{SEM_PREFIX}{ns}:{language}:{_digest(norm)}"
        r.hset(
            key,
            mapping={
                "ns": ns,
                "lang": language,
                "ids": identifiers(question),
                "question": question,
                "response": payload,
                "embedding": vec.tobytes(),
            },
        )
        r.expire(key, ttl)
    except Exception as exc:  # noqa: BLE001
        _failed(exc)


def invalidate() -> None:
    """Make every cached answer unreachable (called after a re-ingest)."""
    r = _redis()
    if r is None:
        return
    try:
        r.incr(GENERATION_KEY)
    except Exception as exc:  # noqa: BLE001
        _failed(exc)
