"""Language detection and request-language resolution.

Shared by both paths:

- **Ingestion** stamps ``metadata.language`` on every chunk (see
  ``src/loaders.py``, which delegates here so there is exactly one copy of
  the detection logic).
- **Query** resolves ``ChatRequest.language`` into a concrete code that
  steers retrieval (``src/retriever.py``) and fills the answer-language
  directive in the prompt (``src/prompt.py``).

Domain-agnostic by design — nothing here knows what the corpus contains,
so the ElBATIMA migration inherits it unchanged. Supporting an additional
language is one entry in ``SUPPORTED_LANGUAGES``, one in ``_LANGUAGE_NAMES``,
and one in the ``ChatRequest.language`` Literal — no logic changes.

Two different confidence thresholds are used on purpose:

- ``INGEST_CONFIDENCE`` (0.7) for document chunks, which are long enough
  that langdetect is reliable and a wrong tag pollutes the index.
- ``QUERY_CONFIDENCE`` (0.5) for questions, which are often a single short
  sentence where langdetect is less certain. Being slightly permissive
  here is cheap: the worst case is answering in the wrong language for one
  request, which the caller can override explicitly.
"""
from __future__ import annotations

from langdetect import DetectorFactory, LangDetectException, detect_langs

# Make langdetect deterministic for tests/reproducibility.
DetectorFactory.seed = 0

SUPPORTED_LANGUAGES = ("en", "fr")
DEFAULT_LANGUAGE = "en"

INGEST_CONFIDENCE = 0.7
QUERY_CONFIDENCE = 0.5

_LANGUAGE_NAMES = {"en": "English", "fr": "French"}


def detect_language(
    text: str,
    default: str = DEFAULT_LANGUAGE,
    confidence_threshold: float = INGEST_CONFIDENCE,
) -> str:
    """Return a supported language code if detected at or above the threshold.

    Only inspects the first 500 chars (langdetect is slow on long text).
    Returns ``default`` for empty/non-textual input, on detector failure, or
    when the winning language falls outside ``SUPPORTED_LANGUAGES``.
    """
    sample = (text or "").strip()[:500]
    if not sample:
        return default
    try:
        candidates = detect_langs(sample)
    except LangDetectException:
        return default
    if not candidates:
        return default
    top = candidates[0]
    if top.prob < confidence_threshold:
        return default
    if top.lang not in SUPPORTED_LANGUAGES:
        return default
    return top.lang


def resolve_language(question: str, requested: str = "auto") -> str:
    """Resolve the effective answer language for one chat request.

    An explicit ``"en"``/``"fr"`` from the client always wins — the user
    knows what they want better than the detector does. ``"auto"`` (the
    schema default) detects from the question text and falls back to
    ``DEFAULT_LANGUAGE`` when the text is too short or ambiguous to call.
    """
    if requested and requested != "auto":
        return requested if requested in SUPPORTED_LANGUAGES else DEFAULT_LANGUAGE
    return detect_language(question, confidence_threshold=QUERY_CONFIDENCE)


def language_name(code: str) -> str:
    """Human-readable name for a language code, for prompt interpolation.

    The LLM follows "answer in French" far more reliably than "answer in fr".
    """
    return _LANGUAGE_NAMES.get(code, _LANGUAGE_NAMES[DEFAULT_LANGUAGE])
