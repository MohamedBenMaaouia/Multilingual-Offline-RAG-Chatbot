"""Ollama LLM singleton.

Per ARCHITECTURE.md §4. Uses the modern `langchain-ollama` package;
the langchain_community.ChatOllama path is deprecated.
"""
from __future__ import annotations

from typing import Optional

from langchain_ollama import ChatOllama

from src.config import settings

_llm: Optional[ChatOllama] = None


def get_llm() -> ChatOllama:
    """Return a process-wide singleton ChatOllama."""
    global _llm
    if _llm is None:
        _llm = ChatOllama(
            model=settings.llm_model,
            base_url=settings.ollama_url,
            temperature=settings.temperature,
            num_predict=settings.max_tokens,
            timeout=settings.llm_timeout_s,
        )
    return _llm
