"""Unit tests for src/prompt.py.

Verifies:
- get_prompt() returns a PromptTemplate-like object.
- .format(context=..., input=...) substitutes both slots.
- The canned refusal string appears in the rendered prompt.
- The persona is filled from settings.assistant_persona.
- Literal curly braces in user input do not crash the formatter.
"""
from __future__ import annotations

import pytest
from langchain_core.prompts import BasePromptTemplate

from src.config import settings
from src.prompt import DEFAULT_ANSWER_LANGUAGE, get_prompt


CANNED_REFUSAL = (
    "I don't have enough information in my knowledge base to answer that reliably."
)


def test_get_prompt_returns_prompt_template():
    p = get_prompt()
    assert isinstance(p, BasePromptTemplate)


def test_format_substitutes_context_and_input():
    p = get_prompt()
    rendered = p.format(context="some retrieved context", input="user question")
    assert "some retrieved context" in rendered
    assert "user question" in rendered


def test_canned_refusal_present_in_formatted_prompt():
    p = get_prompt()
    rendered = p.format(context="ctx", input="q")
    assert CANNED_REFUSAL in rendered


def test_persona_filled_from_settings():
    p = get_prompt()
    rendered = p.format(context="ctx", input="q")
    assert settings.assistant_persona in rendered


def test_answer_language_defaults_when_not_supplied():
    """The template must render without an explicit language (legacy callers)."""
    rendered = get_prompt().format(context="ctx", input="q")
    assert DEFAULT_ANSWER_LANGUAGE in rendered


def test_answer_language_override_reaches_the_prompt():
    """A concrete language name must replace the default in every slot."""
    rendered = get_prompt().format(context="ctx", input="q", answer_language="French")
    assert "ENTIRE answer in French" in rendered
    assert "Answer in French:" in rendered
    assert DEFAULT_ANSWER_LANGUAGE not in rendered


def test_refusal_line_stays_english_regardless_of_answer_language():
    """The canned refusal is a contract string checked verbatim by the API and eval."""
    rendered = get_prompt().format(context="ctx", input="q", answer_language="French")
    assert CANNED_REFUSAL in rendered


def test_curly_braces_in_user_input_do_not_crash():
    """User questions can contain literal '{' / '}' — PromptTemplate must not
    re-interpret them as template variables when they arrive via .format().

    This is the known PromptTemplate footgun: f-string-style template syntax
    treats unmatched braces in the SUBSTITUTED VALUES as literal characters
    (only the template body's own braces are special). The test below confirms
    that contract — passing input with literal braces yields a rendered prompt
    that contains those braces verbatim.
    """
    p = get_prompt()
    rendered = p.format(context="ctx", input="What does {foo} mean?")
    assert "{foo}" in rendered, "literal braces in input were not preserved"
