"""Grounded RAG prompt.

The persona slot is filled from settings.assistant_persona so the
pipeline is domain-agnostic — code never hardcodes a domain string.

This module exposes TWO templates:
- ``get_prompt()``: the system prompt with {context} + {input} slots,
  consumed by ``create_stuff_documents_chain``.
- ``get_document_prompt()``: the per-Document template stuffed into
  {context}. Per §4 each chunk is prefixed with ``[source: <file>, page: <n>]``
  so the LLM has the metadata it needs to cite. ``{page_display}`` is the
  display-side version of the storage sentinel ``page=-1`` (rendered as
  ``"N/A"`` by the chain pre-processor — see src/chain.py).
"""
from __future__ import annotations

from langchain_core.prompts import PromptTemplate

from src.config import settings


SYSTEM_PROMPT_TEMPLATE = """You are an expert {persona} assistant.
Answer the user's question using ONLY the context provided below.

Rules:
- If the context does not contain enough information to answer, reply exactly:
  "I don't have enough information in my knowledge base to answer that reliably."
- Do not invent part numbers, torque values, fluid capacities, diagnostic codes, norm references, or article numbers.
- Cite the source filename in parentheses after each factual claim, e.g. (toyota_corolla_2019.pdf, p.47).
- Reply in the SAME LANGUAGE as the user's question.
- Be concise: 1-4 short paragraphs, or a short bulleted list when steps are involved.

Context:
{context}

Question: {input}

Answer:"""


DOCUMENT_PROMPT_TEMPLATE = """[source: {source}, page: {page_display}]
{page_content}"""


def get_prompt() -> PromptTemplate:
    """Return a PromptTemplate with {persona} pre-filled.

    The remaining variables are {context} (filled by create_stuff_documents_chain
    from the retrieved Documents) and {input} (the user question, the default
    key used by create_retrieval_chain).
    """
    return PromptTemplate.from_template(
        SYSTEM_PROMPT_TEMPLATE,
        partial_variables={"persona": settings.assistant_persona},
    )


def get_document_prompt() -> PromptTemplate:
    """Return the per-Document template injected into {context}.

    Variables: ``source``, ``page_display``, ``page_content``. The chain
    layer pre-computes ``page_display`` from the storage-side ``page``
    int (``"N/A"`` when ``page == -1``).
    """
    return PromptTemplate(
        input_variables=["source", "page_display", "page_content"],
        template=DOCUMENT_PROMPT_TEMPLATE,
    )
