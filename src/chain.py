"""Full RAG chain: retriever + stuff-documents + Mistral.

Per ARCHITECTURE.md §4. Single source of truth for the runtime chain;
the API layer calls get_chain() instead of re-wiring.

Wiring detail: per §4 the {context} slot should present each chunk
prefixed with ``[source: <filename>, page: <n>]``. The storage-side
schema uses ``page=-1`` as the "no page" sentinel (CSV/MD); the display
side renders that as ``"N/A"``. We compute ``page_display`` on a COPY
of each Document so the LLM-facing template sees the enriched metadata,
while the outer ``result["context"]`` returned to the API still carries
the ORIGINAL Documents with ``page=-1`` for the API's own normalization
to ``None`` (see src/api/routes.py:_doc_to_source_ref).
"""
from __future__ import annotations

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from src.llm import get_llm
from src.prompt import get_document_prompt, get_prompt
from src.retriever import get_retriever

_chain = None


def _add_page_display(docs: list[Document]) -> list[Document]:
    """Return COPIES of each Document with a string ``page_display`` field.

    ``page_display = str(page)`` when ``page >= 0`` else ``"N/A"``. We copy
    each Document so the caller's references (the ``context`` slot echoed
    back to the API) are not mutated — only the LLM-facing view changes.
    """
    out: list[Document] = []
    for d in docs:
        md = dict(d.metadata or {})
        raw_page = md.get("page", -1)
        try:
            page_int = int(raw_page)
        except (TypeError, ValueError):
            page_int = -1
        md["page_display"] = "N/A" if page_int == -1 else str(page_int)
        out.append(Document(page_content=d.page_content, metadata=md))
    return out


def _prepare_context(inputs: dict) -> dict:
    """Inject ``page_display`` into the docs the combine chain will see.

    ``create_retrieval_chain`` invokes the combine chain with the full
    inputs dict including ``context`` (list[Document]). We rewrite
    ``context`` to point at the enriched copies; the outer chain's own
    ``context`` reference is untouched because RunnablePassthrough.assign
    materialized it before this lambda fires.
    """
    docs = inputs.get("context") or []
    new_inputs = dict(inputs)
    new_inputs["context"] = _add_page_display(docs)
    return new_inputs


def get_chain():
    """Return a process-wide singleton retrieval chain.

    The chain's invoke contract is ``{"input": "<question>"}``; the
    returned dict has keys ``"input"``, ``"context"`` (list[Document]
    with original metadata — ``page=-1`` preserved), and ``"answer"``.
    """
    global _chain
    if _chain is None:
        stuff_chain = create_stuff_documents_chain(
            llm=get_llm(),
            prompt=get_prompt(),
            document_prompt=get_document_prompt(),
        )
        # Wrap the combine chain so its view of `context` has page_display.
        # The OUTER chain's `context` (returned to caller) is not affected.
        combine_docs_chain = RunnableLambda(_prepare_context) | stuff_chain

        _chain = create_retrieval_chain(
            retriever=get_retriever(),
            combine_docs_chain=combine_docs_chain,
        )
    return _chain
