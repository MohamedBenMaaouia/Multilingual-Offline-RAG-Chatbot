"""Smoke test for the Qdrant + embeddings wiring (no LLM).

Writes 3 dummy chunks into a throwaway collection (``smoke_db``, never the
real one), runs a dense and a hybrid query, and exits 0 only if the P0420
chunk ranks first in both. The throwaway collection is deleted afterwards.

Run from project root (stop the API first when Qdrant runs embedded — the
local folder can only be opened by one process at a time):

    python scripts/smoke_db.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make sure `src.*` is importable when run as a plain script from repo root.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langchain_core.documents import Document  # noqa: E402

from src.config import settings  # noqa: E402

# Never touch the real collection.
settings.collection_name = "smoke_db"

from src.vectorstore import (  # noqa: E402
    add_chunks,
    count_chunks,
    delete_collection,
    get_vectorstore,
)

DUMMY_DOCS = [
    ("The check engine light indicates an emissions system fault.",
     {"source": "dummy_a.pdf", "page": 1, "doc_type": "manual"}),
    ("P0420 means the catalytic converter efficiency is below threshold.",
     {"source": "dummy_b.csv", "page": -1, "doc_type": "obd_code"}),
    ("Use a torque wrench to tighten lug nuts to 100 Nm.",
     {"source": "dummy_c.md", "page": -1, "doc_type": "faq"}),
]
QUERY = "What does P0420 mean?"
EXPECTED_TOP_SOURCE = "dummy_b.csv"


def main() -> int:
    print(f"[1/4] Fresh throwaway collection '{settings.collection_name}'")
    delete_collection()

    print(f"[2/4] Embedding + upserting {len(DUMMY_DOCS)} dummy chunks (dense + BM25)...")
    add_chunks([
        Document(
            page_content=text,
            metadata={**meta, "language": "en", "chunk_id": f"smoke-{i}"},
        )
        for i, (text, meta) in enumerate(DUMMY_DOCS)
    ])
    print(f"      count: {count_chunks()}")

    ok = True
    for step, mode in (("3/4", "dense"), ("4/4", "hybrid")):
        hits = get_vectorstore(mode).similarity_search_with_score(QUERY, k=2)
        print(f"[{step}] {mode} query {QUERY!r}:")
        for rank, (doc, score) in enumerate(hits, start=1):
            print(f"      rank={rank} score={score:.4f} source={doc.metadata['source']!r}")
        if not hits or hits[0][0].metadata["source"] != EXPECTED_TOP_SOURCE:
            ok = False

    delete_collection()
    if ok:
        print("[OK] smoke_db.py passed")
        return 0
    print(f"[FAIL] expected top-1 source {EXPECTED_TOP_SOURCE!r} in both modes")
    return 1


if __name__ == "__main__":
    sys.exit(main())
