"""Smoke test for ChromaDB + embeddings wiring.

Loads the MiniLM model, wipes and re-creates the collection, adds 3 dummy
documents with rich metadata, runs a tiny query, prints the results, and
exits 0 only if the top-1 hit is the P0420 document.

Run from project root:

    python scripts/smoke_db.py
"""
from __future__ import annotations

import sys
import uuid
from pathlib import Path

# Make sure `src.*` is importable when run as a plain script from repo root.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import settings  # noqa: E402
from src.embeddings import get_embeddings  # noqa: E402
from src.vectorstore import get_client, get_collection  # noqa: E402


DUMMY_DOCS = [
    {
        "text": "The check engine light indicates an emissions system fault.",
        "metadata": {
            "source": "dummy_a.pdf",
            "page": 1,
            "doc_type": "manual",
            "language": "en",
        },
    },
    {
        "text": "P0420 means the catalytic converter efficiency is below threshold.",
        "metadata": {
            "source": "dummy_b.csv",
            # Chroma does not accept None values in metadata — use a sentinel.
            "page": -1,
            "doc_type": "obd_code",
            "language": "en",
        },
    },
    {
        "text": "Use a torque wrench to tighten lug nuts to 100 Nm.",
        "metadata": {
            "source": "dummy_c.md",
            "page": -1,
            "doc_type": "faq",
            "language": "en",
        },
    },
]

QUERY = "What does P0420 mean?"
EXPECTED_TOP_SOURCE = "dummy_b.csv"


def truncate(text: str, n: int = 100) -> str:
    return text if len(text) <= n else text[: n - 3] + "..."


def main() -> int:
    print(f"[1/6] Loading embedding model: {settings.embedding_model}")
    embeddings = get_embeddings()
    print("      model loaded.")

    print(f"[2/6] Ensuring a clean collection: {settings.collection_name}")
    client = get_client()
    try:
        client.delete_collection(name=settings.collection_name)
        print(f"      deleted existing collection '{settings.collection_name}'.")
    except Exception as e:  # noqa: BLE001
        # Acceptable: collection doesn't exist yet on fresh DB.
        print(f"      no pre-existing collection to delete ({type(e).__name__}).")

    collection = get_collection()
    print(f"      collection ready: {collection.name}")

    print(f"[3/6] Embedding and adding {len(DUMMY_DOCS)} dummy docs...")
    texts = [d["text"] for d in DUMMY_DOCS]
    metadatas = []
    ids = []
    for d in DUMMY_DOCS:
        chunk_id = str(uuid.uuid4())
        md = dict(d["metadata"])
        md["chunk_id"] = chunk_id
        metadatas.append(md)
        ids.append(chunk_id)

    vectors = embeddings.embed_documents(texts)
    print(f"      computed {len(vectors)} vectors of dim {len(vectors[0])}.")

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=vectors,
        metadatas=metadatas,
    )
    print(f"      collection count now: {collection.count()}")

    print(f"[4/6] Embedding query: {QUERY!r}")
    q_vec = embeddings.embed_query(QUERY)

    print("[5/6] Querying top-2 from the collection...")
    res = collection.query(
        query_embeddings=[q_vec],
        n_results=2,
        include=["documents", "metadatas", "distances"],
    )

    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]

    print("[6/6] Results:")
    for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists), start=1):
        page = meta.get("page")
        page_str = "None" if page in (-1, None) else page
        print(
            f"  rank={i}  distance={dist:.4f}  "
            f"source={meta.get('source')!r}  page={page_str}  "
            f"doc_type={meta.get('doc_type')!r}"
        )
        print(f"        text: {truncate(doc, 100)!r}")

    top_source = metas[0].get("source")
    if top_source == EXPECTED_TOP_SOURCE:
        print("[OK] smoke_db.py passed")
        return 0

    print(
        f"[FAIL] expected top-1 source {EXPECTED_TOP_SOURCE!r}, "
        f"got {top_source!r}. Full top results above."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
