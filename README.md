# Multilingual Offline RAG Chatbot

A laptop-runnable retrieval-augmented generation system built on LangChain, ChromaDB, Ollama (Mistral 7B Instruct), and FastAPI. Answers technical questions in English or French, grounded in a private document corpus, with **zero external API calls** and **zero data leaving the machine**.

The reference corpus that ships with this repo is automotive — Ford F-150 owner's manuals (English + French) plus a 469-row SAE J2012 OBD-II diagnostic code list. The pipeline itself is domain-agnostic: swap the documents and the prompt persona, and everything else runs unchanged.

---

## At a glance

| Metric (on the bundled corpus) | Score |
|---|---|
| Retrieval `hit@k=6` | **100%** (20/20 grounded questions) |
| Mean Reciprocal Rank | 0.73 |
| Keyword recall in answers | 53% |
| Refusal-when-out-of-corpus correctness | 100% (1/1) |
| Test suite | 31 tests, ~30s, no live LLM required |
| End-to-end query latency (CPU) | 5–15s (dominated by LLM generation) |

All numbers reproducible from a clean ingest by running `python eval/run_eval.py`. The committed retrieval config is the winner of an eval-driven tuning matrix documented in [`TUNING.md`](TUNING.md).

---

## Architecture

Two pipelines share a single ChromaDB collection.

**Offline ingestion** reads raw documents, splits them with a structure-aware strategy, embeds the chunks via a multilingual sentence-transformer, and persists vectors + metadata to disk.

**Online query** embeds the incoming question with the same model, retrieves relevant chunks via Maximal Marginal Relevance (MMR), stuffs them into a grounded prompt, and lets a local Mistral 7B generate the answer.

```
                          INGESTION PIPELINE (offline)
   ┌──────────────┐
   │  data/raw/   │   PDFs · CSV · Markdown
   └──────┬───────┘
          ▼
   ┌──────────────┐
   │   Loaders    │   PyPDFLoader · CSVLoader · UnstructuredMarkdownLoader
   │              │   + langdetect for per-document language tagging
   └──────┬───────┘
          ▼
   ┌──────────────┐
   │   Chunker    │   Dual strategy:
   │              │   · CSV rows  → atomic (no splitting)
   │              │   · PDF / MD  → recursive split (800 chars, 150 overlap)
   └──────┬───────┘
          ▼
   ┌──────────────┐
   │  Embedder    │   paraphrase-multilingual-MiniLM-L12-v2
   │              │   384-dim vectors · normalized · CPU
   └──────┬───────┘
          ▼
   ┌──────────────┐
   │  ChromaDB    │   PersistentClient at data/chroma/
   │  collection  │   HNSW index, cosine space
   └──────────────┘


                            QUERY PIPELINE (online)
   POST /chat ──► Pydantic validation
                       │
                       ▼
                 Empty-collection guard ──► canned refusal
                       │
                       ▼
                ┌──────────────┐
                │ Embed query  │   same MiniLM singleton as ingestion
                └──────┬───────┘
                       ▼
                ┌──────────────┐
                │ MMR retriever│   fetch_k=30 candidates → top_k=6
                │              │   λ=0.5 balances relevance vs diversity
                └──────┬───────┘
                       ▼
                ┌──────────────┐
                │ Prompt build │   each chunk prefixed:
                │              │   [source: file.pdf, page: 47] <chunk_text>
                └──────┬───────┘
                       ▼
                ┌──────────────┐
                │  Mistral 7B  │   Ollama-served, T=0.1, 512 max tokens
                └──────┬───────┘
                       ▼
                ┌──────────────┐
                │   Response   │   { answer, sources[], latency_ms, retrieved_k }
                └──────────────┘
```

---

## Technology stack — where it lives, why it was chosen

| Component | Library / Tool | Location in code | Rationale |
|---|---|---|---|
| **Orchestration** | LangChain | All `src/` modules | Standardized `Document` interface, retriever protocol, prompt templating. `create_retrieval_chain` + `create_stuff_documents_chain` give a clean composition pattern; each stage is swappable independently. |
| **Document loaders** | `langchain_community.document_loaders` (PyPDFLoader, CSVLoader, UnstructuredMarkdownLoader) | `src/loaders.py` | Wide format coverage. PyPDF preserves per-page metadata, CSVLoader produces row-level Documents, Unstructured handles MD logical blocks. |
| **Text splitting** | `RecursiveCharacterTextSplitter` | `src/chunker.py` | Recursive separator hierarchy (`\n\n` → `\n` → `. ` → ` ` → `""`) preserves paragraph and sentence boundaries when possible. Far better than fixed-window splits for technical prose. |
| **Embedding model** | `paraphrase-multilingual-MiniLM-L12-v2` via `langchain_huggingface.HuggingFaceEmbeddings` | `src/embeddings.py` (singleton) | Supports 50+ languages including EN/FR/AR. 384-dim vectors keep the index lean. ~120 MB download. Runs on CPU at ~2,500 sentences/sec — fast enough for a laptop ingest. `normalize_embeddings=True` so cosine math behaves. |
| **Vector store** | ChromaDB `PersistentClient` | `src/vectorstore.py` (singleton). Data at `data/chroma/`. | Embedded SQLite-backed persistence, no separate server process. HNSW gives sub-millisecond retrieval over tens of thousands of chunks. Collection metadata `{"hnsw:space": "cosine"}` pairs with normalized embeddings. |
| **Retrieval algorithm** | Maximal Marginal Relevance | `src/retriever.py` — `top_k=6`, `fetch_k=30`, `λ=0.5` | Pure cosine similarity surfaces near-duplicate chunks (same passage from multiple pages, same row in different orderings). MMR diversifies the candidate set so distinct passages reach the LLM. The exact k/fetch_k/λ values are the matrix winners in [`TUNING.md`](TUNING.md). |
| **LLM** | Ollama serving `mistral:7b-instruct` (Q4_K_M, ~4.4 GB) | `src/llm.py` via `langchain_ollama.ChatOllama` | Local, free, no API keys, no rate limits. Mistral 7B has strong French instruction-following — a hard requirement for the multilingual claim. 4-bit quantization makes it tractable on CPU (~5–15s per generation). `temperature=0.1` because factual technical Q&A wants determinism. |
| **API layer** | FastAPI + uvicorn | `src/api/` (`main.py`, `routes.py`, `schemas.py`) | Async-ready, automatic request validation, OpenAPI docs for free. Lifespan context manager warms singletons (embedder, Chroma client, Ollama health probe) on startup. CORS middleware exposes the API to a frontend cleanly. |
| **Schema validation** | Pydantic v2 | `src/api/schemas.py` — six models | Type-safe boundaries between client/server. Bad payloads get a 422 automatically. |
| **Configuration** | `pydantic-settings` | `src/config.py` — single `Settings` class | Reads from `.env` and environment variables. All tunables in one place. Env-var overrides make tuning runs ephemeral and reproducible. |
| **Language detection** | `langdetect` | `src/loaders.py` — first 500 chars per Document | Stamps `metadata.language` so chunks can be filtered by language later. Confidence threshold of 0.7; falls back to `en` if uncertain. |
| **Eval harness** | Custom (`eval/run_eval.py`) + optional RAGAS | `eval/` | Deterministic metrics (hit@k, MRR, keyword recall) computed by hand — cheap, reproducible, CI-friendly. RAGAS hooked behind `--ragas` flag for LLM-as-judge faithfulness and answer relevance when wanted. |
| **Testing** | pytest + pytest-mock + httpx `TestClient` | `tests/` — 31 tests | Loaders, chunker, prompt, retriever, API all covered. ChatOllama mocked so the default suite needs no Ollama. One test explicitly asserts retriever wiring matches `Settings` to catch silent config drift. |

---

## Under the hood — ingestion (point A → Z)

1. **Files land in `data/raw/`.** PDFs in `data/raw/manuals/`, the OBD CSV at `data/raw/obd_codes.csv` (with a `code,description` header), Markdown files under `data/raw/faq/`.

2. **`src/ingest.py` walks `data/raw/` recursively** and dispatches each file to the appropriate loader:
   - PDFs → `load_pdfs()` → `PyPDFLoader` yields **one `Document` per page**, preserving the page number.
   - CSV → `load_csv()` → `CSVLoader` with explicit `fieldnames=["code", "description"]` yields **one `Document` per row**. The explicit fieldnames let it ingest headerless CSVs without misreading the first row as column names.
   - Markdown → `load_markdown()` → `UnstructuredMarkdownLoader` yields a `Document` per logical block.

3. **Every `Document` gets metadata stamped**:
   - `source` — relative path from `data/raw/` (e.g. `manuals/2017-F-150-Owners-Manual-English.pdf`).
   - `page` — integer page number for PDFs; the sentinel `-1` for pageless documents (CSV rows, MD blocks). ChromaDB metadata rejects `None`, so `-1` is the storage-side stand-in. The API layer normalizes `-1` back to `null` at the response boundary.
   - `doc_type` — `"manual"` | `"obd_code"` | `"faq"`.
   - `language` — `"en"` | `"fr"` | `"ar"`, detected by `langdetect` on the first 500 characters, with a 0.7 confidence threshold (falls back to `en`).
   - `chunk_id` — a fresh `uuid4` (regenerated again per output chunk after splitting).

4. **The chunker applies a dual strategy** (`src/chunker.py`):
   - **CSV rows are atomic.** Each row's text becomes one chunk verbatim. Splitting `P0420,Catalyst System Efficiency Below Threshold (Bank 1)` mid-string would lose the code-meaning pairing.
   - **PDF and Markdown go through `RecursiveCharacterTextSplitter`** with `chunk_size=800`, `chunk_overlap=150`, and separators `["\n\n", "\n", ". ", " ", ""]`. The recursive separator order keeps paragraph and sentence boundaries intact when possible. Overlap preserves sentences cut across chunk boundaries.
   - Every output chunk receives a **fresh `chunk_id`** (`uuid4`).

5. **Embeddings are computed** by `HuggingFaceEmbeddings` (singleton — loaded once per process, never re-instantiated). The MiniLM model returns 384-dim vectors with `normalize_embeddings=True`, so they're unit-length and cosine distance behaves well-defined.

6. **Vectors persist to ChromaDB.** `PersistentClient` writes to `data/chroma/`. The collection is configured with `metadata={"hnsw:space": "cosine"}` — paired with the normalized embeddings, that's a consistent similarity space. Add calls are **batched in 4,000-item chunks** to respect ChromaDB's 5,461-item per-call cap.

A typical ingest of the bundled corpus (2 PDFs of ~1,300 pages each + the 469-row OBD CSV) produces ~4,500 chunks in ~2 minutes on CPU.

---

## Under the hood — query (point A → Z)

1. **Client sends `POST /chat`** with `{"question": "...", "language": "auto"}`. Pydantic validates the question is 3–1,000 characters and the language is `auto`/`en`/`fr`/`ar`; bad payloads get a 422.

2. **Empty-collection guard.** If `get_collection().count() == 0`, the route short-circuits and returns the canned refusal with `sources=[]` — no LLM call, no wasted latency. This matters for fresh-deploy / pre-ingest state.

3. **The chain is invoked** (`src/chain.py`). Built once per process via `get_chain()` — wraps `create_stuff_documents_chain` (LLM + prompt + per-document formatter) inside `create_retrieval_chain` (retriever + doc chain). The `"input"` key is what `create_retrieval_chain` expects by default.

4. **The question is embedded** by the same MiniLM singleton used during ingestion. The retriever now has a 384-dim query vector to work with.

5. **MMR retrieval** (`src/retriever.py`):
   - Chroma's HNSW index returns the top **`fetch_k=30`** candidates ranked by cosine similarity.
   - MMR then re-ranks those 30, selecting **`top_k=6`** chunks that balance similarity to the question against dissimilarity to already-chosen chunks (`λ=0.5` is the equal-weight balance).
   - Output: 6 `Document` objects with their original metadata intact.

6. **Documents are wrapped with a citation prefix** before they enter the prompt context. A `RunnableLambda` rewrites each chunk's representation to `[source: <filename>, page: <n>] <chunk_text>`. The sentinel `page=-1` is rendered as `page: N/A` in the prompt. The original Documents (with `page=-1` intact) are kept alongside for the API response — the sentinel-to-null normalization happens at the API boundary, not in the prompt.

7. **The prompt is assembled** (`src/prompt.py`):
   - **System message**: an expert assistant persona, parameterized via `{persona}` from `settings.assistant_persona` so the same prompt code works for any domain.
   - **Strict grounding rules**: refuse with the exact canned line when context is insufficient; cite sources inline as `(filename, p.N)` after each factual claim; never invent identifiers (codes, part numbers); reply in the user's language.
   - **`{context}` slot**: the 6 formatted chunks, each with its source/page prefix.
   - **`{input}` slot**: the user's question, literal text.

8. **Mistral 7B generates an answer** via `ChatOllama` over HTTP to `localhost:11434`. `temperature=0.1` keeps generation factual; `num_predict=512` caps output length; 60s request timeout.

9. **The response is assembled** (`src/api/routes.py`):
   - `result["context"]` (the original Documents with `page=-1` intact) is mapped to `SourceRef` objects: `source` (string), `page` (normalized: `-1` → `null`), `doc_type`, `snippet` (first 200 chars of the chunk).
   - `latency_ms` is measured server-side around the chain call.
   - `retrieved_k` is `len(result["context"])`.

10. **The `ChatResponse` JSON returns** to the client.

End-to-end CPU latency is typically 5–15 seconds, almost entirely Mistral generation time. The retrieval step itself is sub-100ms. A GPU or smaller model would speed up the bottleneck dramatically.

---

## Grounding rules (the contract)

These are enforced by the system prompt and verified by the eval harness:

1. **Refuse** with the canned line *"I don't have enough information in my knowledge base to answer that reliably."* when the retrieved context is insufficient.
2. **Cite sources inline** as `(filename, p.N)` after each factual claim.
3. **Reply in the user's language** — EN / FR / AR.
4. **Never invent identifiers** — part numbers, diagnostic codes, article references.
5. **Temperature stays at 0.1.** No "more natural" creativity at the expense of correctness.

The eval harness includes one out-of-corpus question specifically to test rule 1; refusal correctness is reported as its own metric.

---

## Project structure

```
.
├── ARCHITECTURE.md            design contract; reference for internal changes
├── TUNING.md                  eval-driven tuning matrix and methodology
├── README.md                  this file
├── DEMO.md                    2-minute demo shooting script
├── Makefile                   ingest / serve / eval / test / reset-db targets
├── pyproject.toml             dependencies + pytest config + project metadata
├── .env.example               all configurable knobs documented
│
├── data/
│   ├── raw/                   source documents (gitignored)
│   │   ├── manuals/           PDFs
│   │   ├── obd_codes.csv      diagnostic codes
│   │   └── faq/               markdown
│   └── chroma/                persisted vector store (gitignored)
│
├── src/
│   ├── config.py              Settings (pydantic-settings, env-driven)
│   ├── loaders.py             PDF/CSV/MD loaders + langdetect metadata
│   ├── chunker.py             dual chunking strategy
│   ├── embeddings.py          MiniLM singleton
│   ├── vectorstore.py         PersistentClient + cosine HNSW collection
│   ├── ingest.py              orchestrator (CLI: python -m src.ingest)
│   ├── retriever.py           MMR retriever, settings-driven
│   ├── prompt.py              system prompt + document-prompt template
│   ├── llm.py                 ChatOllama singleton
│   ├── chain.py               full retrieval chain assembly
│   └── api/
│       ├── main.py            FastAPI app + lifespan + CORS
│       ├── routes.py          /chat /health /ingest
│       └── schemas.py         Pydantic v2 request/response models
│
├── tests/
│   ├── conftest.py            tmp_chroma, sample_docs, mock_llm fixtures
│   ├── test_loaders.py        metadata schema, headerless CSV, missing dirs
│   ├── test_chunker.py        size limits, overlap, CSV atomicity, chunk_ids
│   ├── test_prompt.py         substitution, canned refusal, brace escaping
│   ├── test_retriever.py      settings-wiring assertion + end-to-end
│   ├── test_api.py            mocked LLM, empty-corpus fallback, /health
│   └── fixtures/              sample.pdf, sample.csv, sample.md
│
├── eval/
│   ├── qa_set.yaml            21 Q&A pairs (20 grounded + 1 refusal test)
│   ├── run_eval.py            hit@k, MRR, keyword recall, optional RAGAS
│   └── results/               timestamped JSON reports (gitignored)
│
└── scripts/
    ├── smoke_db.py            embed+store regression (no LLM)
    ├── smoke_test.py          end-to-end regression (with LLM)
    ├── ingest_all.sh          wipe + re-ingest
    ├── reset_db.sh            wipe chroma/
    └── pull_ollama_model.sh   ollama pull mistral:7b-instruct
```

---

## Configuration

All tunables live in [`src/config.py`](src/config.py) and are environment-overridable. The most impactful ones:

| Knob | Default | Effect |
|---|---|---|
| `chunk_size` | 800 | Character limit per chunk for PDF/MD content |
| `chunk_overlap` | 150 | Overlap between adjacent chunks |
| `top_k` | 6 | Chunks delivered to the LLM per query |
| `fetch_k` | 30 | MMR candidate pool size before re-ranking |
| `mmr_lambda` | 0.5 | Relevance-vs-diversity balance (0 = pure diversity, 1 = pure relevance) |
| `search_type` | `"mmr"` | `"mmr"` or `"similarity"` |
| `embedding_model` | `paraphrase-multilingual-MiniLM-L12-v2` | HuggingFace model name |
| `collection_name` | `kb_auto` | ChromaDB collection name |
| `llm_model` | `mistral:7b-instruct` | Ollama model tag |
| `temperature` | 0.1 | LLM sampling temperature |
| `max_tokens` | 512 | Cap on LLM output length |
| `assistant_persona` | `automotive repair and maintenance` | Filled into the system prompt's `{persona}` slot |

Copy `.env.example` to `.env` to override any of these.

---

## Setup and run

Requires **Python ≥ 3.11** and a local [Ollama](https://ollama.com/) install.

```bash
# 1. Clone and install
git clone <repo-url>
cd <repo>
python -m venv .venv

# Activate venv:
source .venv/Scripts/activate          # Git Bash on Windows
# or: .\.venv\Scripts\Activate.ps1     # PowerShell
# or: source .venv/bin/activate        # macOS / Linux

pip install -e .

# 2. Pull the LLM (~4.4 GB, one-time)
ollama pull mistral:7b-instruct

# 3. Drop your documents
#    PDFs  → data/raw/manuals/
#    CSV   → data/raw/obd_codes.csv  (with header: code,description)
#    MD    → data/raw/faq/

# 4. Ingest (~2 min on CPU for a multi-PDF corpus)
python -m src.ingest

# 5. Serve
uvicorn src.api.main:app --host 127.0.0.1 --port 8000

# 6. Query
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "your question here"}'

# 7. Evaluate
python eval/run_eval.py

# 8. Test
pytest -m "not requires_llm" -v
```

On macOS/Linux or in Git Bash, `make ingest`, `make serve`, `make eval`, `make test`, `make reset-db` work as shortcuts. On native Windows shells (PowerShell, cmd), use the underlying commands directly — see [`DEMO.md`](DEMO.md) for shell-specific examples.

The multilingual MiniLM embedding model (~120 MB) downloads from HuggingFace on first use and is cached under `~/.cache/huggingface/`.

---

## API endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/chat` | Submit a question, get answer + sources + latency |
| `GET` | `/health` | Liveness check — verifies ChromaDB readable and Ollama reachable |
| `POST` | `/ingest` | Admin: trigger re-ingestion (optional `wipe_first` payload) |

**Example response shape:**

```json
{
  "answer": "Change engine oil every 8,000 km or six months... (manuals/2017-F-150-Owners-Manual-French.pdf, p.626)",
  "sources": [
    {"source": "manuals/2017-F-150-Owners-Manual-French.pdf", "page": 626, "doc_type": "manual",   "snippet": "Changer l'huile moteur et le filtre..."},
    {"source": "obd_codes.csv",                                "page": null, "doc_type": "obd_code", "snippet": "code: P0420 / description: Catalyst System..."}
  ],
  "latency_ms": 8421,
  "retrieved_k": 6
}
```

**Out-of-corpus questions** return the canned refusal with `sources: []` and `retrieved_k: 0`.

---

## Known limitations

- **Images and tables in PDFs are lost.** `PyPDFLoader` extracts text only. Wiring diagrams, exploded views, structured torque-spec tables — none of these become retrievable. Adding multimodal retrieval would be a larger architecture change.
- **Keyword recall (~53%) is the dominant remaining quality gap.** Retrieval is solid (`hit@k=100%`), but Mistral 7B tends to paraphrase retrieved context rather than echoing the exact technical vocabulary. Larger LLMs or prompt few-shots are the natural next experiments.
- **French answers occasionally skip inline citations** despite the prompt rule. Mistral 7B's instruction-following degrades on French.
- **Some language detection misses.** ~10% of French PDF chunks get tagged `language="en"` because their first 500 characters are tables, numerics, or page headers where `langdetect` falls below the confidence threshold.
- **Latency is generation-bound, not retrieval-bound.** Vector search is sub-100ms; Mistral on CPU is 5–15s per answer. A GPU or smaller LLM would shift the bottleneck.

---

## References

- Design contract and rationale: [`ARCHITECTURE.md`](ARCHITECTURE.md)
- Tuning history and methodology: [`TUNING.md`](TUNING.md)
- Demo recording script: [`DEMO.md`](DEMO.md)
