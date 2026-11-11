# RAG POC — Overview

The short version. Full details are in [PROJECT_GUIDE.md](PROJECT_GUIDE.md).

## What it is

An offline chatbot that answers technical questions from a private set of documents. It runs on a laptop and nothing leaves the machine. For every question it finds the most relevant passages, then asks a local LLM to answer **only from those passages**, citing them, in the user's language. If the passages don't contain the answer, it refuses with a fixed sentence.

- **Current data:** 2017 Ford F-150 owner's manual (English + French) and a list of 469 OBD-II diagnostic codes.
- **Real target:** ElBATIMA (building and public works). The car data is a rehearsal. Switching domains should only mean new documents, new eval questions, a new persona line in the prompt and new `.env` values.

## Stack

| Part | Tool |
|---|---|
| Pipeline | LangChain |
| Vector database | ChromaDB (local, cosine similarity) |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` (multilingual, CPU) |
| LLM | Mistral 7B Instruct, served locally by Ollama |
| API | FastAPI + Pydantic |

## How it works

Two paths share one database. Follow the numbers.

```text
  OFFLINE · build the knowledge base   (python -m src.ingest)

  data/raw/                   F-150 manuals (EN + FR PDF) · OBD codes CSV
      │
     [1]   load each file, tag its language (en / fr)
      ▼
  Loaders                     src/loaders.py
      │
     [2]   split: a CSV row stays whole, PDF text -> 800-char chunks
      ▼
  Chunker                     src/chunker.py
      │
     [3]   turn each chunk into a vector (384 numbers)
      ▼
  Embedding model (MiniLM)    src/embeddings.py
      │
     [4]   save vector + text + source file + page
      ▼
  ChromaDB · 4,556 chunks     data/chroma/


  ONLINE · answer one question   (POST /chat)

  Client                      curl or a frontend
      │
     [5]   question + language ("auto", "en" or "fr")
      ▼
  FastAPI /chat               src/api/routes.py
      │
     [6]   detect the question's language, start the chain
      ▼
  Retriever                   src/retriever.py
      │    [7]  embed the question with the same model as [3]
      │    [8]  search ChromaDB -> 6 best chunks, favoring the question's language
      ▼
  Prompt                      src/prompt.py
      │
     [9]   rules + the 6 chunks + the question
      ▼
  Mistral 7B via Ollama       src/llm.py   (runs locally)
      │
     [10]  answer with inline citations + list of sources
      ▼
  Client
```

The key ideas behind the steps:

- **[2] Two chunking rules.** A diagnostic code row (`P0420 → Catalyst System Efficiency Below Threshold`) is useless if cut, so each row stays one chunk. Manual text is split into chunks of up to 800 characters, with 150 characters of overlap so a sentence cut at a boundary survives in one of them.
- **[3] and [7] use the same model.** Questions and chunks must live in the same vector space. Because the model is multilingual, a French question can still match English text.
- **[8] MMR search.** The retriever takes the 30 most similar chunks, then picks 6 that are relevant *and* different from each other, so the LLM doesn't get the same paragraph six times. Four of the six slots go to chunks in the question's language; the other two stay open to any language.
- **[9] The prompt rules:** answer only from the chunks, cite `(file, page)` after each fact, answer in the user's language, never invent codes or part numbers, and otherwise reply *"I don't have enough information in my knowledge base to answer that reliably."* Temperature is 0.1 for near-deterministic answers.

## Evaluation

### How we measure

`eval/run_eval.py` runs 21 fixed questions (15 English, 6 French) through the real pipeline: 6 code lookups, 5 maintenance, 5 symptoms, 4 procedures, and 1 question the documents can't answer. Each question lists the file the answer should come from and keywords the answer should contain.

| Metric | Question it answers |
|---|---|
| **hit@k** | Did a chunk from the right file appear among the k retrieved? |
| **MRR** | How high did the right file rank? (1st = 1.0, 2nd = 0.5, 4th = 0.25) |
| **Keyword recall** | What share of the expected keywords appears in the answer? |
| **Refusal** | Does the model refuse the out-of-scope question? |

A full run takes about 7 minutes and is saved as JSON in `eval/results/`.

### What we did

1. **Day 5 baseline.** First measurement. We later found the retriever had been running plain similarity search instead of MMR, because of a bug.
2. **Day 6 fix and tuning.** Fixed the bug, then tested one change at a time: chunk size 600 and 1,000, 6 chunks instead of 4, MMR λ 0.3 and 0.7, similarity instead of MMR, and a stricter prompt. **The only clear winner was 6 chunks instead of 4**, which was committed. The stricter prompt made the model refuse too often.
3. **September: language-aware answers (not committed yet).** The `language` field in requests was being ignored. It now drives both the answer language and which chunks retrieval favors.

### Results

| Run | hit@k | MRR | Keyword recall | Refusal |
|---|---|---|---|---|
| Day 5 baseline (4 chunks, similarity) | 95% | 0.72 | 42.5% | 1/1 |
| **Day 6 committed (6 chunks, MMR)** | **100%** | **0.73** | **53.2%** | **1/1** |
| September, language-aware | 100% | 1.00 | 47.6% | 1/1 |

### What the numbers mean

- **Retrieval looks perfect but isn't.** hit@k and MRR only check the *file*. For "What does P0420 mean?", any code row counts as a hit, even when the P0420 row itself is missing, and it usually is: it ranks 348th of 4,556. The model correctly refuses, so keyword recall is 0. The embedding model treats all `P0xxx` codes as nearly the same.
- **Answer wording is the main gap.** Only about half the expected keywords appear, partly because Mistral paraphrases and partly because the check needs exact matches ("800 kilomètres" does not match "8 000 km").
- **The language change helps ranking but not answers yet.** English questions no longer get French chunks first, so MRR went to 1.00. But keyword recall dropped and it's a single run, so rerun it before committing.
- **Not measured yet:** hallucination (the optional RAGAS check has never been run). The refusal score comes from one question only.

## Main issues and next steps

1. **Exact code lookup fails.** Add keyword search (BM25) alongside vector search. This matters even more for ElBATIMA, whose documents are full of norm and article numbers.
2. **Make the eval honest.** Check that the right *passage* was retrieved, not just the right file, and accept synonyms in keyword checks.
3. **Rerun the language-aware eval 2–3 times**, then commit it or tune it.
4. **Retest chunk size 1,000.** Day 6 compared it against a mislabeled baseline; against the real one, it did better. The 800 default is a fixed project rule, so changing it needs your decision.
5. **Small fixes:** the CSV header line is stored as a junk chunk, and some English code rows are tagged French.

## Run it

```powershell
.\.venv\Scripts\Activate.ps1
python -m src.ingest                                         # build the database
uvicorn src.api.main:app --host 127.0.0.1 --port 8000        # start the API (Ollama must be running)
python eval/run_eval.py                                      # evaluation
pytest -m "not requires_llm"                                 # tests (54 passing)
```
