from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Literal, Optional


class Settings(BaseSettings):
    # paths
    data_raw_path: Path = Path("./data/raw")

    # vector store — Qdrant. Embedded (local folder) unless qdrant_url is set.
    qdrant_path: Path = Path("./data/qdrant")
    qdrant_url: Optional[str] = None
    qdrant_api_key: Optional[str] = None

    # ingestion
    chunk_size: int = 800
    chunk_overlap: int = 150

    # embeddings
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    collection_name: str = "kb_auto"

    # sparse (BM25) encoder for hybrid search — FastEmbed model name
    sparse_model: str = "Qdrant/bm25"
    # Each chunk stores one BM25 vector per language (English and French
    # stemming/stopwords). True: a question uses its own language's vector.
    # False: always the English one (the pre-Nov-10 behaviour, for A/B evals).
    bm25_per_language: bool = True

    # retrieval
    # "dense": MiniLM vectors only, ranked by search_type (mmr | similarity).
    # "hybrid": dense + BM25 searched together, fused with RRF in Qdrant;
    #           search_type and mmr_lambda do not apply (MMR is dense-only).
    # Hybrid is the default since Nov 2026: passage hit@6 40% -> 80% and
    # diagnostic-code questions 0/6 -> 6/6 on the eval set.
    retrieval_mode: Literal["dense", "hybrid"] = "hybrid"
    top_k: int = 6
    # Candidate pool considered before the final top_k is chosen — by MMR in
    # dense mode, or by the cross-encoder when reranking is on.
    fetch_k: int = 30
    mmr_lambda: float = 0.5
    search_type: Literal["mmr", "similarity"] = "mmr"

    # Cross-encoder reranking of the fetch_k candidates down to top_k.
    # On since Nov 2026: passage MRR 0.55 -> 0.60, every code question ranks
    # its row first; costs ~2 s per question on CPU.
    reranker_enabled: bool = True
    reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

    # Share of top_k reserved for chunks matching the request language.
    # The remaining slots stay language-agnostic so cross-language content
    # (e.g. an English-only code table) is still reachable from a French
    # question. 0.0 disables language-aware retrieval entirely.
    language_pref_ratio: float = 0.67

    # llm
    llm_model: str = "mistral:7b-instruct"
    ollama_url: str = "http://localhost:11434"
    temperature: float = 0.1
    max_tokens: int = 512
    llm_timeout_s: int = 60

    # evaluation — RAGAS LLM-as-judge. Deliberately a different, larger model
    # than llm_model: a model grading its own answers is a weak judge, and
    # RAGAS needs reliable structured (JSON) output that 7B models often break.
    ragas_judge_model: str = "qwen2.5:14b"
    ragas_judge_num_ctx: int = 8192
    ragas_judge_timeout_s: int = 600

    # answer cache (src/cache.py) — Redis Stack (vector search) required.
    # If Redis is unreachable the cache bypasses itself; /chat keeps working.
    cache_enabled: bool = True
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_s: int = 7 * 24 * 3600
    # Minimum cosine similarity between two questions for a semantic hit.
    # Calibrated 2026-11-10 with MiniLM: different questions reach 0.77
    # ("check oil level" vs "check coolant level"), close rewordings 0.86-0.97.
    # 0.88 keeps a ~0.1 margin over the worst lookalike; loose paraphrases
    # (~0.6-0.75) miss, which is the safe failure.
    cache_similarity_threshold: float = 0.88

    # prompt-injection guardrails (src/guardrails.py): question screening,
    # fencing of untrusted text, dropping injected chunks, answer leak check.
    guardrails_enabled: bool = True

    # api
    cors_origins: list[str] = ["http://localhost:3000"]
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # domain
    assistant_persona: str = "automotive repair and maintenance"

    class Config:
        env_file = ".env"


settings = Settings()
