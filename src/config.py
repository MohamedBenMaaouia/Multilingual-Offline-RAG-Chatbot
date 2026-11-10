from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Literal


class Settings(BaseSettings):
    # paths
    data_raw_path: Path = Path("./data/raw")
    chroma_path: Path = Path("./data/chroma")

    # ingestion
    chunk_size: int = 800
    chunk_overlap: int = 150

    # embeddings
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    collection_name: str = "kb_auto"

    # retrieval
    top_k: int = 6
    fetch_k: int = 30
    mmr_lambda: float = 0.5
    search_type: Literal["mmr", "similarity"] = "mmr"

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

    # api
    cors_origins: list[str] = ["http://localhost:3000"]
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # domain
    assistant_persona: str = "automotive repair and maintenance"

    class Config:
        env_file = ".env"


settings = Settings()
