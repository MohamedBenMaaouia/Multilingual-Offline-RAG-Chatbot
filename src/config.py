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

    # llm
    llm_model: str = "mistral:7b-instruct"
    ollama_url: str = "http://localhost:11434"
    temperature: float = 0.1
    max_tokens: int = 512
    llm_timeout_s: int = 60

    # api
    cors_origins: list[str] = ["http://localhost:3000"]
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # domain
    assistant_persona: str = "automotive repair and maintenance"

    class Config:
        env_file = ".env"


settings = Settings()
