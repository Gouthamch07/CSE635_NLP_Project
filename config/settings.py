from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Vertex
    google_cloud_project: str = Field(default="", alias="GOOGLE_CLOUD_PROJECT")
    google_cloud_location: str = Field(default="us-central1", alias="GOOGLE_CLOUD_LOCATION")
    google_cloud_embed_location: str = Field(default="", alias="GOOGLE_CLOUD_EMBED_LOCATION")
    vertex_model: str = Field(default="gemini-2.5-flash-lite", alias="VERTEX_MODEL")
    vertex_embed_model: str = Field(default="text-embedding-004", alias="VERTEX_EMBED_MODEL")
    vertex_thinking_budget: int = Field(default=0, alias="VERTEX_THINKING_BUDGET")
    vertex_thinking_level: str = Field(default="", alias="VERTEX_THINKING_LEVEL")

    # Pinecone
    pinecone_api_key: str = Field(default="", alias="PINECONE_API_KEY")
    pinecone_index: str = Field(default="ub-cse-chatbot", alias="PINECONE_INDEX")
    pinecone_cloud: str = Field(default="aws", alias="PINECONE_CLOUD")
    pinecone_region: str = Field(default="us-east-1", alias="PINECONE_REGION")
    pinecone_dimension: int = Field(default=768, alias="PINECONE_DIMENSION")

    # Neo4j
    neo4j_uri: str = Field(default="bolt://localhost:7687", alias="NEO4J_URI")
    neo4j_user: str = Field(default="neo4j", alias="NEO4J_USER")
    neo4j_password: str = Field(default="password", alias="NEO4J_PASSWORD")
    neo4j_database: str = Field(default="neo4j", alias="NEO4J_DATABASE")

    # Crawler
    crawl_root: str = Field(
        default="https://engineering.buffalo.edu/computer-science-engineering.html",
        alias="CRAWL_ROOT",
    )
    crawl_allowed_domains: str = Field(
        default="engineering.buffalo.edu,buffalo.edu",
        alias="CRAWL_ALLOWED_DOMAINS",
    )
    crawl_max_depth: int = Field(default=4, alias="CRAWL_MAX_DEPTH")
    crawl_max_pages: int = Field(default=1500, alias="CRAWL_MAX_PAGES")
    crawl_concurrency: int = Field(default=8, alias="CRAWL_CONCURRENCY")

    # Rerank
    rerank_model: str = Field(default="BAAI/bge-reranker-v2-m3", alias="RERANK_MODEL")
    rerank_top_k: int = Field(default=8, alias="RERANK_TOP_K")
    cross_encoder_candidate_k: int = Field(default=12, alias="CROSS_ENCODER_CANDIDATE_K")
    answer_context_k: int = Field(default=6, alias="ANSWER_CONTEXT_K")
    answer_context_chars: int = Field(default=1400, alias="ANSWER_CONTEXT_CHARS")
    answer_max_output_tokens: int = Field(default=1600, alias="ANSWER_MAX_OUTPUT_TOKENS")
    concise_answers: bool = Field(default=False, alias="CONCISE_ANSWERS")

    # App
    log_level: str = Field(default="INFO", alias="APP_LOG_LEVEL")
    enable_personal_memory: bool = Field(default=False, alias="ENABLE_PERSONAL_MEMORY")
    memory_store_path: str = Field(default="./data/memory", alias="MEMORY_STORE_PATH")
    warm_start_on_startup: bool = Field(default=True, alias="WARM_START_ON_STARTUP")
    enable_llm_planner: bool = Field(default=False, alias="ENABLE_LLM_PLANNER")
    enable_latency_debug: bool = Field(default=True, alias="ENABLE_LATENCY_DEBUG")

    @property
    def allowed_domains(self) -> list[str]:
        return [d.strip() for d in self.crawl_allowed_domains.split(",") if d.strip()]

    @property
    def data_dir(self) -> Path:
        return ROOT / "data"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
