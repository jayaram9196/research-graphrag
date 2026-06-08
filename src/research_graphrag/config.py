from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = "changeme"
    neo4j_database: str = "neo4j"

    llm_provider: str = "openai"
    llm_model: str = "llama-3.3-70b-versatile"
    openai_api_key: str = ""
    openai_base_url: str = "https://api.groq.com/openai/v1"
    anthropic_api_key: str = ""

    embedding_provider: str = "sentence-transformers"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimensions: int = 384

    openalex_user_agent_email: str = ""

    max_papers_per_seed: int = 500
    max_citation_depth: int = 1
    batch_embedding_size: int = 32


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
