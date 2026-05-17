from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from pathlib import Path


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # LLM
    gemini_api_key: str = ""
    openrouter_api_key: str = ""
    openrouter_model: str = "mistralai/mistral-7b-instruct"

    # App
    app_env: str = "development"
    top_k_results: int = 10
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    catalog_path: str = "data/catalog.json"
    faiss_index_path: str = "data/faiss_index"

    @property
    def catalog_file(self) -> Path:
        return Path(self.catalog_path)

    @property
    def faiss_index_dir(self) -> Path:
        return Path(self.faiss_index_path)


@lru_cache()
def get_settings() -> Settings:
    return Settings()
