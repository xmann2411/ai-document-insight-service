"""
Central configuration, loaded from environment variables (or a .env file).

Every tunable lives here so the behaviour can be changed without touching
code - useful for the Docker image (env vars) vs. local runs (.env file).
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- LLM (Anthropic Claude) ---
    anthropic_api_key: str | None = None
    llm_model: str = "claude-haiku-4-5"
    llm_max_tokens: int = 600

    # --- Retrieval / RAG ---
    # "embedding" -> chunk + embed + FAISS similarity search (default).
    # "full"      -> send every stored document in full as context.
    #                Fallback for environments where sentence-transformers /
    #                torch cannot be installed (e.g. very new Python versions).
    retrieval_mode: str = "embedding"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 900          # characters per chunk
    chunk_overlap: int = 150       # character overlap between consecutive chunks
    top_k: int = 4                 # chunks retrieved per question

    # --- OCR ---
    # OCR (EasyOCR) is used for image uploads and for PDFs with no text layer.
    ocr_enabled: bool = True
    ocr_languages: str = "en"      # comma-separated, e.g. "en,hr"

    # --- Uploads ---
    max_upload_mb: int = 25

    @property
    def ocr_language_list(self) -> list[str]:
        return [lang.strip() for lang in self.ocr_languages.split(",") if lang.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
