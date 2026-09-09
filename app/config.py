"""
Central configuration, loaded from environment variables (or a .env file).

Every tunable lives here so behaviour can change without touching code -
useful for the Docker image (env vars) vs. local runs (.env file).

The whole stack is deliberately torch-free so it installs and runs on any
modern Python (3.11 - 3.14) with no GPU and no heavyweight ML framework.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Question answering backend ---
    # "local"  -> DistilBERT SQuAD run locally via onnxruntime. Free, offline,
    #             no API key. This is the default.
    # "claude" -> Anthropic Claude via API. Better answers (full sentences,
    #             synthesis), needs ANTHROPIC_API_KEY and costs a few cents.
    qa_backend: str = "local"

    # local backend
    qa_local_model: str = "Xenova/distilbert-base-cased-distilled-squad"
    qa_max_answer_chars: int = 320

    # claude backend
    anthropic_api_key: str | None = None
    llm_model: str = "claude-haiku-4-5"
    llm_max_tokens: int = 600

    # --- Retrieval / RAG ---
    # "embedding" -> chunk + embed (fastembed / ONNX) + FAISS similarity search.
    # "full"      -> skip retrieval, pass every chunk to the QA backend.
    retrieval_mode: str = "embedding"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    chunk_size: int = 900          # characters per chunk
    chunk_overlap: int = 150       # character overlap between consecutive chunks
    top_k: int = 4                 # chunks retrieved per question

    # --- OCR ---
    # OCR (Tesseract via pytesseract) reads image uploads and PDFs that have
    # no text layer. Needs the `tesseract` binary on PATH (bundled in Docker).
    ocr_enabled: bool = True
    ocr_languages: str = "eng"     # Tesseract lang codes, e.g. "eng+hrv"

    # --- Uploads ---
    max_upload_mb: int = 25

    @property
    def ocr_lang_string(self) -> str:
        return self.ocr_languages.replace(",", "+")


@lru_cache
def get_settings() -> Settings:
    return Settings()
