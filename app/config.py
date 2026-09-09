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
    # "local"      -> sentence ranking with the cross-encoder (default). Free,
    #                 offline, no key. Answer is a verbatim sentence.
    # "distilbert" -> DistilBERT SQuAD span extraction via onnxruntime. Free,
    #                 offline. Answer is a short extracted span.
    # "claude"     -> Anthropic Claude via API. Rephrased / synthesised answers,
    #                 needs ANTHROPIC_API_KEY, costs a few cents.
    qa_backend: str = "local"
    qa_max_answer_chars: int = 400

    # distilbert backend
    qa_local_model: str = "Xenova/distilbert-base-cased-distilled-squad"

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
    top_k: int = 6                 # chunks passed to the QA backend
    # FAISS returns this many candidates, then the cross-encoder reranks them
    # down to top_k. A cheap, large accuracy win (see app/rag.py).
    rerank_enabled: bool = True
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    rerank_candidates: int = 12

    # --- Named Entity Recognition ---
    # Highlights entities (people, orgs, places, money, dates, %) in answers.
    # Transformer model (CoNLL-2003) via onnxruntime + regex for money/dates.
    ner_enabled: bool = True
    ner_model: str = "Xenova/bert-base-NER"

    # --- OCR ---
    # OCR (Tesseract via pytesseract) reads image uploads and PDFs that have
    # no text layer. Needs the `tesseract` binary on PATH (bundled in Docker).
    ocr_enabled: bool = True
    ocr_languages: str = "eng"     # Tesseract lang codes, e.g. "eng+hrv"

    # --- Answer cache ---
    # In-process LRU cache for /ask results, keyed by (session, question,
    # backend). Swap for Redis in production - same get/put/invalidate API.
    cache_enabled: bool = True
    cache_size: int = 256

    # --- Uploads ---
    max_upload_mb: int = 25

    @property
    def ocr_lang_string(self) -> str:
        return self.ocr_languages.replace(",", "+")


@lru_cache
def get_settings() -> Settings:
    return Settings()
