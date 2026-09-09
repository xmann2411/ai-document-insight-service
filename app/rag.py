"""
Retrieval-Augmented Generation helpers.

Why RAG here: sending whole documents to the QA backend works for one
short file but breaks down as documents get longer or more numerous - too
much context (cost / latency / the 512-token limit of the local model) and
more noise. Instead we:

  1. split each document into overlapping character chunks,
  2. embed chunks with a small ONNX embedding model (fastembed - no torch),
  3. index the vectors in FAISS (cosine similarity via normalised inner
     product),
  4. at question time, embed the question and retrieve the top-k chunks.

`RETRIEVAL_MODE=full` bypasses steps 2-4 and returns every chunk.
"""

from __future__ import annotations

import logging
import threading

from app.config import get_settings

logger = logging.getLogger(__name__)

_embedder = None
_embedder_lock = threading.Lock()


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Split text into overlapping windows, breaking on whitespace when possible."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            pivot = text.rfind(" ", start + overlap, end)
            if pivot != -1:
                end = pivot
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [c for c in chunks if c]


def _get_embedder():
    global _embedder
    if _embedder is None:
        with _embedder_lock:
            if _embedder is None:
                from fastembed import TextEmbedding

                settings = get_settings()
                logger.info("Loading embedding model %s", settings.embedding_model)
                _embedder = TextEmbedding(settings.embedding_model)
    return _embedder


def _embed(texts: list[str]):
    import numpy as np

    vectors = list(_get_embedder().embed(texts))
    arr = np.asarray(vectors, dtype="float32")
    # normalise so inner product == cosine similarity
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


class SessionIndex:
    """Holds the chunks (and, in embedding mode, the FAISS index) for one session."""

    def __init__(self) -> None:
        settings = get_settings()
        self.mode = settings.retrieval_mode
        self.top_k = settings.top_k
        self._chunks: list[dict] = []   # {"filename": str, "text": str}
        self._index = None              # faiss.IndexFlatIP or None

    def add_document(self, filename: str, text: str) -> int:
        settings = get_settings()
        pieces = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
        new_chunks = [{"filename": filename, "text": p} for p in pieces]
        if not new_chunks:
            return 0

        self._chunks.extend(new_chunks)

        if self.mode == "embedding":
            try:
                self._reindex()
            except Exception:  # noqa: BLE001 - degrade rather than fail the upload
                logger.exception(
                    "Embedding index build failed; falling back to full-context mode"
                )
                self.mode = "full"
                self._index = None
        return len(new_chunks)

    def _reindex(self) -> None:
        import faiss

        vectors = _embed([c["text"] for c in self._chunks])
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        self._index = index

    def search(self, question: str) -> list[dict]:
        """Return the most relevant chunks for a question, best first."""
        if not self._chunks:
            return []
        if self.mode != "embedding" or self._index is None:
            return list(self._chunks)

        query = _embed([question])
        k = min(self.top_k, len(self._chunks))
        scores, ids = self._index.search(query, k)
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            hit = dict(self._chunks[idx])
            hit["score"] = float(score)
            results.append(hit)
        return results

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)
