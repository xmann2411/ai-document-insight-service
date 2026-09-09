"""
QA engine - dispatches to the configured backend:

  QA_BACKEND=local      -> app/qa_sentences.py  (sentence ranking, default)
  QA_BACKEND=distilbert -> app/qa_local.py      (DistilBERT SQuAD span extraction)
  QA_BACKEND=claude     -> Anthropic Claude via API

All take the question plus the chunks selected by retrieval (app/rag.py)
and return {"answer": str, "sources": [...], ...}. The retrieval step
decides *which* text the model sees; this module just formats and calls.
"""

from __future__ import annotations

import logging

from app.config import get_settings

logger = logging.getLogger(__name__)


class QAConfigError(RuntimeError):
    """Raised when the selected backend is not usable (e.g. missing API key)."""


SYSTEM_PROMPT = (
    "You are a document assistant. Answer the user's question using ONLY the "
    "information in the provided document excerpts. If the answer is not "
    "present, say clearly that the documents don't contain that information - "
    "do not guess or use outside knowledge. Quote short snippets when helpful."
)

_claude_client = None


def answer_question(question: str, chunks: list[dict]) -> dict:
    backend = get_settings().qa_backend
    if backend == "claude":
        return _answer_with_claude(question, chunks)
    if backend == "local":
        from app import qa_sentences

        return {"backend": "local", **qa_sentences.answer_question(question, chunks)}
    if backend == "distilbert":
        from app import qa_local

        return {"backend": "distilbert", **qa_local.answer_question(question, chunks)}
    raise QAConfigError(
        f"Unknown QA_BACKEND '{backend}' (use 'local', 'distilbert' or 'claude')"
    )


# --- Claude backend -------------------------------------------------------
def _get_claude_client():
    global _claude_client
    if _claude_client is None:
        settings = get_settings()
        if not settings.anthropic_api_key:
            raise QAConfigError(
                "QA_BACKEND=claude but ANTHROPIC_API_KEY is not set. Add the key "
                "or switch to QA_BACKEND=local (free, offline)."
            )
        from anthropic import Anthropic

        _claude_client = Anthropic(api_key=settings.anthropic_api_key)
    return _claude_client


def _format_context(chunks: list[dict]) -> str:
    return "\n\n---\n\n".join(
        f"[Excerpt {i} - source: {c['filename']}]\n{c['text']}"
        for i, c in enumerate(chunks, start=1)
    )


def _answer_with_claude(question: str, chunks: list[dict]) -> dict:
    if not chunks:
        return {
            "backend": "claude",
            "answer": "No documents have been uploaded for this session yet.",
            "sources": [],
        }

    settings = get_settings()
    client = _get_claude_client()

    response = client.messages.create(
        model=settings.llm_model,
        max_tokens=settings.llm_max_tokens,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Document excerpts:\n\n{_format_context(chunks)}\n\n"
                    f"Question: {question}"
                ),
            }
        ],
    )
    answer = "".join(
        block.text for block in response.content if block.type == "text"
    ).strip()

    sources = []
    for c in chunks:
        entry = {
            "filename": c["filename"],
            "snippet": c["text"][:200] + ("..." if len(c["text"]) > 200 else ""),
        }
        if "score" in c:
            entry["score"] = round(c["score"], 3)
        if "rerank_score" in c:
            entry["rerank_score"] = c["rerank_score"]
        sources.append(entry)

    return {"backend": "claude", "answer": answer, "sources": sources}
