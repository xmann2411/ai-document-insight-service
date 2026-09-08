"""
QA engine: answer a question grounded in retrieved document chunks,
using Claude as the LLM.

The retrieval step (app/rag.py) selects which chunks to send; this module
just formats them into a prompt, calls Claude, and returns the answer
together with the sources it was given (useful for the caller to verify
the answer isn't hallucinated).
"""

from __future__ import annotations

import logging

from app.config import get_settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a document assistant. Answer the user's question using ONLY the "
    "information in the provided document excerpts. If the answer is not "
    "present, say clearly that the documents don't contain that information - "
    "do not guess or use outside knowledge. Quote short snippets when helpful."
)

_client = None


class QAConfigError(RuntimeError):
    """Raised when the LLM backend is not configured (e.g. missing API key)."""


def _get_client():
    global _client
    if _client is None:
        settings = get_settings()
        if not settings.anthropic_api_key:
            raise QAConfigError(
                "ANTHROPIC_API_KEY is not set. Add it to your environment or "
                ".env file to enable question answering."
            )
        from anthropic import Anthropic

        _client = Anthropic(api_key=settings.anthropic_api_key)
    return _client


def _format_context(chunks: list[dict]) -> str:
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        blocks.append(
            f"[Excerpt {i} - source: {chunk['filename']}]\n{chunk['text']}"
        )
    return "\n\n---\n\n".join(blocks)


def answer_question(question: str, chunks: list[dict]) -> dict:
    """
    Returns {"answer": str, "sources": [{"filename", "snippet", "score"?}]}.
    """
    if not chunks:
        return {
            "answer": "No documents have been uploaded for this session yet.",
            "sources": [],
        }

    settings = get_settings()
    client = _get_client()

    user_message = (
        f"Document excerpts:\n\n{_format_context(chunks)}\n\n"
        f"Question: {question}"
    )

    response = client.messages.create(
        model=settings.llm_model,
        max_tokens=settings.llm_max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    answer = "".join(
        block.text for block in response.content if block.type == "text"
    ).strip()

    sources = [
        {
            "filename": c["filename"],
            "snippet": c["text"][:200] + ("..." if len(c["text"]) > 200 else ""),
            **({"score": round(c["score"], 3)} if "score" in c else {}),
        }
        for c in chunks
    ]
    return {"answer": answer, "sources": sources}
