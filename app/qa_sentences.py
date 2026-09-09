"""
Local question answering by sentence ranking (default backend).

Instead of asking a small span-extraction model to pick start/end offsets
(fragile on a tiny model), we:

  1. take the chunks retrieval already selected,
  2. split them into sentences,
  3. score every sentence against the question with the same cross-encoder
     used for chunk reranking (a real transformer that reads the question
     and the sentence together),
  4. return the best one or two sentences as the answer.

This is free, offline, no API key, and in practice a lot more reliable
than span extraction for factual questions - the trade-off is that the
answer is a verbatim sentence from the document, not a rephrased one.
"""

from __future__ import annotations

import logging
import re

from app.config import get_settings
from app.rag import get_reranker

logger = logging.getLogger(__name__)

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
_MIN_SENT_CHARS = 12
_LOW_CONFIDENCE = -3.0


def _is_heading(line: str) -> bool:
    """A short label line with no sentence punctuation, e.g. '7. Governing law'."""
    core = re.sub(r"^\d+[.)]\s*", "", line).strip()
    return (
        0 < len(core.split()) <= 5
        and line[-1] not in ".!?:"
        and core[:1].isupper()
    )


def _sentences(text: str) -> list[str]:
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    merged: list[str] = []
    for line in lines:
        if merged and _is_heading(merged[-1]):
            merged[-1] = f"{merged[-1]} {line}"
        else:
            merged.append(line)

    out: list[str] = []
    for block in merged:
        for piece in _SENT_SPLIT.split(block):
            piece = " ".join(piece.split()).strip()
            if len(piece) >= _MIN_SENT_CHARS:
                out.append(piece)
    return out


def answer_question(question: str, chunks: list[dict]) -> dict:
    if not chunks:
        return {
            "answer": "No documents have been uploaded for this session yet.",
            "confidence": 0.0,
            "sources": [],
        }

    settings = get_settings()

    # candidate sentences, remembering which chunk each came from
    candidates: list[tuple[str, dict]] = []
    for chunk in chunks:
        for sent in _sentences(chunk["text"]):
            candidates.append((sent, chunk))

    if not candidates:
        return {"answer": "", "confidence": 0.0, "sources": _sources(chunks, None)}

    try:
        scores = list(get_reranker().rerank(question, [s for s, _ in candidates]))
    except Exception:  # noqa: BLE001 - degrade to the first retrieved sentences
        logger.exception("Reranker unavailable; returning the top chunk's opening")
        top_chunk = chunks[0]
        opening = " ".join(_sentences(top_chunk["text"])[:2])
        return {
            "answer": opening[: settings.qa_max_answer_chars],
            "confidence": 0.0,
            "sources": _sources(chunks, top_chunk),
        }

    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    top_score, (top_sent, top_chunk) = ranked[0]

    # include the next sentence too if it's from the same chunk and also relevant
    answer_parts = [top_sent]
    if len(ranked) > 1:
        second_score, (second_sent, second_chunk) = ranked[1]
        if second_chunk is top_chunk and second_score > top_score - 2.0:
            answer_parts.append(second_sent)

    if top_score < _LOW_CONFIDENCE:
        answer = (
            "The documents don't appear to contain a clear answer to that "
            f'question. Closest match: "{top_sent[:200]}"'
        )
    else:
        answer = " ".join(answer_parts)[: settings.qa_max_answer_chars]

    return {
        "answer": answer,
        "confidence": round(_sigmoid(top_score), 3),
        "sources": _sources(chunks, top_chunk),
    }


def _sigmoid(x: float) -> float:
    import math

    x = max(-30.0, min(30.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def _sources(chunks: list[dict], picked: dict | None) -> list[dict]:
    out = []
    for c in chunks:
        entry = {
            "filename": c["filename"],
            "snippet": c["text"][:200] + ("..." if len(c["text"]) > 200 else ""),
            "used": c is picked,
        }
        if "score" in c:
            entry["score"] = c["score"]
        if "rerank_score" in c:
            entry["rerank_score"] = c["rerank_score"]
        out.append(entry)
    return out
