"""
Local extractive question answering with DistilBERT fine-tuned on SQuAD,
run through onnxruntime (no torch).

This is the assignment's suggested "DistilBERT QA" path: free, offline, no
API key. It is *extractive* - it returns the span of the context that best
answers the question, plus a confidence score. For each retrieved chunk we
run the model once and keep the highest-scoring span across all chunks.
"""

from __future__ import annotations

import logging
import math
import threading

from app.config import get_settings

logger = logging.getLogger(__name__)

_MAX_SEQ_LEN = 384          # question + context tokens fed to the model
_MAX_ANSWER_TOKENS = 30     # cap on span length (in tokens)

_state = None
_lock = threading.Lock()


class _Model:
    def __init__(self, session, tokenizer):
        self.session = session
        self.tokenizer = tokenizer


def _load() -> _Model:
    global _state
    if _state is None:
        with _lock:
            if _state is None:
                import onnxruntime as ort
                from huggingface_hub import hf_hub_download
                from tokenizers import Tokenizer

                repo = get_settings().qa_local_model
                logger.info("Loading local QA model %s", repo)
                onnx_path = _download_onnx(repo, hf_hub_download)
                tok_path = hf_hub_download(repo, "tokenizer.json")
                tokenizer = Tokenizer.from_file(tok_path)
                tokenizer.enable_truncation(_MAX_SEQ_LEN)
                session = ort.InferenceSession(
                    onnx_path, providers=["CPUExecutionProvider"]
                )
                _state = _Model(session, tokenizer)
    return _state


def _download_onnx(repo: str, hf_hub_download) -> str:
    # Prefer the quantized model (smaller / faster), fall back to full.
    for name in ("onnx/model_quantized.onnx", "onnx/model.onnx", "model.onnx"):
        try:
            return hf_hub_download(repo, name)
        except Exception:  # noqa: BLE001
            continue
    raise RuntimeError(f"No ONNX weights found in {repo}")


def _softmax_max(logits) -> float:
    m = max(logits)
    denom = sum(math.exp(x - m) for x in logits)
    return math.exp(max(logits) - m) / denom


def _answer_one(model: _Model, question: str, context: str) -> tuple[str, float]:
    import numpy as np

    enc = model.tokenizer.encode(question, context)
    ids = np.array([enc.ids], dtype=np.int64)
    mask = np.array([enc.attention_mask], dtype=np.int64)

    start_logits, end_logits = model.session.run(
        None, {"input_ids": ids, "attention_mask": mask}
    )
    start_logits, end_logits = start_logits[0], end_logits[0]

    # Only consider tokens that belong to the context (sequence id 1).
    seq_ids = enc.sequence_ids
    context_positions = [i for i, s in enumerate(seq_ids) if s == 1]
    if not context_positions:
        return "", 0.0

    best_score = float("-inf")
    best_span = (context_positions[0], context_positions[0])
    for i in context_positions:
        for j in range(i, min(i + _MAX_ANSWER_TOKENS, context_positions[-1] + 1)):
            score = start_logits[i] + end_logits[j]
            if score > best_score:
                best_score = score
                best_span = (i, j)

    i, j = best_span
    char_start = enc.offsets[i][0]
    char_end = enc.offsets[j][1]
    answer = " ".join(context[char_start:char_end].split())  # collapse whitespace

    # confidence: geometric mean of the start/end probabilities
    conf = math.sqrt(_softmax_max(start_logits) * _softmax_max(end_logits))
    return answer, float(conf)


def answer_question(question: str, chunks: list[dict]) -> dict:
    """
    Returns {"answer", "confidence", "sources"}.
    Runs the model on each retrieved chunk and keeps the best span.
    """
    if not chunks:
        return {
            "answer": "No documents have been uploaded for this session yet.",
            "confidence": 0.0,
            "sources": [],
        }

    model = _load()
    settings = get_settings()

    best = {"answer": "", "confidence": -1.0, "chunk": chunks[0]}
    for chunk in chunks:
        answer, conf = _answer_one(model, question, chunk["text"])
        if answer and conf > best["confidence"]:
            best = {"answer": answer, "confidence": conf, "chunk": chunk}

    answer = best["answer"][: settings.qa_max_answer_chars]
    if not answer:
        answer = "The documents don't appear to contain an answer to that question."

    return {
        "answer": answer,
        "confidence": round(max(best["confidence"], 0.0), 3),
        "sources": _sources(chunks, best["chunk"]),
    }


def _sources(chunks: list[dict], picked: dict) -> list[dict]:
    out = []
    for c in chunks:
        entry = {
            "filename": c["filename"],
            "snippet": c["text"][:200] + ("..." if len(c["text"]) > 200 else ""),
            "used": c is picked,
        }
        if "score" in c:
            entry["score"] = round(c["score"], 3)
        out.append(entry)
    return out
