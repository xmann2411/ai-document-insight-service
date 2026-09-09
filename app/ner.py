"""
Named Entity Recognition for answer text.

Two complementary passes, both torch-free:

1. A transformer token-classifier (`Xenova/bert-base-NER`, CoNLL-2003) run
   on ONNX Runtime - covers PERSON / ORG / LOCATION.
2. Regex patterns for the entity types that matter in contracts / invoices
   but aren't in that model's label set - MONEY, DATE, PERCENT, EMAIL, IBAN.

`extract_entities` returns character spans so a UI can highlight them in
place. Regex hits win over model hits on overlap (they're exact).
"""

from __future__ import annotations

import logging
import re
import threading

from app.config import get_settings

logger = logging.getLogger(__name__)

_MODEL_LABELS = {"PER": "PERSON", "ORG": "ORG", "LOC": "LOCATION"}

_REGEX_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("MONEY", re.compile(
        r"(?:EUR|USD|GBP|€|\$|£)\s?\d[\d.,]*|\d[\d.,]*\s?(?:EUR|USD|GBP|€|\$|£)",
        re.IGNORECASE)),
    ("PERCENT", re.compile(r"\d+(?:\.\d+)?\s?%")),
    ("DATE", re.compile(
        r"\b\d{4}-\d{2}-\d{2}\b"
        r"|\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\s+\d{4}\b"
        r"|\b\d+\s+(?:days?|months?|years?|weeks?|hours?)\b",
        re.IGNORECASE)),
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("IBAN", re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")),
]

_MAX_SEQ_LEN = 384
_state = None
_lock = threading.Lock()


class _Model:
    def __init__(self, session, tokenizer, id2label, has_token_type):
        self.session = session
        self.tokenizer = tokenizer
        self.id2label = id2label
        self.has_token_type = has_token_type


def _load() -> _Model:
    global _state
    if _state is None:
        with _lock:
            if _state is None:
                import json

                import onnxruntime as ort
                from huggingface_hub import hf_hub_download
                from tokenizers import Tokenizer

                repo = get_settings().ner_model
                logger.info("Loading NER model %s", repo)
                onnx_path = _download_onnx(repo, hf_hub_download)
                tokenizer = Tokenizer.from_file(hf_hub_download(repo, "tokenizer.json"))
                tokenizer.enable_truncation(_MAX_SEQ_LEN)
                with open(hf_hub_download(repo, "config.json")) as fh:
                    id2label = json.load(fh)["id2label"]
                session = ort.InferenceSession(
                    onnx_path, providers=["CPUExecutionProvider"]
                )
                names = {i.name for i in session.get_inputs()}
                _state = _Model(session, tokenizer, id2label, "token_type_ids" in names)
    return _state


def _download_onnx(repo: str, hf_hub_download) -> str:
    for name in ("onnx/model.onnx", "onnx/model_quantized.onnx", "model.onnx"):
        try:
            return hf_hub_download(repo, name)
        except Exception:  # noqa: BLE001
            continue
    raise RuntimeError(f"No ONNX weights found in {repo}")


def _regex_entities(text: str) -> list[dict]:
    out = []
    for label, pattern in _REGEX_PATTERNS:
        for m in pattern.finditer(text):
            out.append({"text": m.group().strip(), "label": label,
                        "start": m.start(), "end": m.end()})
    return out


def _model_entities(text: str) -> list[dict]:
    import numpy as np

    model = _load()
    enc = model.tokenizer.encode(text)
    feed = {
        "input_ids": np.array([enc.ids], dtype=np.int64),
        "attention_mask": np.array([enc.attention_mask], dtype=np.int64),
    }
    if model.has_token_type:
        feed["token_type_ids"] = np.array([enc.type_ids], dtype=np.int64)

    logits = model.session.run(None, feed)[0][0]
    preds = logits.argmax(-1)

    entities: list[dict] = []
    cur = None
    for pred, (start, end) in zip(preds, enc.offsets):
        if start == end:  # special token
            continue
        raw = model.id2label[str(int(pred))]
        tag, _, kind = raw.partition("-")
        label = _MODEL_LABELS.get(kind)

        if tag == "B" and label:
            if cur:
                entities.append(cur)
            cur = {"label": label, "start": int(start), "end": int(end)}
        elif tag == "I" and label and cur and cur["label"] == label:
            cur["end"] = int(end)
        else:
            if cur:
                entities.append(cur)
            cur = None
    if cur:
        entities.append(cur)

    for e in entities:
        e["text"] = text[e["start"]:e["end"]]
    return entities


def _dedupe(entities: list[dict]) -> list[dict]:
    """Drop entities that overlap an earlier (higher-priority) one."""
    kept: list[dict] = []
    for e in sorted(entities, key=lambda x: (x["start"], -(x["end"] - x["start"]))):
        if any(e["start"] < k["end"] and k["start"] < e["end"] for k in kept):
            continue
        kept.append(e)
    return kept


def extract_entities(text: str) -> list[dict]:
    if not text or not get_settings().ner_enabled:
        return []
    # regex first so it wins overlaps (exact matches for money/dates)
    entities = _regex_entities(text)
    try:
        entities.extend(_model_entities(text))
    except Exception:  # noqa: BLE001
        logger.exception("NER model failed; returning regex entities only")
    return _dedupe(entities)
