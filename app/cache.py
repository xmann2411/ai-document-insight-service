"""
A tiny in-process LRU cache for /ask answers.

Keyed by (session_id, normalised question, qa_backend). Answering a
question runs a cross-encoder over ~12 chunks (and, for the Claude
backend, a paid API call), so repeated questions - common in a demo or a
UI - are worth caching.

Deliberately in-memory: same rationale as `storage.py`. In production this
is the natural place to drop in Redis; the interface (`get` / `put` /
`invalidate`) wouldn't change.
"""

from __future__ import annotations

import threading
from collections import OrderedDict

from app.config import get_settings

_lock = threading.Lock()
_store: "OrderedDict[tuple, dict]" = OrderedDict()


def _key(session_id: str, question: str, backend: str) -> tuple:
    return (session_id, " ".join(question.lower().split()), backend)


def get(session_id: str, question: str, backend: str) -> dict | None:
    if not get_settings().cache_enabled:
        return None
    key = _key(session_id, question, backend)
    with _lock:
        if key not in _store:
            return None
        _store.move_to_end(key)
        return _store[key]


def put(session_id: str, question: str, backend: str, value: dict) -> None:
    settings = get_settings()
    if not settings.cache_enabled:
        return
    key = _key(session_id, question, backend)
    with _lock:
        _store[key] = value
        _store.move_to_end(key)
        while len(_store) > settings.cache_size:
            _store.popitem(last=False)


def invalidate(session_id: str) -> None:
    """Drop every cached answer for a session (call when its docs change)."""
    with _lock:
        for key in [k for k in _store if k[0] == session_id]:
            del _store[key]


def clear() -> None:
    with _lock:
        _store.clear()


def stats() -> dict:
    with _lock:
        return {"entries": len(_store), "capacity": get_settings().cache_size}
