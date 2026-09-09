"""
In-memory, session-scoped storage for uploaded documents and their
retrieval index.

Intentionally minimal for the assignment: no database, no persistence
across restarts. A production version would back this with Redis / a
vector DB and persist the FAISS index - the interface here (create /
add / get / search) would stay the same.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field

from app.rag import SessionIndex


@dataclass
class Session:
    id: str
    documents: list[dict] = field(default_factory=list)  # {"filename", "chars", "chunks"}
    index: SessionIndex = field(default_factory=SessionIndex)


_SESSIONS: dict[str, Session] = {}
_lock = threading.Lock()


def create_session() -> str:
    session_id = str(uuid.uuid4())
    with _lock:
        _SESSIONS[session_id] = Session(id=session_id)
    return session_id


def session_exists(session_id: str) -> bool:
    return session_id in _SESSIONS


def add_document(session_id: str, filename: str, text: str) -> dict:
    session = _SESSIONS[session_id]
    n_chunks = session.index.add_document(filename, text)
    record = {"filename": filename, "chars": len(text), "chunks": n_chunks}
    session.documents.append(record)
    return record


def get_session(session_id: str) -> Session | None:
    return _SESSIONS.get(session_id)


def search(session_id: str, question: str) -> list[dict]:
    return _SESSIONS[session_id].index.search(question)


def reset() -> None:
    """Test helper - clear all sessions and the answer cache."""
    from app import cache

    with _lock:
        _SESSIONS.clear()
    cache.clear()
