"""
AI-driven Document Insight Service
=================================

REST API to upload PDF / image documents and ask questions about them.

Endpoints
---------
- GET  /health                          - liveness + effective config
- POST /upload                          - upload documents -> new session_id
- POST /sessions/{session_id}/documents - add more documents to a session
- POST /ask                             - ask a question about a session
- GET  /sessions/{session_id}           - inspect what a session contains

Run locally:
    uvicorn app.main:app --reload
Then open http://localhost:8000/docs for Swagger UI.
"""

from __future__ import annotations

import logging

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from app import cache, storage
from app.config import get_settings
from app.extraction import extract_text, is_supported
from app.qa_engine import QAConfigError, answer_question

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
logger = logging.getLogger("app")

settings = get_settings()

app = FastAPI(
    title="AI Document Insight Service",
    version="1.0.0",
    description=(
        "Upload PDF/image documents, then ask questions about them "
        "(RAG retrieval + a local or Claude QA model).\n\n"
        "**Flow:** call `POST /upload` first, copy the `session_id` from the "
        "response, then paste it into `POST /ask`."
    ),
)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "qa_backend": settings.qa_backend,
        "retrieval_mode": settings.retrieval_mode,
        "rerank_enabled": settings.rerank_enabled,
        "ner_enabled": settings.ner_enabled,
        "ocr_enabled": settings.ocr_enabled,
        "claude_key_configured": bool(settings.anthropic_api_key),
        "cache": cache.stats(),
    }


async def _ingest(session_id: str, files: list[UploadFile]) -> list[dict]:
    max_bytes = settings.max_upload_mb * 1024 * 1024
    results: list[dict] = []

    for file in files:
        name = file.filename or "unnamed"
        if not is_supported(name):
            results.append({"filename": name, "status": "skipped (unsupported type)"})
            continue

        file_bytes = await file.read()
        if len(file_bytes) > max_bytes:
            results.append(
                {"filename": name, "status": f"error: exceeds {settings.max_upload_mb} MB"}
            )
            continue

        try:
            text = extract_text(name, file_bytes)
        except ValueError as e:
            results.append({"filename": name, "status": f"error: {e}"})
            continue
        except Exception:  # noqa: BLE001
            logger.exception("Extraction failed for %s", name)
            results.append({"filename": name, "status": "error: extraction failed"})
            continue

        record = storage.add_document(session_id, name, text)
        results.append({**record, "status": "ok"})

    if any(r["status"] == "ok" for r in results):
        cache.invalidate(session_id)  # answers may change now
    return results


@app.post("/upload")
async def upload_documents(files: list[UploadFile] = File(...)) -> dict:
    """
    Upload one or more PDF/image files. Extracts their text, chunks + indexes
    it, and stores everything under a **new** session.

    Returns a `session_id` - copy it and use it with `POST /ask`.
    """
    session_id = storage.create_session()
    results = await _ingest(session_id, files)

    if not any(r["status"] == "ok" for r in results):
        raise HTTPException(
            status_code=422,
            detail={"message": "No documents could be processed", "files": results},
        )
    return {"session_id": session_id, "files": results}


@app.post("/sessions/{session_id}/documents")
async def add_documents(
    session_id: str, files: list[UploadFile] = File(...)
) -> dict:
    """Add more documents to an existing session."""
    if not storage.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    results = await _ingest(session_id, files)
    if not any(r["status"] == "ok" for r in results):
        raise HTTPException(
            status_code=422,
            detail={"message": "No documents could be processed", "files": results},
        )
    return {"session_id": session_id, "files": results}


@app.post("/ask")
async def ask_question(
    session_id: str = Form(..., description="The session_id returned by /upload"),
    question: str = Form(..., description="Your question about the documents"),
) -> dict:
    """Ask a question about the documents uploaded in a given session."""
    if not storage.session_exists(session_id):
        raise HTTPException(
            status_code=404,
            detail="Session not found. Call POST /upload first and use the "
            "session_id it returns.",
        )

    question = question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question must not be empty")

    backend = settings.qa_backend
    cached = cache.get(session_id, question, backend)
    if cached is not None:
        return {"session_id": session_id, "question": question, "cached": True, **cached}

    chunks = storage.search(session_id, question)
    try:
        result = answer_question(question, chunks)
    except QAConfigError as e:
        raise HTTPException(status_code=503, detail=str(e))

    cache.put(session_id, question, backend, result)
    return {"session_id": session_id, "question": question, "cached": False, **result}


@app.get("/sessions/{session_id}")
def session_info(session_id: str) -> dict:
    session = storage.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "session_id": session.id,
        "documents": session.documents,
        "total_chunks": session.index.chunk_count,
        "retrieval_mode": session.index.mode,
    }
