"""
AI-driven Document Insight Service
=================================

REST API to upload PDF / image documents and ask questions about them.

Endpoints
---------
- GET  /health                 - liveness + effective config
- POST /upload                 - upload one or more documents -> session_id
- POST /ask                    - ask a question about a session's documents
- GET  /sessions/{session_id}  - inspect what a session contains

Run locally:
    uvicorn app.main:app --reload
Then open http://localhost:8000/docs for Swagger UI.
"""

from __future__ import annotations

import logging

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from app import storage
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
    description="Upload PDF/image documents, then ask questions about them (RAG + Claude).",
)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "retrieval_mode": settings.retrieval_mode,
        "llm_model": settings.llm_model,
        "llm_configured": bool(settings.anthropic_api_key),
        "ocr_enabled": settings.ocr_enabled,
    }


@app.post("/upload")
async def upload_documents(
    files: list[UploadFile] = File(...),
    session_id: str | None = Form(default=None),
) -> dict:
    """
    Accept one or more PDF/image files, extract their text, chunk + index
    them, and store them under a session.

    Pass an existing `session_id` to add documents to that session;
    omit it to start a new one.
    """
    if session_id:
        if not storage.session_exists(session_id):
            raise HTTPException(status_code=404, detail="Session not found")
    else:
        session_id = storage.create_session()

    max_bytes = settings.max_upload_mb * 1024 * 1024
    results = []

    for file in files:
        if not is_supported(file.filename or ""):
            results.append(
                {"filename": file.filename, "status": "skipped (unsupported type)"}
            )
            continue

        file_bytes = await file.read()
        if len(file_bytes) > max_bytes:
            results.append(
                {
                    "filename": file.filename,
                    "status": f"error: exceeds {settings.max_upload_mb} MB limit",
                }
            )
            continue

        try:
            text = extract_text(file.filename, file_bytes)
        except ValueError as e:
            results.append({"filename": file.filename, "status": f"error: {e}"})
            continue
        except Exception:  # noqa: BLE001
            logger.exception("Extraction failed for %s", file.filename)
            results.append(
                {"filename": file.filename, "status": "error: extraction failed"}
            )
            continue

        record = storage.add_document(session_id, file.filename, text)
        results.append({**record, "status": "ok"})

    if not any(r["status"] == "ok" for r in results):
        raise HTTPException(
            status_code=422,
            detail={"message": "No documents could be processed", "files": results},
        )

    return {"session_id": session_id, "files": results}


@app.post("/ask")
async def ask_question(
    session_id: str = Form(...),
    question: str = Form(...),
) -> dict:
    """Ask a question about the documents uploaded in a given session."""
    if not storage.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    question = question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question must not be empty")

    chunks = storage.search(session_id, question)

    try:
        result = answer_question(question, chunks)
    except QAConfigError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return {"session_id": session_id, "question": question, **result}


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
