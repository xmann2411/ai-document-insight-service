# AI Document Insight Service

A Python REST API (FastAPI) that ingests **PDF or image** documents (scanned
contracts, invoices, receipts…), extracts their text, and answers natural-language
questions about them using **Retrieval-Augmented Generation** (sentence-transformers
embeddings + FAISS) with **Claude** as the answering LLM.

---

## Contents

- [Features](#features)
- [Architecture](#architecture)
- [Quick start (Docker)](#quick-start-docker)
- [Manual installation](#manual-installation)
- [Configuration](#configuration)
- [API reference & examples](#api-reference--examples)
- [Test documents](#test-documents)
- [Running the tests](#running-the-tests)
- [Approach & design choices](#approach--design-choices)
- [Limitations & next steps](#limitations--next-steps)

---

## Features

| Requirement | Status |
|---|---|
| `POST /upload` – one or more documents, session-based retrieval | ✅ |
| `POST /ask` – QA pipeline over stored documents | ✅ |
| Dockerized | ✅ (`Dockerfile` + `docker-compose.yml`) |
| Dummy test documents in the repo | ✅ (`test_docs/`) |
| **Optional enhancement: RAG with embeddings + FAISS** | ✅ |
| Text extraction: PyMuPDF (digital PDFs) **+ EasyOCR** (images / scanned PDFs) | ✅ |
| Answers cite the source excerpts they used | ✅ |
| Structured logging, health check, config via env | ✅ |

---

## Architecture

```
                 ┌──────────────┐
   PDF / image   │  extraction  │  PyMuPDF text layer → EasyOCR fallback
  ───────────────▶│  (app/       │  (images always go through OCR)
                 │ extraction)  │
                 └──────┬───────┘
                        │ plain text
                        ▼
                 ┌──────────────┐   chunk (900 chars, 150 overlap)
                 │   rag.py     │   → embed (all-MiniLM-L6-v2)
                 │ SessionIndex │   → FAISS IndexFlatIP (cosine)
                 └──────┬───────┘
                        │ stored per session_id (app/storage.py, in-memory)
                        ▼
   question       ┌──────────────┐   top-k chunks → prompt
  ───────────────▶│  qa_engine   │   → Claude (claude-haiku-4-5)
                 │              │   → answer + sources
                 └──────────────┘
```

`RETRIEVAL_MODE=full` skips embeddings/FAISS and sends every stored chunk to the
LLM – a fallback for environments where the ML stack can't be installed.

---

## Quick start (Docker)

Requires Docker and an Anthropic API key.

```bash
# 1. build (first build downloads torch + the embedding & OCR models, ~5 min)
docker compose build

# 2. run
ANTHROPIC_API_KEY=sk-ant-xxxx docker compose up
```

The API is now on <http://localhost:8000> – open <http://localhost:8000/docs> for
interactive Swagger UI.

```bash
# smoke test
curl -s localhost:8000/health | jq
```

Plain `docker` without compose:

```bash
docker build -t doc-insight .
docker run -p 8000:8000 -e ANTHROPIC_API_KEY=sk-ant-xxxx doc-insight
```

---

## Manual installation

Python 3.11 or 3.12 recommended.

```bash
python -m venv venv
source venv/bin/activate           # Windows: venv\Scripts\activate

# full install (RAG + OCR)
pip install -r requirements-rag.txt

cp .env.example .env                # then put your ANTHROPIC_API_KEY in .env

uvicorn app.main:app --reload
```

### Lightweight install (no torch / no embedding stack)

The RAG + OCR stack pulls in **torch**, which has no wheels for the newest
Python releases yet (e.g. 3.14). To run the core service anywhere:

```bash
pip install -r requirements.txt     # fastapi + pymupdf + anthropic only
RETRIEVAL_MODE=full OCR_ENABLED=false uvicorn app.main:app --reload
```

You still get digital-PDF Q&A; you lose image OCR and embedding-based retrieval.

---

## Configuration

All settings are environment variables (or lines in `.env`). See `.env.example`.

| Variable | Default | Notes |
|---|---|---|
| `ANTHROPIC_API_KEY` | – | **Required for `/ask`.** `/upload` works without it. |
| `LLM_MODEL` | `claude-haiku-4-5` | Any Claude model id. |
| `LLM_MAX_TOKENS` | `600` | Max answer length. |
| `RETRIEVAL_MODE` | `embedding` | `embedding` or `full`. |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `900` / `150` | Characters. |
| `TOP_K` | `4` | Chunks retrieved per question. |
| `OCR_ENABLED` | `true` | EasyOCR for images and scanned PDFs. |
| `OCR_LANGUAGES` | `en` | Comma-separated, e.g. `en,hr`. |
| `MAX_UPLOAD_MB` | `25` | Per-file limit. |

---

## API reference & examples

Base URL: `http://localhost:8000`

### `GET /health`

```bash
curl -s localhost:8000/health
```
```json
{
  "status": "ok",
  "retrieval_mode": "embedding",
  "llm_model": "claude-haiku-4-5",
  "llm_configured": true,
  "ocr_enabled": true
}
```

### `POST /upload`

`multipart/form-data`

| Field | Type | Notes |
|---|---|---|
| `files` | file(s) | One or more. `.pdf`, `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, `.bmp`, `.webp`. |
| `session_id` | string | Optional. Omit to start a new session; pass an existing one to add documents to it. |

```bash
curl -s -X POST localhost:8000/upload \
  -F "files=@test_docs/sample_invoice.pdf" \
  -F "files=@test_docs/sample_contract.pdf"
```
```json
{
  "session_id": "40e048f5-8d05-49f5-9aeb-c8d0628d9aa2",
  "files": [
    { "filename": "sample_invoice.pdf",  "chars": 717,  "chunks": 1, "status": "ok" },
    { "filename": "sample_contract.pdf", "chars": 1394, "chunks": 2, "status": "ok" }
  ]
}
```

Unsupported or unreadable files are reported per-file with a `status` string; if
*nothing* could be processed the endpoint returns `422`.

### `POST /ask`

`multipart/form-data`

| Field | Type |
|---|---|
| `session_id` | string (required) |
| `question` | string (required) |

```bash
curl -s -X POST localhost:8000/ask \
  -F "session_id=40e048f5-8d05-49f5-9aeb-c8d0628d9aa2" \
  -F "question=What is the total due and the due date?"
```
```json
{
  "session_id": "40e048f5-8d05-49f5-9aeb-c8d0628d9aa2",
  "question": "What is the total due and the due date?",
  "answer": "The total due is EUR 4,475.63 and the due date is 2026-09-14.",
  "sources": [
    {
      "filename": "sample_invoice.pdf",
      "snippet": "INVOICE  #INV-2026-0341\nDate of issue: 2026-08-15\nDue date: 2026-09-14 ...",
      "score": 0.62
    }
  ]
}
```

`score` (cosine similarity, `embedding` mode only) lets the caller judge how
well-grounded the answer is. If no key is configured, `/ask` returns `503`.

### `GET /sessions/{session_id}`

```bash
curl -s localhost:8000/sessions/40e048f5-8d05-49f5-9aeb-c8d0628d9aa2
```
```json
{
  "session_id": "40e048f5-8d05-49f5-9aeb-c8d0628d9aa2",
  "documents": [
    { "filename": "sample_invoice.pdf", "chars": 717, "chunks": 1 }
  ],
  "total_chunks": 1,
  "retrieval_mode": "embedding"
}
```

---

## Test documents

`test_docs/` contains fictional dummy documents and a generator script. See
[`test_docs/README.md`](test_docs/README.md). Regenerate with:

```bash
python test_docs/generate_test_docs.py
```

---

## Running the tests

```bash
pip install -r requirements-dev.txt
pytest
```

The suite runs in `RETRIEVAL_MODE=full` and mocks the Claude call, so it needs
**no API key and no network** (17 tests: extraction, chunking, all endpoints,
error paths).

---

## Approach & design choices

**Framework – FastAPI.** Async file uploads, automatic request validation, and
free OpenAPI/Swagger docs at `/docs` – useful for a reviewer to try the API by
hand.

**Text extraction – PyMuPDF first, EasyOCR fallback.** Most real contracts and
invoices are digitally generated PDFs with an embedded text layer; PyMuPDF reads
that directly – fast, exact, no ML. When a PDF page has (almost) no text layer we
rasterise it and run **EasyOCR**; image uploads always go through EasyOCR. This
covers the assignment's "scanned contracts, invoices" case without paying the OCR
cost when it isn't needed. EasyOCR is imported lazily so the service still starts
without the OCR stack installed.

**Retrieval – RAG with sentence-transformers + FAISS.** Sending a whole document
to the LLM works for one short file but scales badly – more input tokens (cost and
latency) and more noise for the model. Instead each document is split into
overlapping ~900-character chunks, embedded with `all-MiniLM-L6-v2` (small, fast,
runs on CPU), and indexed in a FAISS `IndexFlatIP` (inner product on normalised
vectors = cosine similarity). At question time the question is embedded and the
top-4 chunks become the context. The answer carries its source snippets so the
caller can verify it.

**LLM – Claude (`claude-haiku-4-5`).** The assignment allows "any LLM via API".
Haiku is cheap and fast, which suits short extractive Q&A over a handful of
retrieved chunks; the model is a one-line config change. The system prompt
constrains Claude to answer **only** from the provided excerpts and to say so when
the answer isn't there, which keeps answers grounded.

**Storage – in-memory, session-scoped.** No database: a `session_id` (UUID) maps
to its documents and its FAISS index. This is deliberately minimal for the
assignment; the `storage.py` interface (`create` / `add` / `get` / `search`)
is what you'd keep when swapping in Redis or a persistent vector DB.

**Config & ops.** All tunables are environment variables (12-factor); structured
logging to stdout; `/health` reports effective config; the Docker image
pre-downloads all models so the container needs no network at runtime and has a
`HEALTHCHECK`.

### How AI-generated code was validated

Parts of this were drafted with an LLM. Validation: (1) the automated test suite
covers extraction against real generated PDFs, chunking edge cases, and every
endpoint including error paths; (2) extraction and retrieval were checked
end-to-end against the `test_docs/` files; (3) the Anthropic SDK call was written
against current SDK docs (content-block iteration rather than `content[0].text`,
typed error handling); (4) dependency versions are pinned to ranges and installed
in CI-like fashion.

---

## Limitations & next steps

- **In-memory storage** – sessions are lost on restart and not shared across
  workers. Next: Redis for documents + persisted FAISS indices.
- **No auth / rate limiting** – add JWT + a limiter (e.g. SlowAPI) before exposing
  it.
- **FAISS index is rebuilt on every upload** – fine for demo volumes; use
  `add`-only or an IVF index at scale.
- **Further enhancements from the brief**: Named Entity Recognition to highlight
  entities in answers, a Redis embedding cache, and a Streamlit UI are natural
  next additions (the code is structured to drop them in).
```
