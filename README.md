# AI Document Insight Service

A Python REST API (FastAPI) that ingests **PDF or image** documents (scanned
contracts, invoices, receipts…), extracts their text, and answers natural-language
questions about them using **Retrieval-Augmented Generation**.

The whole stack is **torch-free** – it installs and runs on any modern Python
(3.11 – 3.14), CPU only, and the default question-answering model is **local and
free** (no API key, no payment). Anthropic **Claude** is an optional drop-in
backend for higher-quality answers.

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
| `POST /sessions/{id}/documents` – add documents to a session | ✅ |
| `POST /ask` – QA pipeline over stored documents | ✅ |
| Dockerized | ✅ (`Dockerfile` + `docker-compose.yml`) |
| Dummy test documents in the repo | ✅ (`test_docs/`) |
| **Optional enhancement: RAG with embeddings + FAISS** | ✅ |
| Text extraction: PyMuPDF (digital PDFs) **+ Tesseract OCR** (images / scanned PDFs) | ✅ |
| Two QA backends: local DistilBERT-SQuAD (default) **or** Claude | ✅ |
| Answers cite the source excerpts they used | ✅ |
| Structured logging, health check, env-based config, CI | ✅ |

---

## Architecture

```
                 ┌──────────────┐
   PDF / image   │  extraction  │  PyMuPDF text layer → Tesseract OCR fallback
  ───────────────▶│              │  (image uploads always go through OCR)
                 └──────┬───────┘
                        │ plain text
                        ▼
                 ┌──────────────┐   chunk (900 chars, 150 overlap)
                 │   rag.py     │   → embed (fastembed / ONNX, bge-small)
                 │ SessionIndex │   → FAISS IndexFlatIP (cosine)
                 └──────┬───────┘
                        │ stored per session_id (in-memory)
                        ▼
   question       ┌──────────────┐   top-k chunks ─┐
  ───────────────▶│  qa_engine   │                 ├─▶ QA_BACKEND=local  → DistilBERT-SQuAD (onnxruntime)
                 │  (dispatch)  │                 └─▶ QA_BACKEND=claude → Claude API
                 └──────────────┘   → answer + sources
```

`RETRIEVAL_MODE=full` skips embeddings/FAISS and passes every chunk to the QA
backend.

---

## Quick start (Docker)

No API key needed – the default backend is the local model.

```bash
docker compose up --build
```

First build downloads the embedding + QA models (~150 MB) and bakes them into the
image, so the running container needs no network. The API is then on
<http://localhost:8000> – open <http://localhost:8000/docs> for Swagger UI.

```bash
curl -s localhost:8000/health | jq
```

To use Claude instead:

```bash
QA_BACKEND=claude ANTHROPIC_API_KEY=sk-ant-xxxx docker compose up --build
```

Plain `docker`:

```bash
docker build -t doc-insight .
docker run -p 8000:8000 doc-insight                       # local model
docker run -p 8000:8000 -e QA_BACKEND=claude -e ANTHROPIC_API_KEY=sk-ant-xxx doc-insight
```

---

## Manual installation

Works on Python 3.11 – 3.14.

```bash
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate

pip install -r requirements.txt
uvicorn app.main:app --reload
```

That's it – `POST /upload` and `POST /ask` work immediately with the local model.
The embedding and QA models download automatically on first use (cached in
`~/.cache/huggingface` and a fastembed cache dir).

**OCR** (image uploads / scanned PDFs) additionally needs the Tesseract binary:

| OS | Install |
|---|---|
| Debian/Ubuntu | `sudo apt-get install tesseract-ocr` |
| macOS | `brew install tesseract` |
| Windows | [UB-Mannheim installer](https://github.com/UB-Mannheim/tesseract/wiki), then add it to `PATH` |

Without Tesseract, digital-PDF Q&A still works; set `OCR_ENABLED=false` to silence
image-upload errors.

**Claude backend** (optional): put a key in `.env` and set the backend:

```bash
cp .env.example .env      # edit ANTHROPIC_API_KEY
echo "QA_BACKEND=claude" >> .env
```

---

## Configuration

All settings are environment variables (or lines in `.env`). See `.env.example`.
Nothing is required.

| Variable | Default | Notes |
|---|---|---|
| `QA_BACKEND` | `local` | `local` (DistilBERT-SQuAD, free) or `claude` (API). |
| `QA_LOCAL_MODEL` | `Xenova/distilbert-base-cased-distilled-squad` | Any ONNX SQuAD model on the HF Hub. |
| `ANTHROPIC_API_KEY` | – | Required only when `QA_BACKEND=claude`. |
| `LLM_MODEL` | `claude-haiku-4-5` | Claude model id. |
| `RETRIEVAL_MODE` | `embedding` | `embedding` or `full`. |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | fastembed-supported model. |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `900` / `150` | Characters. |
| `TOP_K` | `4` | Chunks retrieved per question. |
| `OCR_ENABLED` | `true` | Tesseract for images and scanned PDFs. |
| `OCR_LANGUAGES` | `eng` | Tesseract codes, e.g. `eng+hrv`. |
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
  "qa_backend": "local",
  "retrieval_mode": "embedding",
  "claude_key_configured": false,
  "ocr_enabled": true
}
```

### `POST /upload`  · `multipart/form-data`

Uploads documents and creates a **new** session. Field `files` – one or more
`.pdf`, `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, `.bmp`, `.webp`.

```bash
curl -s -X POST localhost:8000/upload \
  -F "files=@test_docs/sample_invoice.pdf" \
  -F "files=@test_docs/sample_contract.pdf"
```
```json
{
  "session_id": "9fabd9fe-540f-47b1-84fc-f39040e5f782",
  "files": [
    { "filename": "sample_invoice.pdf",  "chars": 717,  "chunks": 1, "status": "ok" },
    { "filename": "sample_contract.pdf", "chars": 1394, "chunks": 2, "status": "ok" }
  ]
}
```

Unsupported/unreadable files are reported per-file; if *nothing* processed → `422`.

### `POST /sessions/{session_id}/documents`  · `multipart/form-data`

Adds more documents to an existing session (same `files` field). `404` if the
session doesn't exist.

### `POST /ask`  · `multipart/form-data`

| Field | Type |
|---|---|
| `session_id` | string (required) |
| `question` | string (required) |

**Local backend:**
```bash
curl -s -X POST localhost:8000/ask \
  -F "session_id=9fabd9fe-540f-47b1-84fc-f39040e5f782" \
  -F "question=How long does confidentiality last after the agreement ends?"
```
```json
{
  "session_id": "9fabd9fe-540f-47b1-84fc-f39040e5f782",
  "question": "How long does confidentiality last after the agreement ends?",
  "backend": "local",
  "answer": "2 years",
  "confidence": 0.906,
  "sources": [
    {
      "filename": "sample_contract.pdf",
      "snippet": "SERVICE AGREEMENT ... Confidentiality: Each party shall keep ...",
      "used": true,
      "score": 0.71
    }
  ]
}
```

`answer` is the extracted span; `confidence` is the model's start/end probability
(geometric mean); `used: true` marks the chunk the span came from.

**Claude backend** returns a written sentence and no `confidence`/`used`:
```json
{
  "backend": "claude",
  "answer": "Confidentiality obligations survive for 2 years after the agreement ends.",
  "sources": [ { "filename": "sample_contract.pdf", "snippet": "...", "score": 0.71 } ]
}
```

If `QA_BACKEND=claude` and no key is set, `/ask` returns `503`.

### `GET /sessions/{session_id}`

```json
{
  "session_id": "9fabd9fe-...",
  "documents": [ { "filename": "sample_invoice.pdf", "chars": 717, "chunks": 1 } ],
  "total_chunks": 3,
  "retrieval_mode": "embedding"
}
```

---

## Test documents

`test_docs/` contains fictional dummy documents and a generator script – see
[`test_docs/README.md`](test_docs/README.md). Regenerate:

```bash
python test_docs/generate_test_docs.py
```

---

## Running the tests

```bash
pip install -r requirements-dev.txt
pytest
```

The default suite (19 tests) runs in `RETRIEVAL_MODE=full` with the Claude call
mocked – **no model download, no API key, no network**. Two extra tests actually
download and run the real models:

```bash
RUN_MODEL_TESTS=1 pytest tests/test_models.py
```

CI (`.github/workflows/ci.yml`) runs `pytest` and a `docker build` on every push.

---

## Approach & design choices

**Framework – FastAPI.** Async uploads, request validation, and free
OpenAPI/Swagger docs at `/docs` for a reviewer to try the API by hand.

**Torch-free by design.** The obvious libraries here (sentence-transformers,
EasyOCR, HF `transformers`) all pull in PyTorch – a large dependency with no
wheels for the newest Python releases and a fragile Docker story. Every ML piece
was chosen to avoid it:

- **Embeddings – `fastembed`** (ONNX Runtime under the hood) with
  `BAAI/bge-small-en-v1.5`. Small, fast on CPU, installs on Python 3.14.
- **OCR – Tesseract via `pytesseract`.** A small C++ binary, not a neural net
  framework. Handles the assignment's "scanned contracts" case.
- **Local QA – DistilBERT fine-tuned on SQuAD, run through ONNX Runtime**
  (`app/qa_local.py`, ~40 lines: tokenize, run, pick the best start/end span
  across the retrieved chunks). This is the assignment's suggested "DistilBERT
  QA" path.

Result: one `requirements.txt`, `pip install` works everywhere, the Docker image
is a plain `python:3.12-slim` + `apt-get tesseract-ocr`.

**Text extraction – PyMuPDF first, OCR fallback.** Most real contracts/invoices
are digital PDFs with a text layer; PyMuPDF reads it directly – fast and exact.
When a PDF page has (almost) no text layer we rasterise it and run Tesseract;
image uploads always go through Tesseract.

**Retrieval – RAG with embeddings + FAISS.** Sending a whole document to the QA
model doesn't scale – more context means more cost/latency and, for the local
model, blowing past its 512-token limit. Each document is split into overlapping
~900-char chunks, embedded, and indexed in a FAISS `IndexFlatIP` (inner product
on normalised vectors = cosine). At question time the top-4 chunks are retrieved;
the local backend runs QA on each and keeps the highest-confidence span, the
Claude backend gets them all as context. Answers carry their source snippets.

**Two QA backends.**

| | Local (`distilbert-squad`) | Claude (`claude-haiku-4-5`) |
|---|---|---|
| Cost / key | free, offline, none | API key, ~cents |
| Output | extractive span + confidence | written sentence, can synthesise |
| Quality | good on well-phrased factual questions; can pick a wrong span on ambiguous ones | consistently strong |

The local model is the default so the service is fully usable by anyone with zero
setup; Claude is one env var away when answer quality matters. The dispatch lives
in `app/qa_engine.py` and both backends return the same shape.

**Storage – in-memory, session-scoped.** A `session_id` (UUID) maps to its
documents and its FAISS index. Deliberately minimal; `storage.py`'s interface
(`create` / `add` / `get` / `search`) is what you'd keep when swapping in Redis
or a persistent vector DB.

**Ops.** 12-factor config, structured logging to stdout, `/health` reports
effective config, the Docker image pre-downloads models and has a `HEALTHCHECK`.

### How AI-generated code was validated

Parts were drafted with an LLM. Validation: (1) 19 automated tests cover
extraction against real generated PDFs, chunking edge cases, and every endpoint
including error paths; (2) two `RUN_MODEL_TESTS=1` tests assert the embedding index
ranks the relevant chunk first and the local QA model extracts the right span;
(3) the full pipeline was run end-to-end against the `test_docs/` files in both
backends; (4) the ONNX span-extraction maths (context-only tokens, offset
mapping, span scoring) was checked by hand against known answers; (5) the
Anthropic call follows current SDK docs (content-block iteration, typed errors).

---

## Limitations & next steps

- **Local QA is extractive** – it returns a span, so it can pick a plausible-looking
  wrong span on ambiguous questions and can't synthesise across excerpts. Use
  `QA_BACKEND=claude` for those. A stronger local option is
  `deepset/roberta-base-squad2` (bigger, supports "no answer") via `QA_LOCAL_MODEL`.
- **In-memory storage** – sessions are lost on restart and not shared across
  workers. Next: Redis for documents + persisted FAISS indices.
- **No auth / rate limiting** – add JWT + a limiter before exposing publicly.
- **FAISS index rebuilt on every upload** – fine for demo volumes.
- **Further brief enhancements**: Named Entity Recognition to highlight entities in
  answers, a Redis embedding cache, a Streamlit UI – the code is structured to
  drop them in.
```
