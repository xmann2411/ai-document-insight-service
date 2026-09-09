"""
Text extraction from uploaded documents.

Strategy (matches the assignment's "PDF or image documents, e.g. scanned
contracts, invoices"):

1. PDF with a text layer  -> PyMuPDF. Fast, exact, no ML. Most real
   contracts / invoices are digitally generated and land here.
2. PDF that is a scanned image (no text layer) -> rasterise each page
   with PyMuPDF, then OCR with Tesseract (pytesseract).
3. Image upload (.png/.jpg/...) -> OCR directly with Tesseract.

Tesseract is a small C++ binary (no torch / no GPU). It must be on PATH;
the Docker image installs it. Set OCR_ENABLED=false to disable OCR.
"""

from __future__ import annotations

import io
import logging
import re

import pymupdf

from app.config import get_settings

logger = logging.getLogger(__name__)

_WS_RUN = re.compile(r"[ \t]{2,}")
_MULTI_NL = re.compile(r"\n{3,}")
# a line break that continues the same sentence: previous char isn't sentence-
# ending punctuation and the next line starts lowercase / a conjunction.
_SOFT_WRAP = re.compile(r"(?<=[^.!?:;)\]\"'\n])[ \t]*\n[ \t]*(?=[a-z(])")


def normalize_text(text: str) -> str:
    """
    Clean up extracted text for downstream chunking / QA.

    PDF and OCR output is full of layout artefacts. We:
      - strip leading indentation from every line,
      - join *soft* line wraps (a line break mid-sentence) so a clause isn't
        split, while keeping *hard* line breaks (each invoice field on its
        own line) as sentence boundaries,
      - collapse runs of spaces and blank lines.
    """
    text = text.replace("\r\n", "\n").replace("\xa0", " ")
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _SOFT_WRAP.sub(" ", text)
    text = _MULTI_NL.sub("\n\n", text)
    text = _WS_RUN.sub(" ", text)
    return text.strip()

PDF_EXTENSIONS = {".pdf"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
SUPPORTED_EXTENSIONS = PDF_EXTENSIONS | IMAGE_EXTENSIONS

_MIN_TEXT_LAYER_CHARS = 20  # below this we treat a PDF page as "no text layer"


def is_supported(filename: str) -> bool:
    return _ext(filename) in SUPPORTED_EXTENSIONS


def _ext(filename: str) -> str:
    _, _, tail = filename.rpartition(".")
    return f".{tail.lower()}" if tail else ""


def _ocr(image_bytes: bytes) -> str:
    try:
        import pytesseract
        from PIL import Image
    except ImportError as e:  # pragma: no cover
        raise ValueError(f"OCR dependencies not installed: {e}") from e

    try:
        image = Image.open(io.BytesIO(image_bytes))
        return pytesseract.image_to_string(
            image, lang=get_settings().ocr_lang_string
        ).strip()
    except pytesseract.TesseractNotFoundError as e:
        raise ValueError(
            "The 'tesseract' binary was not found on PATH. Install Tesseract "
            "OCR (or use the Docker image), or set OCR_ENABLED=false."
        ) from e


def extract_text(filename: str, file_bytes: bytes) -> str:
    """
    Extract plain text from an uploaded document.

    Raises ValueError for unsupported types or when nothing could be
    extracted (e.g. a scanned PDF while OCR is disabled / unavailable).
    """
    ext = _ext(filename)
    if ext in PDF_EXTENSIONS:
        return _extract_from_pdf(file_bytes)
    if ext in IMAGE_EXTENSIONS:
        return _extract_from_image(file_bytes)
    raise ValueError(
        f"Unsupported file type '{ext or filename}'. "
        f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
    )


def _extract_from_pdf(file_bytes: bytes) -> str:
    settings = get_settings()
    text_parts: list[str] = []
    pages_needing_ocr: list[int] = []

    with pymupdf.open(stream=file_bytes, filetype="pdf") as doc:
        for page_index, page in enumerate(doc):
            page_text = page.get_text().strip()
            if len(page_text) >= _MIN_TEXT_LAYER_CHARS:
                text_parts.append(page_text)
            else:
                pages_needing_ocr.append(page_index)

        if pages_needing_ocr and settings.ocr_enabled:
            logger.info("OCR fallback for %d page(s)", len(pages_needing_ocr))
            for page_index in pages_needing_ocr:
                pix = doc[page_index].get_pixmap(dpi=200)
                text_parts.append(_ocr(pix.tobytes("png")))

    full_text = "\n\n".join(part for part in text_parts if part).strip()
    if not full_text:
        raise ValueError(
            "No extractable text found in PDF. It looks like a scanned "
            "document without a text layer; enable OCR (OCR_ENABLED=true) "
            "to read it."
        )
    return normalize_text(full_text)


def _extract_from_image(file_bytes: bytes) -> str:
    if not get_settings().ocr_enabled:
        raise ValueError("Image uploads require OCR, but OCR_ENABLED is false.")
    text = _ocr(file_bytes)
    if not text:
        raise ValueError("OCR produced no text for this image.")
    return normalize_text(text)
