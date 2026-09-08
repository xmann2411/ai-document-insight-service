"""
Text extraction from uploaded documents.

Strategy (chosen to match the assignment's "PDF or image documents,
e.g. scanned contracts, invoices"):

1. PDF with a text layer  -> PyMuPDF (fitz). Fast, no ML, exact text.
   Most real contracts/invoices are digitally generated and fall here.
2. PDF that is a scanned image (no text layer) -> rasterise each page
   with PyMuPDF, then OCR with EasyOCR.
3. Image upload (.png/.jpg/.tiff/...) -> OCR directly with EasyOCR.

EasyOCR is imported lazily so the service still starts (and PDF-text
extraction still works) in environments where the OCR stack isn't
installed. Set OCR_ENABLED=false to disable it entirely.
"""

from __future__ import annotations

import io
import logging

import pymupdf

from app.config import get_settings

logger = logging.getLogger(__name__)

PDF_EXTENSIONS = {".pdf"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
SUPPORTED_EXTENSIONS = PDF_EXTENSIONS | IMAGE_EXTENSIONS

# Below this many characters we assume the PDF page has no real text layer
# and fall back to OCR.
_MIN_TEXT_LAYER_CHARS = 20

_ocr_reader = None


def _get_ocr_reader():
    """Lazily build a single shared EasyOCR reader (model load is ~1s)."""
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr  # heavy import - only paid for when OCR is actually needed

        settings = get_settings()
        logger.info("Loading EasyOCR model (languages=%s)", settings.ocr_language_list)
        _ocr_reader = easyocr.Reader(settings.ocr_language_list, gpu=False)
    return _ocr_reader


def _ocr_image_bytes(image_bytes: bytes) -> str:
    reader = _get_ocr_reader()
    # detail=0 -> return just the strings, in reading order
    lines = reader.readtext(image_bytes, detail=0, paragraph=True)
    return "\n".join(lines).strip()


def is_supported(filename: str) -> bool:
    return _ext(filename) in SUPPORTED_EXTENSIONS


def _ext(filename: str) -> str:
    _, _, tail = filename.rpartition(".")
    return f".{tail.lower()}" if tail else ""


def extract_text(filename: str, file_bytes: bytes) -> str:
    """
    Extract plain text from an uploaded document.

    Raises ValueError for unsupported types or when nothing could be
    extracted (e.g. a scanned PDF while OCR is disabled).
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
                text_parts.append(_ocr_image_bytes(pix.tobytes("png")))

    full_text = "\n".join(part for part in text_parts if part).strip()

    if not full_text:
        raise ValueError(
            "No extractable text found in PDF. It looks like a scanned "
            "document without a text layer; enable OCR (OCR_ENABLED=true) "
            "to read it."
        )
    return full_text


def _extract_from_image(file_bytes: bytes) -> str:
    settings = get_settings()
    if not settings.ocr_enabled:
        raise ValueError(
            "Image uploads require OCR, but OCR_ENABLED is false."
        )
    text = _ocr_image_bytes(file_bytes)
    if not text:
        raise ValueError("OCR produced no text for this image.")
    return text


# Backwards-compatible alias (older code / tests imported this name).
def extract_text_from_pdf(file_bytes: bytes) -> str:
    return _extract_from_pdf(file_bytes)
