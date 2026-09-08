import pytest

from app.extraction import extract_text, is_supported
from tests.conftest import TEST_DOCS


@pytest.mark.parametrize(
    "filename,needle",
    [
        ("sample_invoice.pdf", "INV-2026-0341"),
        ("sample_contract.pdf", "SERVICE AGREEMENT"),
        ("service_report.pdf", "INC-1043"),  # on page 2 -> multi-page extraction
    ],
)
def test_extract_text_from_pdfs(filename, needle):
    data = (TEST_DOCS / filename).read_bytes()
    text = extract_text(filename, data)
    assert needle in text


def test_unsupported_type_raises():
    with pytest.raises(ValueError):
        extract_text("notes.docx", b"whatever")


def test_is_supported():
    assert is_supported("a.pdf")
    assert is_supported("scan.PNG")
    assert not is_supported("data.csv")


def test_image_without_ocr_raises():
    # conftest sets OCR_ENABLED=false
    data = (TEST_DOCS / "scanned_receipt.png").read_bytes()
    with pytest.raises(ValueError, match="OCR"):
        extract_text("scanned_receipt.png", data)
