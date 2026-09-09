# Test documents

Fictional dummy documents used for manual testing and the automated test suite.

| File | Type | Extraction path it exercises |
|------|------|------------------------------|
| `sample_invoice.pdf` | Digital PDF (text layer) | PyMuPDF |
| `sample_contract.pdf` | Digital PDF (text layer) | PyMuPDF |
| `service_report.pdf` | 2-page digital PDF | PyMuPDF, multi-page |
| `scanned_receipt.png` | Image, no text layer | Tesseract OCR |

All content is invented (companies, names, amounts, IBANs).

Regenerate with:

```bash
python test_docs/generate_test_docs.py
```

Sample questions to try:

- Invoice: *"What is the total due?"* · *"What is the payment term?"*
- Contract: *"How much notice is required to terminate for convenience?"*
- Contract: *"How long does confidentiality last after the agreement ends?"*
- Report: *"What caused incident INC-1043?"*
- Receipt (OCR): *"What was the total?"*

The local extractive model does best on direct factual questions; for anything
needing synthesis across excerpts, run with `QA_BACKEND=claude`.
