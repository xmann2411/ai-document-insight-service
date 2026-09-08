# Test documents

Fictional dummy documents used for manual testing and the automated test suite.

| File | Type | Extraction path it exercises |
|------|------|------------------------------|
| `sample_invoice.pdf` | Digital PDF (text layer) | PyMuPDF |
| `sample_contract.pdf` | Digital PDF (text layer) | PyMuPDF |
| `service_report.pdf` | 2-page digital PDF | PyMuPDF, multi-page |
| `scanned_receipt.png` | Image, no text layer | EasyOCR |

All content is invented (companies, names, amounts, IBANs).

Regenerate them with:

```bash
python test_docs/generate_test_docs.py
```

Sample questions to try:

- Invoice: *"What is the total due and the due date?"*
- Contract: *"How much notice is required to terminate for convenience?"*
- Contract: *"How long does confidentiality survive after the agreement ends?"*
- Report: *"What caused incident INC-1043 and how was it fixed?"*
- Receipt (OCR): *"What was the total and how was it paid?"*
