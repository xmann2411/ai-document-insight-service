"""
Generate the dummy test documents committed in this folder.

Run:  python test_docs/generate_test_docs.py

Produces:
  sample_invoice.pdf   - digital PDF with a text layer (PyMuPDF path)
  sample_contract.pdf  - digital PDF with a text layer (PyMuPDF path)
  service_report.pdf   - 2-page digital PDF
  scanned_receipt.png  - image only, no text layer (forces the OCR path)

All content is fictional.
"""

from pathlib import Path

import pymupdf

OUT = Path(__file__).parent

INVOICE = """INVOICE  #INV-2026-0341

Date of issue: 2026-08-15
Due date: 2026-09-14
Payment terms: Net 30 days

Bill to:
Acme Corp
123 Business Rd
10000 Zagreb, Croatia
VAT: HR12345678901

From:
TechConsult d.o.o.
Ulica Grada Vukovara 200
10000 Zagreb, Croatia
IBAN: HR1723600001101234565

Line items:
  1. Backend consulting - August 2026      40 hours   EUR 75.00/h    EUR 3,000.00
  2. On-call support retainer                1 month   EUR 400.00     EUR   400.00
  3. Cloud infrastructure pass-through       -         -              EUR   180.50

Subtotal: EUR 3,580.50
VAT (25%): EUR 895.13
Total due: EUR 4,475.63

Late payments are subject to statutory default interest.
Questions about this invoice: billing@techconsult.example
"""

CONTRACT = """SERVICE AGREEMENT

This Service Agreement ("Agreement") is entered into on 1 August 2026
between TechConsult d.o.o. ("Provider") and Acme Corp ("Client").

1. Scope of services
   Provider will deliver software engineering and consulting services
   as described in Appendix A (Statement of Work).

2. Term
   This Agreement is effective for an initial term of 12 months from the
   effective date and renews automatically for successive 12-month terms
   unless either party gives notice of non-renewal.

3. Fees
   Client shall pay Provider EUR 75.00 per hour, invoiced monthly.
   Invoices are payable within 30 days of receipt.

4. Termination
   Either party may terminate this Agreement for convenience with 30
   days' prior written notice. Either party may terminate immediately
   for material breach that remains uncured for 15 days after notice.

5. Confidentiality
   Each party shall keep the other party's confidential information
   secret for the term of this Agreement and for 2 years thereafter.

6. Limitation of liability
   Provider's total liability under this Agreement shall not exceed the
   fees paid by Client in the 6 months preceding the claim.

7. Governing law
   This Agreement is governed by the laws of the Republic of Croatia.
   The courts of Zagreb have exclusive jurisdiction.

Signed:
  For Provider: Ivana Horvat, Managing Director
  For Client:   John Smith, COO
"""

REPORT_PAGE_1 = """MONTHLY SERVICE REPORT - AUGUST 2026
Client: Acme Corp
Prepared by: TechConsult d.o.o.

1. Summary
   System availability for August 2026 was 99.94%, above the 99.9% SLA
   target. Two minor incidents were recorded, both resolved within the
   agreed response time. No data loss occurred.

2. Incident log
   INC-1042  2026-08-07  Elevated API latency (p95 1.8s) for 22 minutes.
                         Cause: undersized connection pool. Fixed by
                         raising pool size and adding an alert.
   INC-1043  2026-08-21  Background job queue backlog of 3,400 messages.
                         Cause: a poison message. Fixed by adding a
                         dead-letter queue.
"""

REPORT_PAGE_2 = """3. Capacity and cost
   Average CPU utilisation: 41%. Peak: 76% on 2026-08-19.
   Storage used: 812 GB of 1,000 GB provisioned.
   Estimated infrastructure cost for August: EUR 180.50.

4. Recommendations
   a. Upgrade the primary database instance before Q4 traffic growth.
   b. Introduce autoscaling for the worker tier.
   c. Schedule a disaster-recovery drill in September 2026.

5. Next review
   The next service review meeting is scheduled for 2026-09-05.
"""

RECEIPT = """CAFE NEBO
Trg bana Jelacica 1, Zagreb

Receipt #  4471
Date: 2026-08-22  14:38

1x Espresso            EUR 1.80
2x Flat white          EUR 6.40
1x Cheesecake          EUR 4.20

Subtotal               EUR 12.40
VAT 13%                 EUR  1.61
TOTAL                   EUR 14.01

Paid by card  ****3921
Thank you!
"""


def _text_pdf(path: Path, pages: list[str]) -> None:
    doc = pymupdf.open()
    for body in pages:
        page = doc.new_page(width=595, height=842)  # A4
        page.insert_text((56, 72), body, fontname="courier", fontsize=10)
    doc.save(path)
    doc.close()
    print("wrote", path.name)


def _image_png(path: Path, body: str) -> None:
    """Render text to a page, then export as a flat PNG (no text layer)."""
    doc = pymupdf.open()
    page = doc.new_page(width=420, height=560)
    page.insert_text((40, 60), body, fontname="courier", fontsize=12)
    pix = page.get_pixmap(dpi=150)
    pix.save(path)
    doc.close()
    print("wrote", path.name)


if __name__ == "__main__":
    _text_pdf(OUT / "sample_invoice.pdf", [INVOICE])
    _text_pdf(OUT / "sample_contract.pdf", [CONTRACT])
    _text_pdf(OUT / "service_report.pdf", [REPORT_PAGE_1, REPORT_PAGE_2])
    _image_png(OUT / "scanned_receipt.png", RECEIPT)
