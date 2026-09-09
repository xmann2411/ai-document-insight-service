"""
Tiny evaluation harness for the QA pipeline.

Runs a fixed question set (with expected-substring answers) through the
full retrieval + QA pipeline for one or more backends and prints an
accuracy table. This is what backs the "~9/10" figure in the README.

Usage:
    python scripts/eval.py                 # local + distilbert
    python scripts/eval.py local           # one backend
    python scripts/eval.py local claude    # needs ANTHROPIC_API_KEY

No server needed - it drives the pipeline in-process.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
TEST_DOCS = ROOT / "test_docs"

# doc, question, expected substring (case-insensitive). "" = expect "no answer".
CASES = [
    ("sample_invoice.pdf",  "What is the total due?",                                 "4,475.63"),
    ("sample_invoice.pdf",  "What is the invoice due date?",                          "2026-09-14"),
    ("sample_invoice.pdf",  "What are the payment terms?",                            "net 30"),
    ("sample_contract.pdf", "How much notice is required to terminate for convenience?", "30 days"),
    ("sample_contract.pdf", "How long does confidentiality last after the agreement ends?", "2 years"),
    ("sample_contract.pdf", "What is the hourly rate?",                               "75"),
    ("sample_contract.pdf", "What is the governing law?",                             "croatia"),
    ("sample_contract.pdf", "Which courts have jurisdiction?",                        "zagreb"),
    ("sample_contract.pdf", "What is the initial term of the agreement?",             "12 months"),
    ("sample_contract.pdf", "Who signed for the client?",                             "smith"),
    ("service_report.pdf",  "What was the system availability in August?",            "99.94"),
    ("service_report.pdf",  "When is the next service review?",                       "2026-09-05"),
    ("scanned_receipt.png", "What was the total on the receipt?",                     "14.01"),
    ("sample_contract.pdf", "What is Acme Corp's employee headcount?",                ""),  # absent
]


def build_session():
    from app import storage
    from app.extraction import extract_text

    sid = storage.create_session()
    for name in sorted({c[0] for c in CASES}):
        data = (TEST_DOCS / name).read_bytes()
        try:
            storage.add_document(sid, name, extract_text(name, data))
        except Exception as e:  # noqa: BLE001
            print(f"  ! could not ingest {name}: {e}")
    return sid


def run_backend(backend: str) -> tuple[int, int]:
    os.environ["QA_BACKEND"] = backend
    from app.config import get_settings

    get_settings.cache_clear()
    from app import storage
    from app.qa_engine import answer_question

    sid = build_session()
    passed = 0
    print(f"\n=== {backend} ===")
    for doc, question, expected in CASES:
        chunks = storage.search(sid, question)
        answer = answer_question(question, chunks)["answer"]
        if expected:
            ok = expected.lower() in answer.lower()
        else:
            ok = "don't appear to contain" in answer.lower()
        passed += ok
        flag = "ok " if ok else "XX "
        print(f"  [{flag}] {question}")
        print(f"         {answer[:100]!r}")
    print(f"  -> {passed}/{len(CASES)}")
    return passed, len(CASES)


def main() -> None:
    backends = sys.argv[1:] or ["local", "distilbert"]
    results = {b: run_backend(b) for b in backends}
    print("\nSUMMARY")
    for b, (p, t) in results.items():
        print(f"  {b:12} {p}/{t}  ({100 * p / t:.0f}%)")


if __name__ == "__main__":
    main()
