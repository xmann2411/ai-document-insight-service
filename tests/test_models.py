"""
Opt-in tests that download and run the real models (embedding + local QA).

Skipped by default because they pull ~150 MB on first run. Enable with:

    RUN_MODEL_TESTS=1 pytest tests/test_models.py
"""

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_MODEL_TESTS") != "1",
    reason="set RUN_MODEL_TESTS=1 to run model-download tests",
)


def test_embedding_retrieval_ranks_relevant_chunk_first():
    from app.rag import SessionIndex

    filler = "The weather in the region is mild with occasional rain. " * 30
    answer_para = (
        "The invoice total due is EUR 4,475.63 and it is payable within 30 days. "
    ) * 5

    idx = SessionIndex()
    idx.mode = "embedding"
    idx.add_document("doc.txt", filler + "\n\n" + answer_para + "\n\n" + filler)

    hits = idx.search("How much money is owed on the invoice?")
    assert len(hits) >= 2
    assert "4,475.63" in hits[0]["text"]
    assert hits[0]["score"] >= hits[-1]["score"]


def test_local_qa_extracts_answer_span():
    from app.qa_local import answer_question

    chunks = [
        {
            "filename": "invoice.pdf",
            "text": (
                "INVOICE #INV-2026-0341. Date of issue: 2026-08-15. "
                "Due date: 2026-09-14. Subtotal EUR 3,580.50. "
                "Total due: EUR 4,475.63. Payment terms: Net 30 days."
            ),
        }
    ]
    result = answer_question("What is the total due?", chunks)
    assert "4,475.63" in result["answer"]
    assert result["confidence"] > 0
    assert result["sources"][0]["used"] is True
