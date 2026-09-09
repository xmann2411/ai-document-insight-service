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


_INVOICE_CHUNK = {
    "filename": "invoice.pdf",
    "text": (
        "INVOICE #INV-2026-0341. Date of issue: 2026-08-15. "
        "Due date: 2026-09-14. Subtotal EUR 3,580.50. "
        "Total due: EUR 4,475.63. Payment terms: Net 30 days."
    ),
}


def test_distilbert_qa_extracts_answer_span():
    from app.qa_local import answer_question

    result = answer_question("What is the total due?", [_INVOICE_CHUNK])
    assert "4,475.63" in result["answer"]
    assert result["confidence"] > 0
    assert result["sources"][0]["used"] is True


def test_ner_tags_people_orgs_money_dates(monkeypatch):
    from app.config import Settings
    from app import ner

    monkeypatch.setattr(ner, "get_settings", lambda: Settings(ner_enabled=True))
    from app.ner import extract_entities

    text = (
        "On 1 August 2026 TechConsult d.o.o. agreed to pay Acme Corp "
        "EUR 4,475.63, signed by John Smith in Zagreb."
    )
    labels = {e["label"] for e in extract_entities(text)}
    assert "MONEY" in labels
    assert "DATE" in labels
    assert "PERSON" in labels
    assert "ORG" in labels
    # spans must be valid slices of the input
    for e in extract_entities(text):
        assert text[e["start"]:e["end"]] == e["text"]


def test_sentence_qa_returns_relevant_sentence():
    from app.qa_sentences import answer_question

    contract = {
        "filename": "contract.pdf",
        "text": (
            "Fees. Client shall pay Provider EUR 75.00 per hour. "
            "Termination. Either party may terminate for convenience with "
            "30 days prior written notice. "
            "Confidentiality. Each party shall keep information secret for "
            "2 years after the agreement ends."
        ),
    }
    result = answer_question(
        "How long does confidentiality last after the agreement ends?",
        [_INVOICE_CHUNK, contract],
    )
    assert "2 years" in result["answer"]
    assert result["sources"], "sources returned"
    assert any(s["used"] for s in result["sources"])
