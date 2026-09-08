"""
Shared test fixtures.

Tests run in RETRIEVAL_MODE=full so they don't need the embedding stack
(sentence-transformers / torch). The Claude call is always mocked - no
network, no API key required.
"""

import os
from pathlib import Path

os.environ.setdefault("RETRIEVAL_MODE", "full")
os.environ.setdefault("OCR_ENABLED", "false")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-used")

import pytest
from fastapi.testclient import TestClient

from app import storage
from app.main import app

TEST_DOCS = Path(__file__).parent.parent / "test_docs"


@pytest.fixture(autouse=True)
def _clean_storage():
    storage.reset()
    yield
    storage.reset()


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def fake_llm(monkeypatch):
    """Replace the Anthropic call with a canned response that echoes context."""

    class _Block:
        type = "text"

        def __init__(self, text):
            self.text = text

    class _Response:
        def __init__(self, text):
            self.content = [_Block(text)]

    class _Messages:
        def create(self, *, model, max_tokens, system, messages):
            user = messages[0]["content"]
            return _Response(f"ANSWERED using model={model}. Context chars={len(user)}")

    class _FakeClient:
        messages = _Messages()

    monkeypatch.setattr("app.qa_engine._get_client", lambda: _FakeClient())
    return _FakeClient
