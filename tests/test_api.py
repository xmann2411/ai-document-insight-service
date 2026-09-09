from tests.conftest import TEST_DOCS


def _upload(client, *filenames, session_id=None):
    files = [
        ("files", (name, (TEST_DOCS / name).read_bytes(), "application/octet-stream"))
        for name in filenames
    ]
    data = {"session_id": session_id} if session_id else {}
    return client.post("/upload", files=files, data=data)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["qa_backend"] in {"local", "claude"}


def test_upload_returns_session_and_records(client):
    r = _upload(client, "sample_invoice.pdf", "sample_contract.pdf")
    assert r.status_code == 200
    body = r.json()
    assert body["session_id"]
    assert [f["status"] for f in body["files"]] == ["ok", "ok"]
    assert body["files"][0]["chars"] > 0


def test_upload_rejects_unsupported_only(client):
    files = [("files", ("notes.docx", b"junk", "application/octet-stream"))]
    r = client.post("/upload", files=files)
    assert r.status_code == 422


def test_ask_flow_with_mocked_llm(client, fake_llm):
    session_id = _upload(client, "sample_contract.pdf").json()["session_id"]
    r = client.post(
        "/ask",
        data={"session_id": session_id, "question": "How can the contract be terminated?"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["session_id"] == session_id
    assert body["answer"].startswith("ANSWERED using model=")
    assert body["sources"], "sources should be returned"
    assert body["sources"][0]["filename"] == "sample_contract.pdf"


def test_ask_unknown_session_404(client, fake_llm):
    r = client.post("/ask", data={"session_id": "nope", "question": "hi"})
    assert r.status_code == 404


def test_ask_empty_question_422(client, fake_llm):
    session_id = _upload(client, "sample_invoice.pdf").json()["session_id"]
    r = client.post("/ask", data={"session_id": session_id, "question": "   "})
    assert r.status_code == 422


def test_ask_with_claude_backend_but_no_key_returns_503(client, monkeypatch):
    from app.qa_engine import QAConfigError

    def _boom():
        raise QAConfigError("ANTHROPIC_API_KEY is not set.")

    monkeypatch.setattr("app.qa_engine._get_claude_client", _boom)
    session_id = _upload(client, "sample_invoice.pdf").json()["session_id"]
    r = client.post("/ask", data={"session_id": session_id, "question": "total?"})
    assert r.status_code == 503


def test_session_info(client):
    session_id = _upload(client, "service_report.pdf").json()["session_id"]
    r = client.get(f"/sessions/{session_id}")
    assert r.status_code == 200
    body = r.json()
    assert body["documents"][0]["filename"] == "service_report.pdf"
    assert body["total_chunks"] >= 1
