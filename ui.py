"""
Streamlit demo UI for the Document Insight Service.

A thin client over the REST API - it calls POST /upload and POST /ask just
like any other consumer. Point it at a running API with API_URL
(default http://localhost:8000).

Run:
    uvicorn app.main:app --port 8000      # in one terminal
    streamlit run ui.py                   # in another
"""

from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="Document Insight", page_icon="📄", layout="centered")
st.session_state.setdefault("history", [])
st.session_state.setdefault("docs", [])


@st.cache_data(ttl=5)
def get_health():
    try:
        return requests.get(f"{API_URL}/health", timeout=5).json()
    except requests.RequestException:
        return None


def do_upload(files) -> None:
    payload = [
        ("files", (f.name, f.getvalue(), f.type or "application/octet-stream"))
        for f in files
    ]
    sid = st.session_state.get("session_id")
    url = f"{API_URL}/sessions/{sid}/documents" if sid else f"{API_URL}/upload"
    try:
        resp = requests.post(url, files=payload, timeout=180)
    except requests.RequestException as e:
        st.error(f"Upload failed: {e}")
        return
    if resp.status_code not in (200, 201):
        st.error(f"Upload failed ({resp.status_code}): {resp.text}")
        return
    data = resp.json()
    st.session_state.session_id = data["session_id"]
    st.session_state.docs.extend(data["files"])
    ok = sum(f["status"] == "ok" for f in data["files"])
    st.toast(f"Indexed {ok} file(s)")


def do_ask(question: str):
    try:
        resp = requests.post(
            f"{API_URL}/ask",
            data={"session_id": st.session_state.session_id, "question": question},
            timeout=180,
        )
    except requests.RequestException as e:
        return {"error": str(e)}
    if resp.status_code != 200:
        return {"error": f"{resp.status_code}: {resp.text}"}
    return resp.json()


def render_answer(result: dict) -> None:
    if "error" in result:
        st.error(result["error"])
        return
    st.write(result["answer"])
    left, right = st.columns(2)
    left.caption(f"backend: {result.get('backend', '?')}")
    if "confidence" in result:
        right.caption(f"confidence: {result['confidence']:.2f}")
    srcs = result.get("sources", [])
    with st.expander(f"Sources ({len(srcs)})"):
        for s in srcs:
            used = " ⭐ used" if s.get("used") else ""
            if "rerank_score" in s:
                score = f" · rerank {s['rerank_score']}"
            elif "score" in s:
                score = f" · score {s['score']}"
            else:
                score = ""
            st.markdown(f"**{s['filename']}**{used}{score}")
            st.caption(s["snippet"])


# ----------------------------------------------------------------- sidebar
health = get_health()
with st.sidebar:
    st.subheader("Service")
    if health:
        st.caption(f"API: `{API_URL}`")
        st.caption(f"QA backend: **{health['qa_backend']}**")
        st.caption(
            f"retrieval: {health['retrieval_mode']} · "
            f"OCR: {'on' if health['ocr_enabled'] else 'off'}"
        )
    else:
        st.error(f"API not reachable at {API_URL}")

    st.divider()
    st.subheader("Documents")
    uploaded = st.file_uploader(
        "PDF or image files",
        type=["pdf", "png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"],
        accept_multiple_files=True,
    )
    if st.button("Upload", disabled=not uploaded, use_container_width=True):
        do_upload(uploaded)

    for d in st.session_state.docs:
        if d["status"] == "ok":
            st.write(f"✅ {d['filename']} · {d['chunks']} chunks")
        else:
            st.write(f"⚠️ {d['filename']} · {d['status']}")
    if st.session_state.get("session_id"):
        st.caption(f"session: `{st.session_state.session_id}`")
        if st.button("New session", use_container_width=True):
            for k in ("session_id", "docs", "history"):
                st.session_state.pop(k, None)
            st.rerun()


# ------------------------------------------------------------------- main
st.title("📄 Document Insight")
st.caption("Upload documents in the sidebar, then ask questions about them.")

if not st.session_state.get("session_id"):
    st.info("⬅️ Upload one or more documents to get started.")
    st.stop()

for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["q"])
    with st.chat_message("assistant"):
        render_answer(turn["a"])

question = st.chat_input("Ask a question about the documents")
if question:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            result = do_ask(question)
        render_answer(result)
    st.session_state.history.append({"q": question, "a": result})
