# docs

Screenshots referenced from the top-level README.

| File | Shows |
|------|-------|
| `ui.png` | The Streamlit UI (`ui.py`) after a question – answer with highlighted entities, backend/confidence, and the expanded "Sources" panel with rerank scores. |
| `swagger.png` | `http://localhost:8000/docs` – the API surface. |

To regenerate `ui.png`:

```bash
uvicorn app.main:app --port 8000
API_URL=http://localhost:8000 \
  DEMO_DOCS=test_docs/sample_contract.pdf,test_docs/sample_invoice.pdf \
  streamlit run ui.py
```

Ask *"What is the total due and the due date on the invoice?"* and screenshot.
