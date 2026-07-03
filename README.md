# AI Research Assistant

Source-grounded RAG app for reading dense research papers. Upload a PDF, ask questions, generate summaries, and turn methodology sections into Mermaid mind maps or flowcharts.

## Features

- PDF text extraction with PyMuPDF
- Chunking with LangChain recursive splitters
- Vector retrieval with ChromaDB
- Gemini 2.5 Flash responses grounded only in retrieved chunks
- Google `text-embedding-004` embeddings
- Page/chunk citations for answer traceability
- Mermaid mind map and flowchart generation
- Streamlit UI ready for local use or cloud deployment
- API-free tests and deterministic benchmark script

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export GOOGLE_API_KEY="your_google_api_key"
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501).

## Deploy

### Streamlit Community Cloud

1. Push this repo to GitHub.
2. Go to Streamlit Community Cloud and create a new app from this repository.
3. Set the entry point to `app.py`.
4. Add `GOOGLE_API_KEY` in app secrets.
5. Deploy.

### Render

This repo includes `render.yaml`, so it can be deployed as a Render Blueprint.

1. Create a new Render Blueprint from the GitHub repo.
2. Add `GOOGLE_API_KEY` as an environment variable.
3. Deploy the generated web service.

### Docker

```bash
docker build -t ai-research-assistant .
docker run -p 8501:8501 --env-file .env ai-research-assistant
```

## Validate

```bash
pip install -r requirements-dev.txt
pytest -q
python scripts/benchmark_metrics.py
```

Latest local validation:

```json
{
  "tests": "2 passed",
  "sample_words": 1702,
  "indexed_chunks": 19,
  "benchmark_queries": 5,
  "retrieval_hit_rate": 1.0,
  "index_build_ms": 271.12,
  "median_retrieval_ms": 0.98,
  "p95_retrieval_ms": 1.01
}
```

## Resume Metrics

- Built a source-grounded RAG research assistant using Gemini 2.5 Flash, Google embeddings, ChromaDB, PyMuPDF, and Streamlit for PDF Q&A, summaries, citation-backed answers, and Mermaid visualizations.
- Refactored the notebook prototype into a deployable Python project with Docker, Render blueprint config, CI, deterministic benchmarking, and API-free unit tests.
- Validated retrieval behavior over 1,702 sample words, 19 indexed chunks, 5 research-workflow query types, 100% retrieval hit rate, and 0.98 ms median retrieval latency.

## Repository Notes

The original course materials are preserved under `Submissions/`, `Assignments/`, and `Codes/`. The runnable project lives at the repository root.
