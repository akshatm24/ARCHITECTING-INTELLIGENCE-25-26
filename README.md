# AI Research Assistant

Hybrid RAG service for scientific literature review. Upload a PDF, index the paper, ask grounded questions, generate summaries, and convert methodology sections into Mermaid flowcharts or mind maps.

## What It Does

- Extracts PDF text with PyMuPDF and chunks pages with LangChain splitters.
- Builds a hybrid retriever with ChromaDB vector search and BM25 lexical ranking.
- Uses Gemini Flash for source-grounded answers, summaries, and Mermaid diagrams.
- Returns cited chunks for traceability.
- Exposes an asynchronous FastAPI API ready for Render or Docker deployment.
- Keeps deterministic tests and benchmarks API-free.

## Architecture

```mermaid
flowchart TD
    A["PDF upload"] --> B["PyMuPDF page extraction"]
    B --> C["Overlapping text chunks"]
    C --> D["Google embeddings"]
    C --> E["BM25 tokens"]
    D --> F["ChromaDB vector index"]
    E --> G["BM25 lexical index"]
    F --> H["Hybrid retrieval merge"]
    G --> H
    H --> I["Gemini Flash prompt"]
    I --> J["Answer, citations, summary, Mermaid"]
```

## API

Run locally:

```bash
python3 -m pip install -r requirements-dev.txt
export GOOGLE_API_KEY="your_google_api_key"
export GOOGLE_MODEL="gemini-1.5-flash"
uvicorn app:app --reload
```

Open `http://127.0.0.1:8000/docs`.

Main endpoints:

- `GET /health` checks service readiness.
- `POST /ingest/pdf` uploads and indexes a PDF.
- `POST /ingest/text` indexes plain text for testing or demos.
- `POST /query` answers a question with citations.
- `POST /summary` summarizes the indexed paper.
- `POST /diagram` returns Mermaid `flowchart` or `mindmap` code.

Optional Streamlit UI:

```bash
streamlit run streamlit_app.py
```

## Deploy

### Render

This repo includes `render.yaml`.

1. Create a Render Blueprint from this GitHub repository.
2. Set `GOOGLE_API_KEY` in the service environment.
3. Deploy the generated web service.

Render runs:

```bash
uvicorn app:app --host 0.0.0.0 --port $PORT
```

### Docker

```bash
docker build -t ai-research-assistant .
docker run -p 8501:8501 --env-file .env ai-research-assistant
```

Then open `http://127.0.0.1:8501/docs`.

## Validate

```bash
pytest -q
python scripts/benchmark_metrics.py
```

The benchmark is deterministic and uses synthetic research-paper sections, so it does not spend Gemini API quota.

## Latest Local Metrics

Run on the deterministic benchmark after the cleanup:

```json
{
  "synthetic_papers": 120,
  "sample_words": 14640,
  "indexed_chunks": 160,
  "benchmark_queries": 5,
  "vector_only_hit_rate": 1.0,
  "hybrid_retrieval_hit_rate": 1.0,
  "hybrid_precision_lift_pct": 0.0,
  "index_build_ms": 25649.44,
  "median_retrieval_ms": 3.6,
  "p95_retrieval_ms": 3.85
}
```

The hybrid retriever is implemented and benchmarked, but this synthetic set does not show a lift because vector-only retrieval already reaches 100% hit rate.

## Resume Bullets

- Engineered a hybrid RAG pipeline using Gemini Flash to extract scientific methodologies from research papers into Mermaid flowcharts and citation-backed summaries.
- Implemented ChromaDB vector retrieval with BM25 lexical ranking, validating 100% retrieval hit rate across a 120-paper synthetic benchmark with 3.6 ms median retrieval latency.
- Deployed a non-blocking FastAPI architecture with Docker and Render blueprint support, plus API-free tests and deterministic retrieval benchmarks.

## Repository Layout

```text
.
├── ai_research_assistant/   # RAG pipeline and retrieval logic
├── app.py                   # FastAPI deployment entry point
├── streamlit_app.py         # Optional local Streamlit interface
├── scripts/                 # Benchmark script
├── tests/                   # API-free unit tests
├── Dockerfile
├── render.yaml
└── requirements.txt
```
