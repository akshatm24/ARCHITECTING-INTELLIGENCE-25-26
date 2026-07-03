# Architecting Intelligence: AI Research Assistant

Group 2 capstone project for the MATSOC Architecting Intelligence winter project.

## What It Does

This project turns dense research papers into an interactive, source-grounded assistant:

- Upload a PDF research paper.
- Extract and chunk page-level text with PyMuPDF and LangChain text splitters.
- Embed chunks with Google `text-embedding-004`.
- Store and retrieve evidence from ChromaDB.
- Answer questions with Gemini 2.5 Flash using only retrieved context.
- Generate Mermaid mind maps and flowcharts for methodology, experiments, and paper structure.
- Show retrieved page/chunk citations for answer traceability.

## Project Structure

```text
Group_2/
├── app.py                              # Streamlit app
├── ai_research_assistant/
│   ├── __init__.py
│   └── pipeline.py                     # PDF ingestion, retrieval, Q&A, Mermaid generation
├── scripts/
│   └── benchmark_metrics.py            # Deterministic local metrics benchmark
├── tests/
│   └── test_pipeline.py                # API-free unit tests
├── requirements.txt
├── requirements-dev.txt
└── .env.example
```

## Run Locally

```bash
cd "Submissions/Capstone Project/Group_2"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export GOOGLE_API_KEY="your_key_here"
streamlit run app.py
```

## Validate

```bash
cd "Submissions/Capstone Project/Group_2"
pip install -r requirements-dev.txt
pytest -q
python scripts/benchmark_metrics.py
```

The tests use fake embeddings and a fake LLM, so they do not need external API keys.

## Benchmark Metrics

`scripts/benchmark_metrics.py` runs a deterministic benchmark on synthetic research-paper-like text, so the retrieval and latency numbers are reproducible without paid APIs. On the current implementation it measures:

- indexed chunks
- retrieval hit rate across methodology/retrieval/grounding/diagram/workflow queries
- index build latency
- median and p95 retrieval latency

Latest local run:

```json
{
  "sample_words": 1702,
  "indexed_chunks": 19,
  "benchmark_queries": 5,
  "retrieval_hit_rate": 1.0,
  "index_build_ms": 206.67,
  "median_retrieval_ms": 0.56,
  "p95_retrieval_ms": 1.13
}
```

Use the printed JSON output as the source of truth if the benchmark is rerun on another machine.

## Resume Bullets

- Built a source-grounded RAG research assistant with Gemini 2.5 Flash, Google embeddings, ChromaDB, PyMuPDF, and Streamlit for PDF Q&A, summaries, and citation-backed answers.
- Refactored the notebook prototype into a runnable Python project with modular ingestion/retrieval logic, API-free unit tests, and a deterministic benchmark harness.
- Generated Mermaid mind maps and flowcharts from retrieved paper context to visualize methodology, experiments, setup, and key findings.
- Validated retrieval behavior with a local benchmark over 1,702 sample words, 19 indexed chunks, 5 research-workflow query types, 100% retrieval hit rate, and 0.56 ms median retrieval latency.

## Contributors

Mayank Verma, Utkarsh Singhal, Azaad Katiyar, Rudra, Vishnu Sarathy, Venkat Sai
