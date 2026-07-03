from __future__ import annotations

import json
import math
import sys
import statistics
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai_research_assistant.pipeline import RAGPipeline


class KeywordEmbeddings:
    VOCAB = [
        "retrieval",
        "embedding",
        "chromadb",
        "citation",
        "mermaid",
        "flowchart",
        "summary",
        "hallucination",
        "experiment",
        "methodology",
    ]

    def _vec(self, text: str) -> list[float]:
        lower = text.lower()
        values = [lower.count(term) for term in self.VOCAB]
        values.append(len(text.split()) / 250)
        norm = math.sqrt(sum(value * value for value in values)) or 1.0
        return [value / norm for value in values]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


class StaticLLM:
    def invoke(self, prompt):
        class Response:
            content = "Retrieved context supports the answer and includes page citations [p. 1]."

        return Response()


SAMPLE_SECTIONS = [
    "Methodology: The system extracts PDF text, splits it into overlapping chunks, creates Google-style embeddings, and stores vectors in ChromaDB for retrieval augmented generation.",
    "Retrieval: User questions are embedded and matched against the ChromaDB index. Sparse BM25 keyword matching rescues exact scientific terminology and method names.",
    "Grounding: The answer prompt requires page citations and instructs the model to say when evidence is missing, reducing hallucination risk.",
    "Visualization: Gemini generates Mermaid flowcharts and mind maps to summarize methodology, experiments, setup, results, and limitations.",
    "Research workflow: Scientists can upload dense papers, ask conversational questions, generate summaries, and inspect cited chunks.",
    "Experiments: Evaluation queries cover methodology, retrieval, hallucination control, diagrams, and literature review workflow.",
]

QUERIES = [
    ("methodology", "methodology chunks embeddings chromadb"),
    ("retrieval", "retrieval matched chunks index"),
    ("grounding", "citations hallucination missing evidence"),
    ("diagrams", "mermaid flowchart mind map"),
    ("workflow", "research workflow dense papers"),
]


def main() -> None:
    synthetic_papers = 120
    sample_text = "\n\n".join(
        f"Paper {paper_id}. {section}"
        for paper_id in range(1, synthetic_papers + 1)
        for section in SAMPLE_SECTIONS
    )
    rag = RAGPipeline(embeddings=KeywordEmbeddings(), llm=StaticLLM(), retrieval_k=4)

    ingest_start = time.perf_counter()
    chunks = rag.ingest_text(sample_text, source="synthetic benchmark")
    ingest_ms = (time.perf_counter() - ingest_start) * 1000

    latencies = []
    hybrid_hits = 0
    vector_hits = 0
    for label, query in QUERIES:
        start = time.perf_counter()
        docs = rag.retrieve(query)
        latencies.append((time.perf_counter() - start) * 1000)
        hybrid_joined = " ".join(doc.page_content.lower() for doc in docs)
        hybrid_hits += int(label in hybrid_joined or query.split()[0] in hybrid_joined)

        vector_docs = rag.retriever.invoke(query)
        vector_joined = " ".join(doc.page_content.lower() for doc in vector_docs)
        vector_hits += int(label in vector_joined or query.split()[0] in vector_joined)

    vector_hit_rate = vector_hits / len(QUERIES)
    hybrid_hit_rate = hybrid_hits / len(QUERIES)
    precision_lift = 0 if vector_hit_rate == 0 else ((hybrid_hit_rate - vector_hit_rate) / vector_hit_rate) * 100

    metrics = {
        "synthetic_papers": synthetic_papers,
        "sample_words": len(sample_text.split()),
        "indexed_chunks": chunks,
        "benchmark_queries": len(QUERIES),
        "vector_only_hit_rate": round(vector_hit_rate, 3),
        "hybrid_retrieval_hit_rate": round(hybrid_hit_rate, 3),
        "hybrid_precision_lift_pct": round(precision_lift, 1),
        "index_build_ms": round(ingest_ms, 2),
        "median_retrieval_ms": round(statistics.median(latencies), 2),
        "p95_retrieval_ms": round(sorted(latencies)[int(len(latencies) * 0.95) - 1], 2),
    }
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
