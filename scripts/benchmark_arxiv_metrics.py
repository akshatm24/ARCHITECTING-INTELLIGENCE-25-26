from __future__ import annotations

import json
import math
import statistics
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai_research_assistant.pipeline import RAGPipeline


ARXIV_CACHE = Path(__file__).resolve().parents[1] / ".cache" / "arxiv_cs_ai_120.json"


class TopicEmbeddings:
    VOCAB = [
        "language",
        "vision",
        "retrieval",
        "reasoning",
        "agent",
        "graph",
        "diffusion",
        "transformer",
        "optimization",
        "alignment",
        "multimodal",
        "benchmark",
    ]

    def _vec(self, text: str) -> list[float]:
        lower = text.lower()
        values = [lower.count(term) for term in self.VOCAB]
        values.append(len(text.split()) / 400)
        norm = math.sqrt(sum(value * value for value in values)) or 1.0
        return [value / norm for value in values]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


class StaticLLM:
    def invoke(self, prompt):
        class Response:
            content = "The retrieved context supports the answer with citations [p. 1]."

        return Response()


def fetch_arxiv_papers(limit: int = 120) -> list[dict[str, str]]:
    if ARXIV_CACHE.exists():
        return json.loads(ARXIV_CACHE.read_text())

    papers: list[dict[str, str]] = []
    batch_size = 30
    for start in range(0, limit, batch_size):
        payload = fetch_arxiv_batch(start=start, batch_size=batch_size)
        papers.extend(parse_arxiv_payload(payload))
        time.sleep(3.1)

    if len(papers) < 100:
        raise RuntimeError(f"Expected at least 100 arXiv papers, received {len(papers)}.")

    ARXIV_CACHE.parent.mkdir(parents=True, exist_ok=True)
    ARXIV_CACHE.write_text(json.dumps(papers[:limit], indent=2))
    return papers[:limit]


def fetch_arxiv_batch(start: int, batch_size: int) -> bytes:
    query = urllib.parse.urlencode(
        {
            "search_query": "cat:cs.CL OR cat:cs.AI OR cat:cs.LG",
            "start": start,
            "max_results": batch_size,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
    )
    last_error: Exception | None = None
    for attempt in range(3):
        request = urllib.request.Request(
            f"https://export.arxiv.org/api/query?{query}",
            headers={"User-Agent": "ai-research-assistant-benchmark/1.0"},
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except Exception as exc:
            last_error = exc
            time.sleep(4 + attempt * 4)
    raise RuntimeError(f"arXiv API request failed for start={start}: {last_error}") from last_error


def parse_arxiv_payload(payload: bytes) -> list[dict[str, str]]:
    root = ET.fromstring(payload)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    papers = []
    for index, entry in enumerate(root.findall("atom:entry", ns), start=1):
        title = " ".join(entry.findtext("atom:title", default="", namespaces=ns).split())
        summary = " ".join(entry.findtext("atom:summary", default="", namespaces=ns).split())
        arxiv_id = entry.findtext("atom:id", default=f"paper-{index}", namespaces=ns).rsplit("/", 1)[-1]
        if title and summary:
            papers.append({"id": arxiv_id, "title": title, "summary": summary})
    return papers


def paper_text(papers: list[dict[str, str]]) -> str:
    sections = []
    for page, paper in enumerate(papers, start=1):
        sections.append(
            " ".join(
                [
                    f"Paper {page}",
                    f"arXiv ID: {paper['id']}.",
                    f"Title: {paper['title']}.",
                    f"Abstract: {paper['summary']}",
                ]
            )
        )
    return "\n\n".join(sections)


def title_query(title: str) -> str:
    words = [
        word.strip(".,:;()[]{}\"'").lower()
        for word in title.split()
        if len(word.strip(".,:;()[]{}\"'")) >= 7
    ]
    return " ".join(words[:5]) or title.lower()


def hit_at_k(docs, expected_id: str) -> int:
    return int(any(doc.metadata.get("source") == expected_id or expected_id in doc.page_content for doc in docs))


def main() -> None:
    papers = fetch_arxiv_papers()
    rag = RAGPipeline(embeddings=TopicEmbeddings(), llm=StaticLLM(), retrieval_k=5)

    ingest_start = time.perf_counter()
    docs = []
    from langchain_core.documents import Document

    for page, paper in enumerate(papers, start=1):
        docs.append(Document(page_content=paper_text([paper]), metadata={"page": page, "source": paper["id"]}))
    chunks = rag.ingest_documents(docs)
    index_build_ms = (time.perf_counter() - ingest_start) * 1000

    queries = [(paper["id"], title_query(paper["title"])) for paper in papers]
    vector_hits = 0
    hybrid_hits = 0
    hybrid_latencies = []
    vector_latencies = []

    for expected_id, query in queries:
        start = time.perf_counter()
        vector_docs = rag.retrieve_vector_only(query)
        vector_latencies.append((time.perf_counter() - start) * 1000)
        vector_hits += hit_at_k(vector_docs, expected_id)

        start = time.perf_counter()
        hybrid_docs = rag.retrieve(query)
        hybrid_latencies.append((time.perf_counter() - start) * 1000)
        hybrid_hits += hit_at_k(hybrid_docs, expected_id)

    vector_precision = vector_hits / len(queries)
    hybrid_precision = hybrid_hits / len(queries)
    precision_lift = 0 if vector_precision == 0 else ((hybrid_precision - vector_precision) / vector_precision) * 100

    metrics = {
        "arxiv_papers": len(papers),
        "indexed_chunks": chunks,
        "benchmark_queries": len(queries),
        "vector_only_precision_at_5": round(vector_precision, 3),
        "hybrid_precision_at_5": round(hybrid_precision, 3),
        "hybrid_precision_lift_pct": round(precision_lift, 1),
        "index_build_ms": round(index_build_ms, 2),
        "median_vector_retrieval_ms": round(statistics.median(vector_latencies), 2),
        "median_hybrid_retrieval_ms": round(statistics.median(hybrid_latencies), 2),
        "p95_hybrid_retrieval_ms": round(sorted(hybrid_latencies)[int(len(hybrid_latencies) * 0.95) - 1], 2),
        "latency_target_ms": 1800,
        "latency_target_met": statistics.median(hybrid_latencies) < 1800,
    }
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
