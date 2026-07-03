import math

from ai_research_assistant.pipeline import RAGPipeline, clean_mermaid


class HashEmbeddings:
    def _vec(self, text: str) -> list[float]:
        words = text.lower().split()
        features = [
            sum("retrieval" in word or "rag" in word for word in words),
            sum("flowchart" in word or "mermaid" in word for word in words),
            sum("citation" in word or "grounded" in word for word in words),
            len(words) / 100,
        ]
        norm = math.sqrt(sum(value * value for value in features)) or 1.0
        return [value / norm for value in features]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


class EchoLLM:
    def invoke(self, prompt):
        class Response:
            content = "The answer is grounded in retrieved context [p. 1]."

        return Response()


def test_clean_mermaid_removes_fences():
    assert clean_mermaid("```mermaid\ngraph TD\nA-->B\n```") == "graph TD\nA-->B"


def test_ingest_retrieve_and_answer_without_external_api(tmp_path):
    text = (
        "Retrieval augmented generation indexes paper chunks and retrieves evidence. "
        "Grounded answers cite the source page and avoid unsupported claims. "
        "Mermaid flowcharts summarize the method and experiments. "
    ) * 12
    rag = RAGPipeline(
        embeddings=HashEmbeddings(),
        llm=EchoLLM(),
        persist_directory=tmp_path,
        retrieval_k=2,
    )
    chunks = rag.ingest_text(text)
    answer = rag.answer_query("How does retrieval keep answers grounded?")

    assert chunks >= 1
    assert "grounded" in answer.text.lower()
    assert answer.citations
    assert answer.latency_ms >= 0


def test_hybrid_retrieval_uses_bm25_signal(tmp_path):
    text = (
        "Alpha section explains neural retrieval and embeddings. "
        "Beta section describes sparse keyword matching with bm25 ranking. "
        "Gamma section covers mermaid flowcharts for methodology review. "
    ) * 12
    rag = RAGPipeline(
        embeddings=HashEmbeddings(),
        llm=EchoLLM(),
        persist_directory=tmp_path,
        retrieval_k=3,
    )

    rag.ingest_text(text)
    docs = rag.retrieve("bm25 sparse keyword matching")

    assert docs
    assert any("bm25" in doc.page_content.lower() for doc in docs)
