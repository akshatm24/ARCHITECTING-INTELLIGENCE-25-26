from __future__ import annotations

import os
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol

import fitz
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


class EmbeddingModel(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...


@dataclass(frozen=True)
class Citation:
    page: int
    chunk: int
    preview: str


@dataclass(frozen=True)
class Answer:
    text: str
    citations: list[Citation]
    latency_ms: float


def clean_mermaid(raw: str) -> str:
    """Strip markdown fences and keep only Mermaid code."""
    code = raw.strip()
    code = re.sub(r"^```(?:mermaid)?", "", code, flags=re.IGNORECASE).strip()
    code = re.sub(r"```$", "", code).strip()
    return code


def extract_pdf_pages(pdf_path: str | Path) -> list[Document]:
    pages: list[Document] = []
    with fitz.open(str(pdf_path)) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()
            if text:
                pages.append(Document(page_content=text, metadata={"page": page_number}))
    return pages


def split_documents(
    docs: Iterable[Document],
    chunk_size: int = 900,
    chunk_overlap: int = 150,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )
    chunks = splitter.split_documents(list(docs))
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk"] = index
    return chunks


class RAGPipeline:
    def __init__(
        self,
        api_key: str | None = None,
        embeddings: EmbeddingModel | None = None,
        llm: object | None = None,
        persist_directory: str | Path | None = None,
        retrieval_k: int = 6,
    ) -> None:
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.retrieval_k = retrieval_k
        self.persist_directory = Path(persist_directory or tempfile.mkdtemp(prefix="ai-research-chroma-"))
        self.full_text = ""
        self.chunks: list[Document] = []
        self.vector_store: Chroma | None = None
        self.retriever = None

        if embeddings is not None:
            self.embeddings = embeddings
        else:
            if not self.api_key:
                raise ValueError("GOOGLE_API_KEY is required unless a custom embedding model is supplied.")
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004",
                google_api_key=self.api_key,
            )

        if llm is not None:
            self.llm = llm
        else:
            if not self.api_key:
                raise ValueError("GOOGLE_API_KEY is required unless a custom LLM is supplied.")
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.2,
                google_api_key=self.api_key,
            )

    def ingest_pdf(self, pdf_path: str | Path) -> int:
        pages = extract_pdf_pages(pdf_path)
        if not pages:
            raise ValueError("No extractable text found in the PDF.")
        return self.ingest_documents(pages)

    def ingest_text(self, text: str, source: str = "inline") -> int:
        if len(text.strip()) < 100:
            raise ValueError("Input text is too short to index.")
        docs = [Document(page_content=text, metadata={"page": 1, "source": source})]
        return self.ingest_documents(docs)

    def ingest_documents(self, docs: list[Document]) -> int:
        self.full_text = "\n\n".join(doc.page_content for doc in docs)
        self.chunks = split_documents(docs)
        if not self.chunks:
            raise ValueError("No chunks were created from the document.")

        self.vector_store = Chroma.from_documents(
            documents=self.chunks,
            embedding=self.embeddings,
            persist_directory=str(self.persist_directory),
            collection_metadata={"hnsw:space": "cosine"},
        )
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": self.retrieval_k})
        return len(self.chunks)

    def retrieve(self, query: str) -> list[Document]:
        if self.retriever is None:
            raise RuntimeError("Ingest a document before querying.")
        return list(self.retriever.invoke(query))

    def answer_query(self, query: str) -> Answer:
        start = time.perf_counter()
        docs = self.retrieve(query)
        context = self._format_docs(docs)
        prompt = ChatPromptTemplate.from_template(
            """You are a source-grounded research assistant.
Answer only from the retrieved paper context. If the context does not contain the answer, say that clearly.
Include concise page citations in the form [p. N].

Context:
{context}

Question: {question}

Answer:"""
        )
        rendered_prompt = prompt.format(context=context, question=query)
        text = self._invoke_llm(rendered_prompt)
        latency_ms = (time.perf_counter() - start) * 1000
        return Answer(text=text, citations=self._citations(docs), latency_ms=latency_ms)

    def generate_summary(self) -> str:
        if not self.full_text:
            raise RuntimeError("Ingest a document before summarizing.")
        prompt = f"""Summarize this research paper with: objective, method, experiments, results, limitations.
Use compact bullet points and avoid unsupported claims.

Paper:
{self.full_text[:12000]}"""
        return self._invoke_llm(prompt)

    def generate_mindmap(self, topic: str | None = None) -> str:
        context = self._diagram_context(topic)
        prompt = f"""Generate Mermaid mindmap code for the research paper context.
Return only raw Mermaid code. Use this format:
mindmap
  root((Paper))
    Method
      Step

Keep labels short and ASCII-only.

Context:
{context}"""
        return clean_mermaid(self._invoke_llm(prompt))

    def generate_flowchart(self, topic: str | None = None) -> str:
        context = self._diagram_context(topic)
        prompt = f"""Generate a top-down Mermaid flowchart for the paper methodology.
Return only raw Mermaid code. Use graph TD and quote every node label, for example:
graph TD
    A["Problem"] --> B["Method"]

Keep labels short and ASCII-only.

Context:
{context}"""
        return clean_mermaid(self._invoke_llm(prompt))

    def _diagram_context(self, topic: str | None) -> str:
        if topic:
            docs = self.retrieve(topic)
            return self._format_docs(docs)
        if not self.full_text:
            raise RuntimeError("Ingest a document before generating diagrams.")
        return self.full_text[:12000]

    @staticmethod
    def _format_docs(docs: list[Document]) -> str:
        if not docs:
            return "No relevant context retrieved."
        blocks = []
        for doc in docs:
            page = doc.metadata.get("page", "?")
            chunk = doc.metadata.get("chunk", "?")
            blocks.append(f"[page {page}, chunk {chunk}]\n{doc.page_content}")
        return "\n\n---\n\n".join(blocks)

    def _invoke_llm(self, prompt: str) -> str:
        response = self.llm.invoke(prompt)
        return getattr(response, "content", response)

    @staticmethod
    def _citations(docs: list[Document]) -> list[Citation]:
        citations: list[Citation] = []
        for doc in docs:
            preview = re.sub(r"\s+", " ", doc.page_content).strip()[:180]
            citations.append(
                Citation(
                    page=int(doc.metadata.get("page", 0) or 0),
                    chunk=int(doc.metadata.get("chunk", 0) or 0),
                    preview=preview,
                )
            )
        return citations
