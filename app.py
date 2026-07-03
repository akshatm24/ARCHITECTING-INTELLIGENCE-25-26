from __future__ import annotations

import os
import tempfile
from typing import TYPE_CHECKING
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from ai_research_assistant import RAGPipeline


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3)


class DiagramRequest(BaseModel):
    topic: str | None = None
    diagram_type: str = Field("flowchart", pattern="^(flowchart|mindmap)$")


class IngestTextRequest(BaseModel):
    text: str = Field(..., min_length=100)
    source: str = "api"


app = FastAPI(
    title="AI Research Assistant API",
    version="1.0.0",
    description="Hybrid ChromaDB + BM25 RAG API for research-paper question answering.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag: "RAGPipeline | None" = None


@app.get("/")
async def root() -> dict[str, str]:
    return {
        "service": "AI Research Assistant API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "ready": rag is not None,
        "model": os.getenv("GOOGLE_MODEL", "gemini-1.5-flash"),
    }


@app.post("/ingest/pdf")
async def ingest_pdf(file: UploadFile = File(...)) -> dict[str, Any]:
    if file.content_type not in {"application/pdf", "application/octet-stream"}:
        raise HTTPException(status_code=400, detail="Upload a PDF file.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        pipeline = _new_pipeline()
        chunk_count = await run_in_threadpool(pipeline.ingest_pdf, tmp_path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        os.unlink(tmp_path)

    _set_pipeline(pipeline)
    return {"indexed_chunks": chunk_count, "retrieval": "hybrid_chromadb_bm25"}


@app.post("/ingest/text")
async def ingest_text(payload: IngestTextRequest) -> dict[str, Any]:
    try:
        pipeline = _new_pipeline()
        chunk_count = await run_in_threadpool(pipeline.ingest_text, payload.text, payload.source)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    _set_pipeline(pipeline)
    return {"indexed_chunks": chunk_count, "retrieval": "hybrid_chromadb_bm25"}


@app.post("/query")
async def query(payload: QueryRequest) -> dict[str, Any]:
    pipeline = _get_pipeline()
    answer = await run_in_threadpool(pipeline.answer_query, payload.question)
    return {
        "answer": answer.text,
        "citations": [citation.__dict__ for citation in answer.citations],
        "latency_ms": round(answer.latency_ms, 2),
    }


@app.post("/diagram")
async def diagram(payload: DiagramRequest) -> dict[str, str]:
    pipeline = _get_pipeline()
    if payload.diagram_type == "mindmap":
        code = await run_in_threadpool(pipeline.generate_mindmap, payload.topic)
    else:
        code = await run_in_threadpool(pipeline.generate_flowchart, payload.topic)
    return {"mermaid": code}


@app.post("/summary")
async def summary() -> dict[str, str]:
    pipeline = _get_pipeline()
    text = await run_in_threadpool(pipeline.generate_summary)
    return {"summary": text}


def _new_pipeline() -> "RAGPipeline":
    from ai_research_assistant import RAGPipeline

    return RAGPipeline()


def _set_pipeline(pipeline: "RAGPipeline") -> None:
    global rag
    rag = pipeline


def _get_pipeline() -> "RAGPipeline":
    if rag is None:
        raise HTTPException(status_code=409, detail="Ingest a PDF or text document before querying.")
    return rag
