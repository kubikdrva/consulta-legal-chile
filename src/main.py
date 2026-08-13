from __future__ import annotations

import uuid
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Optional

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.graph import graph
from src.ingest import ingest_directory, ingest_file
from src.tracing import flush_langfuse, get_langfuse_handler

app = FastAPI(title="RAG Agent", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    sources: list
    grounded: bool
    retries: int
    trace_url: Optional[str] = None


class IngestResponse(BaseModel):
    chunks_ingested: int
    message: str


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    session_id = request.session_id or str(uuid.uuid4())

    langfuse_handler = get_langfuse_handler(
        trace_name="rag-query",
        session_id=session_id,
        user_id=request.user_id,
        tags=["rag-agent"],
        metadata={"query": request.question},
    )

    config = {"run_name": "rag-query"}
    if langfuse_handler:
        config["callbacks"] = [langfuse_handler]

    initial_state = {
        "query": request.question,
        "reformulated_query": "",
        "documents": [],
        "answer": "",
        "is_grounded": False,
        "grade_reason": "",
        "retry_count": 0,
        "sources": [],
    }

    result = await graph.ainvoke(initial_state, config=config)

    trace_url = None
    if langfuse_handler:
        trace_url = langfuse_handler.get_trace_url()

    flush_langfuse()

    return QueryResponse(
        answer=result["answer"],
        sources=result.get("sources", []),
        grounded=result["is_grounded"],
        retries=result["retry_count"],
        trace_url=trace_url,
    )


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)):
    suffix = Path(file.filename or "upload.txt").suffix
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        count = await ingest_file(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return IngestResponse(
        chunks_ingested=count,
        message=f"Successfully ingested {file.filename}",
    )


@app.post("/ingest-dir", response_model=IngestResponse)
async def ingest_dir(directory: str):
    count = await ingest_directory(directory)
    return IngestResponse(
        chunks_ingested=count,
        message=f"Successfully ingested directory {directory}",
    )


@app.on_event("shutdown")
async def shutdown_event():
    flush_langfuse()
