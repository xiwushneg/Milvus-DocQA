"""FastAPI 服务入口：文档入库 / 检索问答 / 文档管理。

启动: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
交互文档: http://localhost:8000/docs
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.config import load_config
from app.schemas import ChatRequest, ChatResponse, DocStats, IngestResponse
from core.pipeline import RagPipeline, build_pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    app.state.pipeline = build_pipeline(cfg)
    yield


app = FastAPI(
    title="Milvus-DocQA 文档检索问答系统",
    description="基于 Milvus 与开源大模型的文档检索问答系统（RAG）",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


def get_pipeline() -> RagPipeline:
    return app.state.pipeline


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/ingest", response_model=IngestResponse)
async def ingest(
    files: list[UploadFile] = File(...),
    pipeline: RagPipeline = Depends(get_pipeline),
):
    if not files:
        raise HTTPException(status_code=400, detail="请上传至少一个文件")
    docs = []
    for f in files:
        data = await f.read()
        if not data:
            continue
        docs.append(pipeline.ingest_bytes(f.filename or "unnamed", data))
    return IngestResponse(documents=docs)


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest, pipeline: RagPipeline = Depends(get_pipeline)):
    try:
        out = pipeline.ask(req.question, top_k=req.top_k)
    except Exception as exc:  # 网络/向量库瞬时故障 → 503 而非裸 500
        raise HTTPException(
            status_code=503,
            detail=f"问答服务暂时不可用（多为网络波动，请稍后重试）: {exc}",
        ) from exc
    return ChatResponse(**out)


@app.get("/api/documents", response_model=DocStats)
def documents(pipeline: RagPipeline = Depends(get_pipeline)):
    return DocStats(
        total_chunks=pipeline.store.count(),
        sources=pipeline.store.list_sources(),
    )


@app.delete("/api/documents")
def delete_document(source: str, pipeline: RagPipeline = Depends(get_pipeline)):
    deleted = pipeline.store.delete_by_source(source)
    pipeline.refresh_bm25()
    if deleted == 0:
        raise HTTPException(status_code=404, detail=f"未找到文档: {source}")
    return {"deleted": deleted}
