"""FastAPI 服务入口：文档入库 / 检索问答 / 文档管理。

启动: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
交互文档: http://localhost:8000/docs
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager

import httpx
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.config import load_config
from app.schemas import ChatRequest, ChatResponse, DocStats, IngestResponse
from core.llm import LLMClient
from core.pipeline import RagPipeline, build_pipeline


def build_llm_profiles(cfg: dict) -> dict[str, dict]:
    """按 config.llm.profiles 构建可切换模型档案: id → {client, label, model, base_url}。

    cloud 固定为主配置；其余档案覆盖 base_url/model，密钥/温度等继承主配置。
    """
    base = dict(cfg.get("llm", {}))
    profiles = base.pop("profiles", None) or {}
    out = {
        "cloud": {
            "client": LLMClient(base),
            "label": f"云端 API · {base.get('model', '')}",
            "model": base.get("model", ""),
            "base_url": base.get("base_url", ""),
        }
    }
    for pid, prof in profiles.items():
        merged = {**base, **prof}
        out[pid] = {
            "client": LLMClient(merged),
            "label": prof.get("label") or merged.get("model", pid),
            "model": merged.get("model", ""),
            "base_url": merged.get("base_url", ""),
        }
    return out


def _probe_available(base_url: str) -> bool:
    """1.5s 快速探测服务是否可达（收到任何 HTTP 响应即视为在线）。"""
    if not base_url:
        return False
    try:
        httpx.get(f"{base_url.rstrip('/')}/models", timeout=1.5)
        return True
    except Exception:
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    app.state.pipeline = build_pipeline(cfg)
    app.state.llms = build_llm_profiles(cfg)
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


@app.get("/api/llms")
def list_llms(request: Request):
    """列出可切换的生成模型档案及在线状态（供前端切换器使用）。"""
    return [
        {
            "id": pid,
            "label": info["label"],
            "model": info["model"],
            "available": _probe_available(info["base_url"]),
        }
        for pid, info in request.app.state.llms.items()
    ]


@app.post("/api/chat", response_model=ChatResponse)
def chat(
    req: ChatRequest, request: Request, pipeline: RagPipeline = Depends(get_pipeline)
):
    llms = request.app.state.llms
    chosen_id = req.llm if req.llm in llms else "cloud"
    try:
        out = pipeline.ask(
            req.question,
            top_k=req.top_k,
            llm=llms[chosen_id]["client"],
            history=[m.model_dump() for m in req.history],
        )
    except Exception as exc:  # 网络/向量库瞬时故障 → 503 而非裸 500
        raise HTTPException(
            status_code=503,
            detail=f"问答服务暂时不可用（多为网络波动，请稍后重试）: {exc}",
        ) from exc
    return ChatResponse(llm=chosen_id, **out)


@app.post("/api/chat/stream")
async def chat_stream(
    req: ChatRequest, request: Request, pipeline: RagPipeline = Depends(get_pipeline)
):
    """SSE 流式问答：rewrite? → refs（检索+引用）→ delta*（逐段生成）→ done（耗时）。

    异步端点：检索（短阻塞）放线程池，生成阶段异步流式转发——客户端断连时
    asyncio 取消可干净关闭 LLM 连接，不会遗留阻塞线程拖垮服务。
    """
    import time

    from starlette.concurrency import run_in_threadpool

    from core.llm import build_messages

    llms = request.app.state.llms
    chosen_id = req.llm if req.llm in llms else "cloud"

    async def event_gen():
        t0 = time.perf_counter()
        try:
            prepared = await run_in_threadpool(
                pipeline.prepare,
                req.question,
                req.top_k,
                llms["cloud"]["client"],
                [m.model_dump() for m in req.history],
            )
            rewritten = prepared["rewritten"]
            if rewritten != req.question:
                yield f"data: {json.dumps({'type': 'rewrite', 'query': rewritten, 'llm': chosen_id}, ensure_ascii=False)}\n\n"
            yield "data: " + json.dumps(
                {
                    "type": "refs",
                    "references": pipeline._format_refs(prepared["contexts"]),
                    "timings": prepared["timings"],
                    "llm": chosen_id,
                },
                ensure_ascii=False,
            ) + "\n\n"
            t1 = time.perf_counter()
            answer = ""
            async for chunk in llms[chosen_id]["client"].astream_chat(
                build_messages(rewritten, prepared["contexts"])
            ):
                answer += chunk
                yield "data: " + json.dumps(
                    {"type": "delta", "text": chunk, "llm": chosen_id}, ensure_ascii=False
                ) + "\n\n"
            timings = dict(prepared["timings"])
            timings["llm_ms"] = int((time.perf_counter() - t1) * 1000)
            yield "data: " + json.dumps(
                {
                    "type": "done",
                    "answer": answer,
                    "timings": timings,
                    "total_ms": int((time.perf_counter() - t0) * 1000),
                    "llm": chosen_id,
                },
                ensure_ascii=False,
            ) + "\n\n"
        except Exception as exc:  # 流已开始，无法改 HTTP 状态码 → 以 error 事件告知
            detail = {"type": "error", "detail": str(exc), "llm": chosen_id}
            yield f"data: {json.dumps(detail, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


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
