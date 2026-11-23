"""FastAPI 服务入口：文档入库 / 检索问答 / 文档管理。

启动: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
交互文档: http://localhost:8000/docs
"""
from __future__ import annotations

import asyncio
import hmac
import json
import os
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock

import httpx
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from app.agent_api import router as agent_router
from app.config import load_config
from app.schemas import ChatRequest, ChatResponse, DocStats, IngestResponse
from core.loader import SUPPORTED_SUFFIXES
from core.llm import LLMClient
from core.pipeline import RagPipeline, build_pipeline


_BOOT_SERVER = load_config().get("server") or {}
_CORS_ORIGINS = list(_BOOT_SERVER.get("cors_origins") or ["*"])
_RATE_LOCK = Lock()
_RATE_HITS: dict[str, deque] = defaultdict(deque)


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


async def _probe_available(
    base_url: str, client: httpx.AsyncClient, headers: dict | None = None
) -> bool:
    """1.5s 异步探测；仅 2xx 视为可用（401/404 不算在线）。"""
    if not base_url:
        return False
    try:
        resp = await client.get(
            f"{base_url.rstrip('/')}/models", headers=headers or {}
        )
        return 200 <= resp.status_code < 300
    except Exception:
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    server_cfg = cfg.get("server") or {}
    app.state.api_key_env = str(server_cfg.get("api_key_env") or "").strip()
    app.state.rate_limit_per_min = int(server_cfg.get("rate_limit_per_min", 0) or 0)
    app.state.llm_probe_ttl_s = float(server_cfg.get("llm_probe_ttl_s", 30) or 0)
    app.state.llm_probe_cache = None
    app.state.pipeline = build_pipeline(cfg)
    app.state.llms = build_llm_profiles(cfg)
    # Agent 默认关闭：关闭时不导入/构建任何 Agent 组件，旧链路零开销
    app.state.agent = None
    if (cfg.get("agent") or {}).get("enabled", False):
        from core.agentic.runtime import build_agent_runtime

        app.state.agent = build_agent_runtime(cfg, app.state.pipeline, app.state.llms)
    yield
    # 关闭连接池，避免 uvicorn 退出时遗留未释放的 HTTP 连接
    for info in getattr(app.state, "llms", {}).values():
        closer = getattr(info.get("client"), "aclose", None)
        if callable(closer):
            try:
                await closer()
            except Exception:
                pass
    pipeline = getattr(app.state, "pipeline", None)
    if pipeline is not None:
        pipeline.close()


app = FastAPI(
    title="Milvus-DocQA 文档检索问答系统",
    description="基于 Milvus 与开源大模型的文档检索问答系统（RAG）",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def api_guard(request: Request, call_next):
    """可选 API Key 鉴权 + 按 IP 限流；仅作用于 /api/*（health 放行）。"""
    path = request.url.path
    if not path.startswith("/api/") or path == "/api/health":
        return await call_next(request)

    key_env = getattr(app.state, "api_key_env", "")
    expected = os.environ.get(key_env, "") if key_env else ""
    if expected:
        provided = request.headers.get("X-API-Key", "")
        if not provided:
            auth = request.headers.get("Authorization", "")
            if auth.lower().startswith("bearer "):
                provided = auth[7:].strip()
        if not hmac.compare_digest(provided, expected):
            return JSONResponse({"detail": "unauthorized"}, status_code=401)

    limit = int(getattr(app.state, "rate_limit_per_min", 0) or 0)
    if limit > 0:
        client = request.client.host if request.client else "unknown"
        now = time.monotonic()
        with _RATE_LOCK:
            bucket = _RATE_HITS[client]
            while bucket and now - bucket[0] > 60:
                bucket.popleft()
            if len(bucket) >= limit:
                retry_after = max(1, int(60 - (now - bucket[0])))
                return JSONResponse(
                    {"detail": "rate limit exceeded"},
                    status_code=429,
                    headers={"Retry-After": str(retry_after)},
                )
            bucket.append(now)
    return await call_next(request)


app.include_router(agent_router)


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
    skipped = []
    ingested = False
    try:
        for f in files:
            name = f.filename or "unnamed"
            data = await f.read()
            if not data:
                skipped.append({"source": name, "reason": "文件为空"})
                continue
            suffix = Path(name).suffix.lower()
            if suffix not in SUPPORTED_SUFFIXES:
                raise HTTPException(
                    status_code=415,
                    detail=f"不支持的文件类型: {name}（支持 {sorted(SUPPORTED_SUFFIXES)}）",
                )
            try:
                # 批量入库先跳过 BM25 重建，全部写完只重建一次
                stats = pipeline.ingest_bytes(name, data, rebuild_bm25=False)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=f"无法解析 {name}: {exc}") from exc
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"入库失败 {name}: {exc}") from exc
            if stats.get("skipped"):
                skipped.append(
                    {
                        "source": name,
                        "reason": "内容重复（sha256 相同）",
                        "duplicate_of": stats.get("duplicate_of", ""),
                    }
                )
                continue
            docs.append(stats)
            ingested = True
    finally:
        if ingested:
            pipeline.refresh_bm25()
    if not docs:
        raise HTTPException(
            status_code=400,
            detail={"message": "没有可入库的文件", "skipped": skipped},
        )
    return IngestResponse(documents=docs, skipped=skipped)


@app.get("/api/llms")
async def list_llms(request: Request):
    """列出可切换的生成模型档案及在线状态（并发探测 + TTL 缓存）。"""
    llms = request.app.state.llms
    now = time.monotonic()
    ttl = float(getattr(request.app.state, "llm_probe_ttl_s", 30.0))
    cached = getattr(request.app.state, "llm_probe_cache", None)
    if ttl > 0 and cached and now - cached[0] < ttl:
        availability = cached[1]
    else:
        async with httpx.AsyncClient(timeout=1.5) as client:
            values = await asyncio.gather(
                *[
                    _probe_available(
                        info["base_url"],
                        client,
                        getattr(info["client"], "_headers", {}),
                    )
                    for info in llms.values()
                ]
            )
        availability = dict(zip(llms.keys(), values))
        request.app.state.llm_probe_cache = (now, availability)
    return [
        {
            "id": pid,
            "label": info["label"],
            "model": info["model"],
            "available": availability.get(pid, False),
        }
        for pid, info in llms.items()
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
    deleted = pipeline.forget_source(source)
    if deleted == 0:
        raise HTTPException(status_code=404, detail=f"未找到文档: {source}")
    return {"deleted": deleted}
