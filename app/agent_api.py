"""Agentic RAG API：默认关闭时返回 409，不影响现有 /api/chat。"""
from __future__ import annotations

import json
import re
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.schemas import (
    AgentChatRequest,
    AgentChatResponse,
    AgentCitation,
    AgentMetrics,
    AgentReference,
)

router = APIRouter(prefix="/api/agent", tags=["agent"])

_TRACE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def get_agent(request: Request):
    agent = getattr(request.app.state, "agent", None)
    if agent is None:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "agent_disabled",
                "message": "Agentic RAG 未启用（config.agent.enabled=false）",
            },
        )
    return agent


def _session_id(req: AgentChatRequest) -> str:
    return req.session_id or f"sess-{uuid.uuid4().hex[:12]}"


def _history(req: AgentChatRequest) -> list[dict]:
    return [m.model_dump() for m in req.history]


def _to_response(out: dict) -> AgentChatResponse:
    return AgentChatResponse(
        answer=out["answer"],
        references=[AgentReference(**r) for r in out.get("references", [])],
        citations=[AgentCitation(**c) for c in out.get("citations", [])],
        conflicts=out.get("conflicts", []),
        tool_results=out.get("tool_results", []),
        confidence=out.get("confidence", 0.0),
        refused=out.get("refused", False),
        refusal_reason=out.get("refusal_reason"),
        session_id=out["session_id"],
        trace_id=out["trace_id"],
        metrics=AgentMetrics(**out["metrics"]),
        llm=out.get("llm", "cloud"),
        trace=out.get("trace"),
    )


@router.post("/chat", response_model=AgentChatResponse)
def agent_chat(req: AgentChatRequest, agent=Depends(get_agent)):
    try:
        out = agent.ask(
            req.question,
            session_id=_session_id(req),
            llm=req.llm,
            top_k=req.top_k,
            history=_history(req),
            include_trace=req.include_trace,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Agent 服务暂时不可用: {exc}",
        ) from exc

    if out.get("status") == "error":
        raise HTTPException(
            status_code=503,
            detail=f"Agent 执行失败: {out.get('error', '')}",
        )
    return _to_response(out)


@router.post("/chat/stream")
async def agent_chat_stream(req: AgentChatRequest, agent=Depends(get_agent)):
    """SSE 流式接口：session → 实时 trace → delta → refs → done。"""
    from starlette.concurrency import iterate_in_threadpool

    session_id = _session_id(req)
    events = agent.ask_stream(
        req.question,
        session_id=session_id,
        llm=req.llm,
        top_k=req.top_k,
        history=_history(req),
    )

    async def event_gen():
        try:
            async for item in iterate_in_threadpool(events):
                if item.get("type") == "done" and item.get("status") == "error":
                    yield _sse(
                        {
                            "type": "error",
                            "detail": item.get("error") or "Agent 执行失败",
                            "session_id": session_id,
                            "llm": req.llm,
                        }
                    )
                    continue
                if item.get("type") == "done":
                    item.setdefault("session_id", session_id)
                    item.setdefault(
                        "total_ms", (item.get("metrics") or {}).get("latency_ms", 0)
                    )
                yield _sse(item)
        except Exception as exc:  # 流已开始，只能以 error 事件告知
            yield _sse(
                {
                    "type": "error",
                    "detail": str(exc),
                    "session_id": session_id,
                    "llm": req.llm,
                }
            )

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/traces")
def list_trace_files(limit: int = 20, agent=Depends(get_agent)):
    """按时间倒序列出可回放的轨迹摘要。"""
    from core.agentic.trace import list_traces

    return {"traces": list_traces(agent.trace_dir, limit=limit)}


@router.get("/traces/{trace_id}")
def get_trace(trace_id: str, agent=Depends(get_agent)):
    if not _TRACE_ID_RE.fullmatch(trace_id):
        raise HTTPException(status_code=404, detail="trace 不存在")
    base = Path(agent.trace_dir).resolve()
    path = (base / f"{trace_id}.json").resolve()
    if base not in path.parents or not path.is_file():
        raise HTTPException(status_code=404, detail="trace 不存在")
    return json.loads(path.read_text(encoding="utf-8"))


def _sse(obj: dict) -> str:
    return "data: " + json.dumps(obj, ensure_ascii=False) + "\n\n"
