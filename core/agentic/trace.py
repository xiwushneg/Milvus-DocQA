"""全链路轨迹记录：节点事件、brain/answer 分账、token 与延迟统计。"""
from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from core.agentic.state import RoleUsage, Status, Usage, new_trace_id


def estimate_tokens(text: str) -> int:
    """粗略估算 token：中文约 1.6 字符/token；仅用于服务端未返回 usage 时。"""
    if not text:
        return 0
    return max(1, int(len(text) / 1.6))


class TraceRecorder:
    def __init__(
        self,
        trace_id: str | None = None,
        session_id: str = "",
        enabled: bool = True,
        on_event: Callable[[dict], None] | None = None,
    ):
        self.trace_id = trace_id or new_trace_id()
        self.session_id = session_id
        self.enabled = enabled
        self.on_event = on_event
        self.events: list[dict] = []
        self.usage = Usage()
        self.started_at = time.perf_counter()
        self.status = Status.RUNNING.value
        self.stop_reason = ""
        self.total_ms = 0

    def emit(
        self,
        stage: str,
        *,
        status: str = "ok",
        round: int | None = None,
        model_role: str | None = None,
        latency_ms: int | None = None,
        payload: dict[str, Any] | None = None,
    ) -> dict:
        event = {
            "type": "trace",
            "trace_id": self.trace_id,
            "stage": stage,
            "status": status,
            "round": round,
            "model_role": model_role,
            "latency_ms": latency_ms,
            "payload": payload or {},
            "ts": time.time(),
        }
        if self.enabled:
            self.events.append(event)
        # 流式接口依赖事件回调，即使不落盘也照常推送
        if self.on_event is not None:
            try:
                self.on_event(dict(event))
            except Exception:
                pass
        return event

    def record_llm(
        self,
        role: str,
        messages: list[dict],
        completion: str,
        latency_ms: int,
        *,
        usage: dict | None = None,
    ) -> None:
        """记录一次 LLM 调用；role 为 brain/answer，用量缺失时按字符估算。"""
        prompt_text = "\n".join(str(m.get("content", "")) for m in messages)
        if usage:
            prompt_tokens = int(usage.get("prompt_tokens", 0))
            completion_tokens = int(usage.get("completion_tokens", 0))
            estimated = False
        else:
            prompt_tokens = estimate_tokens(prompt_text)
            completion_tokens = estimate_tokens(completion or "")
            estimated = True

        bucket: RoleUsage = self.usage.brain if role == "brain" else self.usage.answer
        bucket.add(prompt_tokens, completion_tokens, latency_ms, estimated)
        self.emit(
            "llm",
            model_role=role,
            latency_ms=latency_ms,
            payload={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "estimated": estimated,
            },
        )

    def record_tool(self, name: str, latency_ms: int, ok: bool, payload: dict | None = None) -> None:
        self.usage.tool_calls += 1
        self.emit(
            "tool",
            status="ok" if ok else "error",
            latency_ms=latency_ms,
            payload={"name": name, **(payload or {})},
        )

    def finalize(self, status: str | Status, stop_reason: str = "") -> None:
        self.status = status.value if isinstance(status, Status) else str(status)
        self.stop_reason = stop_reason or ""
        self.total_ms = int((time.perf_counter() - self.started_at) * 1000)
        self.emit(
            "finalize",
            status=self.status,
            payload={"stop_reason": self.stop_reason, "total_ms": self.total_ms},
        )

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "status": self.status,
            "stop_reason": self.stop_reason,
            "total_ms": self.total_ms,
            "usage": self.usage.to_dict(),
            "events": self.events,
        }

    def write(self, directory: str | Path) -> str:
        """原子写入 data/traces/{trace_id}.json，返回文件路径。"""
        base = Path(directory)
        base.mkdir(parents=True, exist_ok=True)
        target = base / f"{self.trace_id}.json"
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, target)
        return str(target)


def list_traces(directory: str | Path, limit: int = 20) -> list[dict]:
    """按修改时间倒序列出轨迹摘要，供回放面板选择。"""
    base = Path(directory)
    if not base.is_dir():
        return []
    files = sorted(
        base.glob("*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )[: max(1, min(limit, 200))]

    out: list[dict] = []
    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        out.append(
            {
                "trace_id": data.get("trace_id") or path.stem,
                "session_id": data.get("session_id", ""),
                "status": data.get("status", ""),
                "stop_reason": data.get("stop_reason", ""),
                "total_ms": data.get("total_ms", 0),
                "usage": data.get("usage", {}),
                "events": len(data.get("events") or []),
                "mtime": path.stat().st_mtime,
            }
        )
    return out
