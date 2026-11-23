"""Agent API 契约：响应字段、SSE 事件序、trace 读取。"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.agent_api import router


class FakeAgent:
    def __init__(self, trace_dir: Path):
        self.trace_dir = str(trace_dir)

    def ask(self, question: str, **kwargs) -> dict:
        return {
            "answer": "答案[E1]",
            "references": [
                {
                    "evidence_id": "E1",
                    "source": "doc_a.md",
                    "page": 1,
                    "chunk_idx": 0,
                    "snippet": "片段",
                    "score": 0.9,
                    "channels": ["dense", "bm25"],
                }
            ],
            "citations": [{"sentence": "答案[E1]", "evidence_ids": ["E1"]}],
            "conflicts": [],
            "confidence": 0.8,
            "refused": False,
            "refusal_reason": None,
            "session_id": kwargs.get("session_id", ""),
            "trace_id": "tr-test",
            "metrics": {
                "rounds": 1,
                "sub_queries": 1,
                "tool_calls": 0,
                "brain_llm_calls": 0,
                "brain_latency_ms": 0,
                "answer_llm_calls": 1,
                "answer_latency_ms": 120,
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "cost_cny": 0.0,
                "latency_ms": 300,
                "stop_reason": "answered",
            },
            "llm": "ft",
            "status": "answered",
            "error": "",
            "trace": [
                {
                    "type": "trace",
                    "trace_id": "tr-test",
                    "stage": "analyze",
                    "status": "ok",
                    "round": 0,
                    "model_role": None,
                    "latency_ms": 1,
                    "payload": {"intent": "fact"},
                    "ts": 0.0,
                }
            ],
        }

    def ask_stream(self, question: str, **kwargs):
        out = self.ask(question, **kwargs)
        yield {
            "type": "session",
            "session_id": out["session_id"],
            "trace_id": out["trace_id"],
            "llm": out["llm"],
        }
        yield {
            "type": "trace",
            "trace_id": out["trace_id"],
            "stage": "analyze",
            "status": "ok",
            "round": 0,
            "model_role": None,
            "latency_ms": 1,
            "payload": {"intent": "fact"},
            "ts": 0.0,
        }
        yield {"type": "delta", "text": "答案"}
        yield {
            "type": "refs",
            "references": out["references"],
            "citations": out["citations"],
            "conflicts": out["conflicts"],
            "tool_results": out.get("tool_results", []),
            "confidence": out["confidence"],
        }
        yield {"type": "done", **out}


def _client(tmp_path: Path) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.state.agent = FakeAgent(tmp_path)
    return TestClient(app)


def test_agent_chat_contract(tmp_path):
    response = _client(tmp_path).post(
        "/api/agent/chat",
        json={"question": "HNSW 是什么", "llm": "ft", "include_trace": True},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "答案[E1]"
    assert data["references"][0]["evidence_id"] == "E1"
    assert data["metrics"]["rounds"] == 1
    assert data["session_id"].startswith("sess-")
    assert data["trace"][0]["stage"] == "analyze"


def test_agent_stream_event_order(tmp_path):
    with _client(tmp_path).stream(
        "POST", "/api/agent/chat/stream", json={"question": "HNSW 是什么"}
    ) as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())

    events = [
        json.loads(line[5:])
        for line in body.splitlines()
        if line.startswith("data: ")
    ]
    types = [e["type"] for e in events]
    assert types[0] == "session"
    assert "trace" in types
    assert "refs" in types
    assert "delta" in types
    assert types[-1] == "done"
    # 流式下 delta 先于最终 refs，两者都在 done 之前
    assert types.index("delta") < types.index("refs") < types.index("done")

    done = events[-1]
    assert done["confidence"] == 0.8
    assert done["citations"][0]["evidence_ids"] == ["E1"]
    assert done["total_ms"] == 300


class ErrorFakeAgent(FakeAgent):
    def ask_stream(self, question: str, **kwargs):
        yield {
            "type": "session",
            "session_id": kwargs.get("session_id", ""),
            "trace_id": "tr-err",
            "llm": "cloud",
        }
        yield {
            "type": "done",
            "answer": "",
            "status": "error",
            "error": "节点异常",
            "metrics": {"latency_ms": 1},
        }


def test_agent_stream_error_event(tmp_path):
    app = FastAPI()
    app.include_router(router)
    app.state.agent = ErrorFakeAgent(tmp_path)

    with TestClient(app).stream(
        "POST", "/api/agent/chat/stream", json={"question": "HNSW 是什么"}
    ) as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())

    events = [
        json.loads(line[5:])
        for line in body.splitlines()
        if line.startswith("data: ")
    ]
    assert events[-1]["type"] == "error"
    assert events[-1]["detail"] == "节点异常"
    assert all(e["type"] != "done" for e in events)


def test_trace_list_is_newest_first(tmp_path):
    older = {
        "trace_id": "tr-old",
        "session_id": "s1",
        "status": "answered",
        "stop_reason": "answered",
        "total_ms": 10,
        "usage": {},
        "events": [],
    }
    newer = {**older, "trace_id": "tr-new", "total_ms": 20}
    (tmp_path / "tr-old.json").write_text(json.dumps(older), encoding="utf-8")
    (tmp_path / "tr-new.json").write_text(json.dumps(newer), encoding="utf-8")

    response = _client(tmp_path).get("/api/agent/traces?limit=5")
    assert response.status_code == 200
    traces = response.json()["traces"]
    assert [t["trace_id"] for t in traces] == ["tr-new", "tr-old"]
    assert traces[0]["total_ms"] == 20
    assert traces[0]["events"] == 0


def test_trace_file_is_served(tmp_path):
    payload = {"trace_id": "tr-test", "events": [], "usage": {}, "metrics": {}}
    (tmp_path / "tr-test.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    response = _client(tmp_path).get("/api/agent/traces/tr-test")
    assert response.status_code == 200
    assert response.json()["trace_id"] == "tr-test"


def test_invalid_trace_id_is_404(tmp_path):
    response = _client(tmp_path).get("/api/agent/traces/..%2Fsecret")
    assert response.status_code == 404
