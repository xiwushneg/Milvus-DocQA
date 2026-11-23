"""默认关闭隔离：旧接口可用，新接口 409，不构建 Agent。"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.agent_api import router
from core.agentic.runtime import build_agent_runtime


def _client(agent=None) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.state.agent = agent
    return TestClient(app)


def test_build_runtime_returns_none_when_disabled():
    assert build_agent_runtime({"agent": {"enabled": False}}, None, {"cloud": {}}) is None


def test_agent_chat_returns_409_when_disabled():
    response = _client().post("/api/agent/chat", json={"question": "你好"})
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "agent_disabled"


def test_agent_stream_returns_409_when_disabled():
    response = _client().post("/api/agent/chat/stream", json={"question": "你好"})
    assert response.status_code == 409


def test_trace_endpoint_returns_409_when_disabled():
    response = _client().get("/api/agent/traces/tr-test")
    assert response.status_code == 409
