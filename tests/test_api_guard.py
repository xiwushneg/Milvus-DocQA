"""可选 API Key 鉴权与按 IP 限流中间件（不依赖 Milvus / lifespan）。"""
from __future__ import annotations

from fastapi.testclient import TestClient

import app.main as main


def make_client() -> TestClient:
    # 未进入 lifespan：受保护接口会因缺少 pipeline 返回 500，
    # 但鉴权/限流在进入端点前即可验证。
    return TestClient(main.app, raise_server_exceptions=False)


def test_health_is_public_even_with_api_key(monkeypatch):
    monkeypatch.setattr(main.app.state, "api_key_env", "DOCQA_TEST_KEY", raising=False)
    monkeypatch.setenv("DOCQA_TEST_KEY", "secret")
    client = make_client()

    assert client.get("/api/health").status_code == 200
    assert client.get("/api/health", headers={"X-API-Key": "wrong"}).status_code == 200


def test_api_requires_configured_key(monkeypatch):
    monkeypatch.setattr(main.app.state, "api_key_env", "DOCQA_TEST_KEY", raising=False)
    monkeypatch.setenv("DOCQA_TEST_KEY", "secret")
    client = make_client()

    assert client.get("/api/documents").status_code == 401
    resp = client.get("/api/documents", headers={"X-API-Key": "secret"})
    assert resp.status_code != 401


def test_api_accepts_bearer_token(monkeypatch):
    monkeypatch.setattr(main.app.state, "api_key_env", "DOCQA_TEST_KEY", raising=False)
    monkeypatch.setenv("DOCQA_TEST_KEY", "secret")
    client = make_client()

    resp = client.get(
        "/api/documents", headers={"Authorization": "Bearer secret"}
    )
    assert resp.status_code != 401


def test_api_key_disabled_when_env_missing(monkeypatch):
    monkeypatch.setattr(main.app.state, "api_key_env", "DOCQA_TEST_KEY", raising=False)
    monkeypatch.delenv("DOCQA_TEST_KEY", raising=False)
    client = make_client()

    assert client.get("/api/documents").status_code != 401


def test_rate_limit_returns_429(monkeypatch):
    main._RATE_HITS.clear()
    monkeypatch.setattr(main.app.state, "api_key_env", "", raising=False)
    monkeypatch.setattr(main.app.state, "rate_limit_per_min", 1, raising=False)
    client = make_client()

    first = client.get("/api/documents")
    second = client.get("/api/documents")

    assert first.status_code != 429
    assert second.status_code == 429
    assert int(second.headers["Retry-After"]) >= 1
    main._RATE_HITS.clear()
