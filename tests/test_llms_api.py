"""``/api/llms`` 并发探测与 TTL 缓存（不依赖 Milvus / lifespan）。"""
from __future__ import annotations

import time

from fastapi.testclient import TestClient

import app.main as main


def _install_state(monkeypatch, *, ttl: float = 60.0) -> None:
    monkeypatch.setattr(main.app.state, "api_key_env", "", raising=False)
    monkeypatch.setattr(main.app.state, "rate_limit_per_min", 0, raising=False)
    monkeypatch.setattr(main.app.state, "llm_probe_ttl_s", ttl, raising=False)
    monkeypatch.setattr(main.app.state, "llm_probe_cache", None, raising=False)
    monkeypatch.setattr(
        main.app.state,
        "llms",
        {
            "cloud": {
                "client": object(),
                "label": "云端",
                "model": "m1",
                "base_url": "http://cloud.test/v1",
            },
            "ft": {
                "client": object(),
                "label": "本地",
                "model": "m2",
                "base_url": "http://local.test:8009/v1",
            },
        },
        raising=False,
    )


def test_llms_reports_only_2xx_as_available(monkeypatch):
    calls: list[str] = []

    async def fake_probe(base_url: str, client, headers=None) -> bool:
        calls.append(base_url)
        return base_url.startswith("http://local.test")

    monkeypatch.setattr(main, "_probe_available", fake_probe)
    _install_state(monkeypatch)
    client = TestClient(main.app)

    resp = client.get("/api/llms")

    assert resp.status_code == 200
    body = {item["id"]: item for item in resp.json()}
    assert body["cloud"]["available"] is False
    assert body["ft"]["available"] is True
    assert body["cloud"]["model"] == "m1"
    assert sorted(calls) == ["http://cloud.test/v1", "http://local.test:8009/v1"]


def test_llms_caches_within_ttl(monkeypatch):
    calls: list[str] = []

    async def fake_probe(base_url: str, client, headers=None) -> bool:
        calls.append(base_url)
        return True

    monkeypatch.setattr(main, "_probe_available", fake_probe)
    _install_state(monkeypatch, ttl=60.0)
    client = TestClient(main.app)

    assert client.get("/api/llms").status_code == 200
    assert len(calls) == 2
    assert client.get("/api/llms").status_code == 200
    assert len(calls) == 2  # TTL 内命中缓存，不再探测


def test_llms_reprobes_after_ttl(monkeypatch):
    calls: list[str] = []

    async def fake_probe(base_url: str, client, headers=None) -> bool:
        calls.append(base_url)
        return True

    monkeypatch.setattr(main, "_probe_available", fake_probe)
    _install_state(monkeypatch, ttl=60.0)
    client = TestClient(main.app)

    assert client.get("/api/llms").status_code == 200
    assert len(calls) == 2
    # 手动把缓存时间戳推到 TTL 之外
    cached = main.app.state.llm_probe_cache
    main.app.state.llm_probe_cache = (time.monotonic() - 120.0, cached[1])
    assert client.get("/api/llms").status_code == 200
    assert len(calls) == 4


def test_probe_available_rejects_non_2xx(monkeypatch):
    class FakeResponse:
        def __init__(self, status_code: int):
            self.status_code = status_code

    class FakeClient:
        def __init__(self, status_code: int):
            self.status_code = status_code

        async def get(self, url: str, headers=None):
            return FakeResponse(self.status_code)

    async def run(status_code: int) -> bool:
        return await main._probe_available("http://x/v1", FakeClient(status_code))

    import asyncio

    assert asyncio.run(run(200)) is True
    assert asyncio.run(run(204)) is True
    assert asyncio.run(run(401)) is False
    assert asyncio.run(run(404)) is False


def test_probe_available_sends_auth_header():
    class FakeResponse:
        status_code = 200

    class FakeClient:
        def __init__(self):
            self.headers = None

        async def get(self, url: str, headers=None):
            self.headers = headers
            return FakeResponse()

    import asyncio

    client = FakeClient()
    ok = asyncio.run(
        main._probe_available(
            "http://x/v1", client, {"Authorization": "Bearer k"}
        )
    )
    assert ok is True
    assert client.headers == {"Authorization": "Bearer k"}
