"""HTTP 客户端关闭：可重复调用，且不依赖是否有异步连接池。"""
from __future__ import annotations

import asyncio

from core.embedder import ApiEmbedder
from core.llm import LLMClient


def test_llm_client_close_is_idempotent():
    client = LLMClient({"base_url": "http://127.0.0.1:9", "model": "stub"})
    client.close()
    client.close()
    asyncio.run(client.aclose())


def test_api_embedder_close_is_idempotent():
    embedder = ApiEmbedder(model="m", api_base="http://127.0.0.1:9", api_key="k")
    embedder.close()
    embedder.close()
    assert embedder.name == "api:m"
