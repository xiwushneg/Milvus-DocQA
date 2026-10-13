"""端到端冒烟测试：MockEmbedder + Milvus Standalone + StubLLM，
验证 入库 → 混合检索 → 融合 → 问答 全链路与文档删除。

前置条件：docker compose up -d（Milvus 运行在 127.0.0.1:19530）。
Milvus 未启动时本模块自动跳过（skipif），单元测试不受影响。
"""
import os
import socket
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

from core.embedder import MockEmbedder  # noqa: E402
from core.pipeline import RagPipeline  # noqa: E402

MILVUS_URI = os.environ.get("MILVUS_URI", "http://127.0.0.1:19530")

DEMO_TEXT = (
    "# Milvus 入门\n\nMilvus 是一款开源向量数据库，支持 HNSW 索引，"
    "轻量版 Milvus Lite 可通过 pip 安装。\n\n"
    "# RAG 原理\n\nRAG 结合检索与生成，混合检索使用 RRF 融合，常数 k 取 60。\n\n"
    "# 重排序\n\n交叉编码器重排序可以提升答案质量，典型做法是 Top50 精选 Top5。\n"
)


class StubLLM:
    def __init__(self):
        self.calls = 0

    def chat(self, messages):
        self.calls += 1
        return f"StubAnswer#{self.calls}: " + messages[-1]["content"][:40]


def make_cfg(collection):
    return {
        "milvus": {
            "uri": MILVUS_URI,
            "collection": collection,
            "dense_dim": 256,
        },
        # chunk_size 取小值让语料切成 ~5 块：BM25 的 IDF 需要足够大的语料
        # （查询词仅出现在少数块中）才能产生正分，这与真实语料一致。
        "chunking": {"chunk_size": 60, "chunk_overlap": 15, "min_chunk_size": 10},
        "retrieval": {
            "dense_top_k": 5,
            "bm25_top_k": 5,
            "rrf_k": 60,
            "final_top_k": 3,
            "rerank": {"enabled": False},
        },
        "embedding": {"provider": "mock"},
        "llm": {"base_url": "http://stub", "model": "stub"},
    }


def _milvus_reachable() -> bool:
    try:
        with socket.create_connection(("127.0.0.1", 19530), timeout=1):
            return True
    except OSError:
        return False


requires_milvus = pytest.mark.skipif(
    not _milvus_reachable(), reason="Milvus 未启动（docker compose up -d 后执行）"
)


@pytest.fixture()
def pipeline():
    # 每轮测试使用独立 collection，互不干扰，测完即删
    collection = f"smoke_{uuid.uuid4().hex[:8]}"
    p = RagPipeline(make_cfg(collection), embedder=MockEmbedder(dim=256), llm=StubLLM())
    yield p
    p.store.drop()


@requires_milvus
def test_ingest_and_ask(pipeline):
    stats = pipeline.ingest_bytes("demo.md", DEMO_TEXT.encode("utf-8"))
    assert stats["chunks"] >= 2
    assert pipeline.store.count() >= 2

    out = pipeline.ask("Milvus 支持什么索引？")
    assert out["answer"].startswith("StubAnswer")
    assert out["references"], "必须返回参考来源"
    assert out["references"][0]["source"] == "demo.md"
    assert all(k in out["timings"] for k in ("embed_ms", "dense_ms", "bm25_ms", "llm_ms"))


@requires_milvus
def test_bm25_keyword_hit(pipeline):
    """专有名词类查询应能通过 BM25 通道命中。"""
    pipeline.ingest_bytes("demo.md", DEMO_TEXT.encode("utf-8"))
    hits = pipeline.bm25.search("HNSW 索引", top_k=3)
    assert hits, "BM25 应命中 HNSW 相关块"


@requires_milvus
def test_delete_source(pipeline):
    pipeline.ingest_bytes("demo.md", DEMO_TEXT.encode("utf-8"))
    deleted = pipeline.store.delete_by_source("demo.md")
    assert deleted >= 2
    assert pipeline.store.count() == 0
