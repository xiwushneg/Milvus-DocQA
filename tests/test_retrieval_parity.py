"""检索 parity：默认参数下 RetrievalAdapter 必须与 _retrieve 完全一致。"""
from __future__ import annotations

import re

import pytest

from core.agentic.retrieval import RetrievalAdapter
from core.agentic.state import SubQuery
from core.pipeline import RagPipeline


class FakeEmbedder:
    def embed_query(self, query: str) -> list[float]:
        return [0.1, 0.2, 0.3, 0.4]


class FakeStore:
    def __init__(self):
        self.rows = [
            {
                "id": 1,
                "text": "Milvus 是开源向量数据库，支持 HNSW 索引与稠密检索。",
                "source": "doc_a.md",
                "page": 1,
                "chunk_idx": 0,
            },
            {
                "id": 2,
                "text": "RAG 结合检索与生成，使用 RRF 融合多路召回结果。",
                "source": "doc_a.md",
                "page": 2,
                "chunk_idx": 1,
            },
            {
                "id": 3,
                "text": "交叉编码器重排序可以提升最终上下文的相关性。",
                "source": "doc_b.md",
                "page": 1,
                "chunk_idx": 0,
            },
            {
                "id": 4,
                "text": "BM25 是稀疏检索算法，擅长关键词精确匹配。",
                "source": "doc_b.md",
                "page": 2,
                "chunk_idx": 1,
            },
            {
                "id": 5,
                "text": "Milvus Lite 适合本地开发与快速验证。",
                "source": "doc_c.md",
                "page": 1,
                "chunk_idx": 0,
            },
        ]
        self.last_filter = ""
        self.search_calls = 0

    def all_chunks(self, max_rows: int = 100000, wait_visible: bool = False) -> list[dict]:
        return [dict(r) for r in self.rows]

    def search_dense(
        self, query_vec: list[float], top_k: int = 20, filter_expr: str = ""
    ) -> list[dict]:
        self.search_calls += 1
        self.last_filter = filter_expr
        rows = list(self.rows)
        match = re.search(r'source == "(.*?)"', filter_expr)
        if match:
            rows = [r for r in rows if r["source"] == match.group(1)]
        match = re.search(r"page == (\d+)", filter_expr)
        if match:
            page = int(match.group(1))
            rows = [r for r in rows if r["page"] == page]
        hits = []
        for i, row in enumerate(rows):
            hit = dict(row)
            hit["dense_score"] = round(1.0 - i * 0.15, 6)
            hits.append(hit)
        return hits[:top_k]


class FakeReranker:
    enabled = False

    def rerank(self, query: str, hits: list[dict], top_n: int = 5) -> list[dict]:
        return hits[:top_n]


class FakeLLM:
    def chat(self, messages: list[dict]) -> str:
        return "stub"


def _pipeline() -> tuple[RagPipeline, FakeStore]:
    store = FakeStore()
    cfg = {
        "retrieval": {
            "dense_top_k": 5,
            "bm25_top_k": 5,
            "rrf_k": 60,
            "final_top_k": 3,
            "rerank": {"enabled": False},
        }
    }
    pipeline = RagPipeline(
        cfg,
        store=store,
        embedder=FakeEmbedder(),
        llm=FakeLLM(),
        reranker=FakeReranker(),
    )
    return pipeline, store


def test_default_adapter_matches_retrieve():
    pipeline, _ = _pipeline()
    baseline, _ = pipeline._retrieve("HNSW 索引", top_k=3)
    adapter = RetrievalAdapter.from_pipeline(pipeline)
    adapted, _ = adapter.search(SubQuery(text="HNSW 索引"), top_k=3)

    assert [h["id"] for h in adapted] == [h["id"] for h in baseline]
    for got, expected in zip(adapted, baseline):
        assert got["rrf_score"] == pytest.approx(expected["rrf_score"])
        assert got["channels"] == expected["channels"]


def test_source_page_filter_uses_existing_fields_only():
    pipeline, store = _pipeline()
    adapter = RetrievalAdapter.from_pipeline(pipeline)
    hits, _ = adapter.search(
        SubQuery(text="HNSW 索引", source="doc_a.md", page=2),
        top_k=3,
    )

    assert hits
    assert all(h["source"] == "doc_a.md" and h["page"] == 2 for h in hits)
    assert store.last_filter == 'source == "doc_a.md" and page == 2'
