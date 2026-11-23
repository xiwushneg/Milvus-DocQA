"""Agentic RAG 检索适配器。

只组合现有检索组件，不改 RagPipeline._retrieve：
MilvusStore.search_dense → BM25Index.search → rrf_fuse → Reranker.rerank。
默认参数下与 _retrieve 完全等价（见 tests/test_retrieval_parity.py）。
"""
from __future__ import annotations

import time
from typing import Callable

from core.agentic.state import SubQuery
from core.fusion import rrf_fuse


def _bm25_predicate(sub_query: SubQuery) -> Callable[[dict], bool] | None:
    if sub_query.source is None and sub_query.page is None:
        return None

    def ok(meta: dict) -> bool:
        if sub_query.source is not None and meta.get("source") != sub_query.source:
            return False
        if sub_query.page is not None and int(meta.get("page", -1)) != sub_query.page:
            return False
        return True

    return ok


class RetrievalAdapter:
    def __init__(
        self,
        store,
        bm25,
        embedder,
        reranker,
        *,
        rrf_k: int = 60,
        dense_top_k: int = 20,
        bm25_top_k: int = 20,
        ensure_bm25: Callable[[], None] | None = None,
    ):
        self.store = store
        self.bm25 = bm25
        self.embedder = embedder
        self.reranker = reranker
        self.rrf_k = rrf_k
        self.dense_top_k = dense_top_k
        self.bm25_top_k = bm25_top_k
        self._ensure_bm25 = ensure_bm25

    @classmethod
    def from_pipeline(cls, pipeline) -> "RetrievalAdapter":
        rc = pipeline.cfg.get("retrieval", {})
        return cls(
            pipeline.store,
            pipeline.bm25,
            pipeline.embedder,
            pipeline.reranker,
            rrf_k=int(rc.get("rrf_k", 60)),
            dense_top_k=int(rc.get("dense_top_k", 20)),
            bm25_top_k=int(rc.get("bm25_top_k", 20)),
            ensure_bm25=getattr(pipeline, "_ensure_bm25", None),
        )

    def search(self, sub_query: SubQuery, top_k: int = 5) -> tuple[list[dict], dict]:
        if top_k <= 0:
            return [], {}

        if self._ensure_bm25:
            self._ensure_bm25()

        channels = set(sub_query.channels)
        t0 = time.perf_counter()
        dense_hits: list[dict] = []
        if "dense" in channels:
            # HyDE：假设答案只影响 dense，BM25 仍用原始查询词
            qvec = self.embedder.embed_query(sub_query.dense_query)
            t1 = time.perf_counter()
            dense_hits = self.store.search_dense(
                qvec,
                top_k=self.dense_top_k,
                filter_expr=sub_query.filter_expr,
            )
        else:
            t1 = time.perf_counter()
        t2 = time.perf_counter()

        bm25_hits: list[dict] = []
        if "bm25" in channels:
            bm25_hits = self.bm25.search(
                sub_query.text,
                top_k=self.bm25_top_k,
                predicate=_bm25_predicate(sub_query),
            )
        t3 = time.perf_counter()

        # 固定按 dense/bm25 顺序传入，保证 rrf_fuse 的通道命名稳定
        fused = rrf_fuse([dense_hits, bm25_hits], k=self.rrf_k)
        t4 = time.perf_counter()

        if self.reranker is not None and getattr(self.reranker, "enabled", False):
            contexts = self.reranker.rerank(sub_query.text, fused, top_n=top_k)
        else:
            contexts = fused[:top_k]
        t5 = time.perf_counter()

        timings = {
            "embed_ms": int((t1 - t0) * 1000),
            "dense_ms": int((t2 - t1) * 1000),
            "bm25_ms": int((t3 - t2) * 1000),
            "fuse_ms": int((t4 - t3) * 1000),
            "rerank_ms": int((t5 - t4) * 1000),
            "dense_hits": len(dense_hits),
            "bm25_hits": len(bm25_hits),
            "fused_hits": len(fused),
        }
        return contexts, timings
