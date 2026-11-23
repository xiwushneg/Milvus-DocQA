"""BM25 稀疏检索（jieba 中文分词 + rank_bm25），进程内存索引。

设计取舍：BM25 语料由 Milvus 全量 chunk 在内存重建，实现简单、零额外部署，
适合万级文本块的毕设规模；更大规模可升级为 Milvus 2.5+ 原生 BM25 稀疏向量检索。
"""
from __future__ import annotations

from collections.abc import Callable
from threading import Lock

import jieba
from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    return [t for t in jieba.lcut(text.lower()) if t.strip()]


class BM25Index:
    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._meta: list[dict] = []
        # 重建（入库/删除后）与问答可能并发：build 在锁内一次性替换
        # _meta 与 _bm25，search 在锁内取同一版本快照，避免新旧索引错配。
        self._lock = Lock()

    def build(self, rows: list[dict]) -> None:
        """rows: Milvus 全量 chunk（含 id/text/source/page/chunk_idx）。"""
        meta = list(rows)
        bm25 = BM25Okapi([tokenize(r["text"]) for r in meta]) if meta else None
        with self._lock:
            self._meta = meta
            self._bm25 = bm25

    def search(
        self,
        query: str,
        top_k: int = 20,
        predicate: Callable[[dict], bool] | None = None,
    ) -> list[dict]:
        """BM25 检索；predicate 用于 source/page 过滤。

        默认 predicate=None 时保持原有行为：先取分数前 top_k，再丢弃零分项。
        带过滤时扫描全量排序结果，保证过滤后仍能取满 top_k。
        """
        with self._lock:
            bm25 = self._bm25
            meta = self._meta
        if not bm25:
            return []
        q = tokenize(query)
        if not q:
            return []
        scores = bm25.get_scores(q)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        if predicate is None:
            order = order[:top_k]
        out: list[dict] = []
        for i in order:
            if scores[i] <= 0:
                continue
            h = dict(meta[i])
            if predicate is not None and not predicate(h):
                continue
            h["bm25_score"] = float(scores[i])
            out.append(h)
            if len(out) >= top_k:
                break
        return out
