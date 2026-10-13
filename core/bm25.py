"""BM25 稀疏检索（jieba 中文分词 + rank_bm25），进程内存索引。

设计取舍：BM25 语料由 Milvus 全量 chunk 在内存重建，实现简单、零额外部署，
适合万级文本块的毕设规模；更大规模可升级为 Milvus 2.5+ 原生 BM25 稀疏向量检索。
"""
from __future__ import annotations

import jieba
from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    return [t for t in jieba.lcut(text.lower()) if t.strip()]


class BM25Index:
    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._meta: list[dict] = []

    def build(self, rows: list[dict]) -> None:
        """rows: Milvus 全量 chunk（含 id/text/source/page/chunk_idx）。"""
        self._meta = rows
        self._bm25 = BM25Okapi([tokenize(r["text"]) for r in rows]) if rows else None

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        if not self._bm25:
            return []
        q = tokenize(query)
        if not q:
            return []
        scores = self._bm25.get_scores(q)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        out: list[dict] = []
        for i in order:
            if scores[i] > 0:
                h = dict(self._meta[i])
                h["bm25_score"] = float(scores[i])
                out.append(h)
        return out
