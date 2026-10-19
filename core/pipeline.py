"""RAG 核心编排。

入库链路: 解析(parse) → 分块(split) → 向量化(embed) → Milvus 入库 → 重建 BM25
问答链路: 问题向量化 → 稠密检索 ∥ BM25 检索 → RRF 融合 → (可选重排) → LLM 生成
每个阶段记录耗时，供论文性能分析使用。
"""
from __future__ import annotations

import time
from pathlib import Path

from core.bm25 import BM25Index
from core.embedder import build_embedder
from core.fusion import rrf_fuse
from core.loader import file_sha256, parse_bytes
from core.llm import LLMClient, build_messages
from core.reranker import Reranker
from core.splitter import split_segments
from core.vectorstore import MilvusStore


class RagPipeline:
    def __init__(
        self,
        cfg: dict,
        store: MilvusStore | None = None,
        embedder=None,
        llm=None,
        reranker: Reranker | None = None,
    ):
        self.cfg = cfg
        self.store = store or MilvusStore(
            uri=cfg["milvus"]["uri"],
            collection=cfg["milvus"].get("collection", "docqa_chunks"),
            dense_dim=int(cfg["milvus"].get("dense_dim", 1024)),
        )
        self.embedder = embedder or build_embedder(cfg)
        self.llm = llm or LLMClient(cfg.get("llm", {}))
        self.reranker = reranker or Reranker(cfg.get("retrieval", {}).get("rerank", {}))
        self.bm25 = BM25Index()
        self._bm25_ready = False
        self._bm25_dirty = False  # 刚写入过数据，刷新时需等待可见

    # ==================== 入库 ====================
    def ingest_bytes(self, filename: str, data: bytes) -> dict:
        t0 = time.perf_counter()
        segments = parse_bytes(filename, data)
        chunks = split_segments(
            segments,
            source=filename,
            chunk_size=int(self.cfg.get("chunking", {}).get("chunk_size", 500)),
            chunk_overlap=int(self.cfg.get("chunking", {}).get("chunk_overlap", 80)),
            min_chunk_size=int(self.cfg.get("chunking", {}).get("min_chunk_size", 30)),
        )
        t1 = time.perf_counter()
        vectors = self.embedder.embed_documents([c.text for c in chunks]) if chunks else []
        t2 = time.perf_counter()
        rows = []
        for c, v in zip(chunks, vectors):
            row = c.to_row()
            row["dense"] = v
            rows.append(row)
        self.store.insert_chunks(rows)
        self._bm25_dirty = True
        self.refresh_bm25()
        return {
            "source": filename,
            "sha256": file_sha256(data),
            "segments": len(segments),
            "chunks": len(chunks),
            "split_ms": int((t1 - t0) * 1000),
            "embed_ms": int((t2 - t1) * 1000),
            "elapsed_ms": int((time.perf_counter() - t0) * 1000),
        }

    def ingest_paths(self, paths: list[str]) -> list[dict]:
        out = []
        for p in paths:
            path = Path(p)
            out.append(self.ingest_bytes(path.name, path.read_bytes()))
        return out

    # ==================== BM25 同步 ====================
    def refresh_bm25(self) -> int:
        rows = self.store.all_chunks(wait_visible=self._bm25_dirty)
        self.bm25.build(rows)
        self._bm25_ready = True
        self._bm25_dirty = False
        return len(rows)

    def _ensure_bm25(self) -> None:
        if not self._bm25_ready:
            self.refresh_bm25()

    # ==================== 问答 ====================
    def ask(
        self, question: str, top_k: int | None = None, llm: LLMClient | None = None
    ) -> dict:
        rc = self.cfg.get("retrieval", {})
        dense_top_k = int(rc.get("dense_top_k", 20))
        bm25_top_k = int(rc.get("bm25_top_k", 20))
        rrf_k = int(rc.get("rrf_k", 60))
        final_top_k = int(top_k or rc.get("final_top_k", 5))

        t0 = time.perf_counter()
        self._ensure_bm25()
        qvec = self.embedder.embed_query(question)
        t1 = time.perf_counter()
        dense_hits = self.store.search_dense(qvec, top_k=dense_top_k)
        t2 = time.perf_counter()
        bm25_hits = self.bm25.search(question, top_k=bm25_top_k)
        t3 = time.perf_counter()
        fused = rrf_fuse([dense_hits, bm25_hits], k=rrf_k)
        t4 = time.perf_counter()
        if self.reranker.enabled:
            contexts = self.reranker.rerank(question, fused, top_n=final_top_k)
        else:
            contexts = fused[:final_top_k]
        t5 = time.perf_counter()
        answer = (llm or self.llm).chat(build_messages(question, contexts))
        t6 = time.perf_counter()

        return {
            "answer": answer,
            "references": [
                {
                    "source": h["source"],
                    "page": h["page"],
                    "chunk_idx": h.get("chunk_idx", -1),
                    "snippet": h["text"][:200],
                    "score": round(float(h.get("rrf_score", 0.0)), 6),
                    "channels": h.get("channels", []),
                }
                for h in contexts
            ],
            "timings": {
                "embed_ms": int((t1 - t0) * 1000),
                "dense_ms": int((t2 - t1) * 1000),
                "bm25_ms": int((t3 - t2) * 1000),
                "fuse_ms": int((t4 - t3) * 1000),
                "rerank_ms": int((t5 - t4) * 1000),
                "llm_ms": int((t6 - t5) * 1000),
            },
            "total_ms": int((time.perf_counter() - t0) * 1000),
        }


def build_pipeline(cfg: dict) -> RagPipeline:
    """按配置装配完整流水线（供服务端/脚本使用）。"""
    return RagPipeline(cfg)
