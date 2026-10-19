"""RAG 核心编排。

入库链路: 解析(parse) → 分块(split) → 向量化(embed) → Milvus 入库 → 重建 BM25
问答链路: 问题向量化 → 稠密检索 ∥ BM25 检索 → RRF 融合 → (可选重排) → LLM 生成
每个阶段记录耗时，供论文性能分析使用。
"""
from __future__ import annotations

import time
from pathlib import Path
from threading import Lock

from core.bm25 import BM25Index
from core.embedder import build_embedder
from core.fusion import rrf_fuse
from core.loader import file_sha256, parse_bytes
from core.llm import LLMClient, build_messages, build_rewrite_messages
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
        self._bm25_lock = Lock()  # 并发冷启动时只构建一次，避免重复全量拉取竞争

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
        with self._bm25_lock:
            rows = self.store.all_chunks(wait_visible=self._bm25_dirty)
            self.bm25.build(rows)
            self._bm25_ready = True
            self._bm25_dirty = False
            return len(rows)

    def _ensure_bm25(self) -> None:
        if not self._bm25_ready:
            self.refresh_bm25()

    # ==================== 检索（问答与流式问答共用） ====================
    def _retrieve(self, question: str, top_k: int | None = None) -> tuple[list[dict], dict]:
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
        timings = {
            "embed_ms": int((t1 - t0) * 1000),
            "dense_ms": int((t2 - t1) * 1000),
            "bm25_ms": int((t3 - t2) * 1000),
            "fuse_ms": int((t4 - t3) * 1000),
            "rerank_ms": int((t5 - t4) * 1000),
        }
        return contexts, timings

    @staticmethod
    def _format_refs(contexts: list[dict]) -> list[dict]:
        return [
            {
                "source": h["source"],
                "page": h["page"],
                "chunk_idx": h.get("chunk_idx", -1),
                "snippet": h["text"][:200],
                "score": round(float(h.get("rrf_score", 0.0)), 6),
                "channels": h.get("channels", []),
            }
            for h in contexts
        ]

    # ==================== 问答 ====================
    def _rewrite_query(self, history: list[dict], question: str, llm: LLMClient) -> str:
        """指代消解：结合历史把追问改写为独立问题；失败时回退原问题。"""
        if not history:
            return question
        try:
            rewritten = llm.chat(build_rewrite_messages(history, question)).strip().strip('"“”')
            if rewritten and len(rewritten) <= 300:
                return rewritten
        except Exception:
            pass
        return question

    def ask(
        self,
        question: str,
        top_k: int | None = None,
        llm: LLMClient | None = None,
        history: list[dict] | None = None,
    ) -> dict:
        t0 = time.perf_counter()
        eff_llm = llm or self.llm
        # 改写固定用主配置模型（cloud）：微调模型训练分布是 RAG 问答，指令改写遵循差
        rewritten = self._rewrite_query(history or [], question, self.llm)
        contexts, timings = self._retrieve(rewritten, top_k)
        answer = eff_llm.chat(build_messages(rewritten, contexts))
        timings["llm_ms"] = int(
            (time.perf_counter() - t0) * 1000 - sum(timings.values())
        )
        return {
            "answer": answer,
            "references": self._format_refs(contexts),
            "timings": timings,
            "total_ms": int((time.perf_counter() - t0) * 1000),
        }

    def prepare(
        self,
        question: str,
        top_k: int | None = None,
        rewrite_llm: LLMClient | None = None,
        history: list[dict] | None = None,
    ) -> dict:
        """流式问答的检索阶段（同步短阻塞，供异步端点放线程池执行）。

        返回改写后的问题、上下文块与检索耗时；生成阶段由调用方异步流式完成。
        """
        t0 = time.perf_counter()
        history = history or []
        if history:
            # 改写固定用主配置模型（cloud）：微调模型训练分布是 RAG 问答，指令改写遵循差
            rewritten = self._rewrite_query(history, question, rewrite_llm or self.llm)
        else:
            rewritten = question
        contexts, timings = self._retrieve(rewritten, top_k)
        timings["prep_ms"] = int((time.perf_counter() - t0) * 1000)
        return {
            "rewritten": rewritten,
            "contexts": contexts,
            "timings": timings,
        }


def build_pipeline(cfg: dict) -> RagPipeline:
    """按配置装配完整流水线（供服务端/脚本使用）。"""
    return RagPipeline(cfg)
