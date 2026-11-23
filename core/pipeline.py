"""RAG 核心编排。

入库链路: 解析(parse) → 分块(split) → 向量化(embed) → Milvus 入库 → 重建 BM25
问答链路: 问题向量化 → 稠密检索 ∥ BM25 检索 → RRF 融合 → (可选重排) → LLM 生成
每个阶段记录耗时，供论文性能分析使用。
"""
from __future__ import annotations

import json
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
        # sha256 去重清单：未配置 manifest_path 时关闭（单元测试/脚本零副作用）
        ingest_cfg = cfg.get("ingest") or {}
        self._manifest_path = str(ingest_cfg.get("manifest_path") or "").strip()
        self._preserve_docx_order = bool(ingest_cfg.get("preserve_docx_order", False))
        self._manifest_lock = Lock()
        self._manifest: dict[str, dict] = self._load_manifest()

    # ==================== 去重清单（sha256 → 入库记录） ====================
    def _load_manifest(self) -> dict[str, dict]:
        if not self._manifest_path:
            return {}
        path = Path(self._manifest_path)
        if not path.is_file():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(payload, dict):
            return {}
        if payload.get("collection") not in (None, self.store.collection):
            return {}  # collection 已切换，旧清单不再可信
        files = payload.get("files")
        if not isinstance(files, dict):
            return {}
        return {str(k): v for k, v in files.items() if isinstance(v, dict)}

    def _save_manifest(self) -> None:
        if not self._manifest_path:
            return
        path = Path(self._manifest_path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "version": 1,
                "collection": self.store.collection,
                "files": self._manifest,
            }
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            tmp.replace(path)
        except OSError:
            pass  # 清单落盘失败不影响入库主流程

    def _dedup_hit(self, sha: str) -> dict | None:
        """sha256 命中且原 source 仍在库中时返回清单条目。"""
        with self._manifest_lock:
            entry = self._manifest.get(sha)
            if not entry:
                return None
            source = str(entry.get("source", ""))
            if source and self.store.count_source(source) > 0:
                return entry
            # 原文档已被删除或 collection 重建：清掉过期条目，允许重新入库
            self._manifest.pop(sha, None)
            self._save_manifest()
            return None

    def _record_ingest(self, sha: str, source: str, chunks: int) -> None:
        if not self._manifest_path:
            return
        with self._manifest_lock:
            self._manifest[sha] = {
                "source": source,
                "chunks": int(chunks),
                "ingested_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            self._save_manifest()

    def forget_source(self, source: str) -> int:
        """删除文档全部块、清理去重清单并重建 BM25。"""
        deleted = self.store.delete_by_source(source)
        if self._manifest_path:
            with self._manifest_lock:
                stale = [k for k, v in self._manifest.items() if v.get("source") == source]
                for key in stale:
                    self._manifest.pop(key, None)
                if stale:
                    self._save_manifest()
        self._refresh_bm25_or_defer()
        return deleted

    # ==================== 入库 ====================
    def ingest_bytes(
        self, filename: str, data: bytes, rebuild_bm25: bool = True
    ) -> dict:
        t0 = time.perf_counter()
        sha = file_sha256(data)
        duplicate = self._dedup_hit(sha)
        if duplicate is not None:
            return {
                "source": filename,
                "sha256": sha,
                "segments": 0,
                "chunks": 0,
                "split_ms": 0,
                "embed_ms": 0,
                "elapsed_ms": int((time.perf_counter() - t0) * 1000),
                "skipped": True,
                "reason": "duplicate",
                "duplicate_of": duplicate.get("source", ""),
            }
        segments = parse_bytes(
            filename, data, preserve_docx_order=self._preserve_docx_order
        )
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
        self._record_ingest(sha, filename, len(chunks))
        self._bm25_dirty = True
        if rebuild_bm25:
            self._refresh_bm25_or_defer()
        return {
            "source": filename,
            "sha256": sha,
            "segments": len(segments),
            "chunks": len(chunks),
            "split_ms": int((t1 - t0) * 1000),
            "embed_ms": int((t2 - t1) * 1000),
            "elapsed_ms": int((time.perf_counter() - t0) * 1000),
        }

    def ingest_paths(self, paths: list[str]) -> list[dict]:
        """批量入库：全部文件落库后只重建一次 BM25（避免 O(n²)）。"""
        out = []
        for p in paths:
            path = Path(p)
            out.append(self.ingest_bytes(path.name, path.read_bytes(), rebuild_bm25=False))
        if any(not s.get("skipped") for s in out):
            self._refresh_bm25_or_defer()
        return out

    # ==================== BM25 同步 ====================
    def _refresh_bm25_or_defer(self) -> None:
        try:
            self.refresh_bm25()
        except Exception:
            self._bm25_ready = False  # 本次重建失败：下次检索时重试
            raise

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

    def close(self) -> None:
        """释放嵌入/LLM 的 HTTP 连接池（应用 shutdown 时调用）。"""
        for obj in (self.embedder, self.llm):
            close = getattr(obj, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass

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
