"""可选重排序模块：bge-reranker 交叉编码器（api / local 两种模式）。

交叉编码器对（问题, 文档）联合编码打分，精度高于向量相似度但计算更贵，
因此只对 RRF 融合后的少量候选精排（典型：Top50 → Top5）。
"""
from __future__ import annotations

import os


class Reranker:
    def __init__(self, rcfg: dict):
        self.enabled = bool(rcfg.get("enabled", False))
        self.mode = rcfg.get("mode", "api")
        self.model = rcfg.get("model", "BAAI/bge-reranker-v2-m3")
        self._api_base = (rcfg.get("api_base") or "").rstrip("/")
        self._key_env = rcfg.get("api_key_env", "EMBEDDING_API_KEY")
        self._local_model = None  # 延迟加载

    def rerank(self, query: str, hits: list[dict], top_n: int = 5) -> list[dict]:
        if not self.enabled or not hits:
            return hits[:top_n]
        docs = [h["text"] for h in hits]
        if self.mode == "api":
            scored = self._rerank_api(query, docs)
        else:
            scored = self._rerank_local(query, docs)
        out: list[dict] = []
        for idx, score in scored[:top_n]:
            h = dict(hits[idx])
            h["rerank_score"] = float(score)
            out.append(h)
        return out

    def _rerank_api(self, query: str, docs: list[str]):
        """OpenAI 风格 /rerank 接口（如 SiliconFlow 托管 bge-reranker-v2-m3）。"""
        import httpx

        key = os.environ.get(self._key_env, "")
        resp = httpx.post(
            f"{self._api_base}/rerank",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": self.model,
                "query": query,
                "documents": docs,
                "top_n": len(docs),
            },
            timeout=120,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        return [(r["index"], r.get("relevance_score", 0.0)) for r in results]

    def _rerank_local(self, query: str, docs: list[str]):
        from sentence_transformers import CrossEncoder  # 延迟导入

        if self._local_model is None:
            self._local_model = CrossEncoder(self.model, max_length=512)
        scores = self._local_model.predict([(query, d) for d in docs])
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [(i, float(scores[i])) for i in order]
