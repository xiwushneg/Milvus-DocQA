"""B2 自反思闭环：不足 → 改写 → 再检索，以及预算/轮次/新颖度停止。"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from core.agentic.runtime import AgentRuntime
from core.pipeline import RagPipeline


class FakeEmbedder:
    def __init__(self):
        self.queries: list[str] = []
        self.last_query = ""

    def embed_query(self, query: str) -> list[float]:
        self.last_query = query
        self.queries.append(query)
        return [0.1, 0.2, 0.3, 0.4]


class GapStore:
    """第一轮返回普通块；含"缺口"的查询返回新增块。"""

    def __init__(self, embedder: FakeEmbedder):
        self.embedder = embedder
        self.search_calls = 0
        self.rows = [
            {"id": 1, "text": "Milvus 是开源向量数据库，支持 HNSW 索引。",
             "source": "doc_a.md", "page": 1, "chunk_idx": 0},
            {"id": 2, "text": "BM25 是稀疏检索算法，擅长关键词精确匹配。",
             "source": "doc_b.md", "page": 1, "chunk_idx": 0},
            {"id": 3, "text": "RRF 将多路召回结果融合为一个排名。",
             "source": "doc_b.md", "page": 2, "chunk_idx": 1},
            {"id": 4, "text": "交叉编码器重排提升上下文相关性。",
             "source": "doc_c.md", "page": 1, "chunk_idx": 0},
            {"id": 5, "text": "缺口一：向量检索使用分层图结构。",
             "source": "doc_a.md", "page": 3, "chunk_idx": 1},
            {"id": 6, "text": "缺口二：稀疏检索依赖词频统计。",
             "source": "doc_b.md", "page": 3, "chunk_idx": 1},
        ]

    def all_chunks(self, max_rows: int = 100000, wait_visible: bool = False) -> list[dict]:
        return [dict(r) for r in self.rows]

    def search_dense(
        self, query_vec: list[float], top_k: int = 20, filter_expr: str = ""
    ) -> list[dict]:
        self.search_calls += 1
        want_gap = "缺口" in self.embedder.last_query
        rows = [r for r in self.rows if ("缺口" in r["text"]) == want_gap]
        match = re.search(r'source == "(.*?)"', filter_expr)
        if match:
            rows = [r for r in rows if r["source"] == match.group(1)]
        hits = []
        for i, row in enumerate(rows):
            hit = dict(row)
            hit["dense_score"] = round(1.0 - i * 0.1, 6)
            hits.append(hit)
        return hits[:top_k]


class FakeReranker:
    enabled = False


class AnswerClient:
    def __init__(self):
        self.calls = 0

    def chat(self, messages):
        self.calls += 1
        return "[answer] 综合多轮证据回答。"


class ScriptedBrain:
    def __init__(self, assess_scores, rewrites=None, conflicts=None, hyde=True):
        self.assess_scores = list(assess_scores)
        self.rewrites = list(rewrites or [])
        self.conflicts = conflicts or []
        self.assess_calls = 0
        self.rewrite_calls = 0
        self.hyde_calls = 0
        self.hyde_enabled = hyde

    def chat(self, messages):
        system = messages[0]["content"]
        question = messages[-1]["content"]
        if "查询规划器" in system:
            return json.dumps(
                {
                    "intent": "fact",
                    "sub_queries": [{"text": question, "purpose": "主查询"}],
                },
                ensure_ascii=False,
            )
        if "充分性评估器" in system:
            idx = min(self.assess_calls, len(self.assess_scores) - 1)
            score = self.assess_scores[idx]
            self.assess_calls += 1
            return json.dumps(
                {
                    "support": [{"sub_query": "q", "score": score}] * 3,
                    "missing_aspects": ["缺口一"] if score < 1 else [],
                    "conflicts": self.conflicts if score < 1 else [],
                },
                ensure_ascii=False,
            )
        if "查询改写器" in system:
            if self.rewrite_calls >= len(self.rewrites):
                return "{}"
            replies = self.rewrites[self.rewrite_calls]
            self.rewrite_calls += 1
            return json.dumps(replies, ensure_ascii=False)
        if "领域专家" in system:
            if not self.hyde_enabled:
                return ""
            self.hyde_calls += 1
            return "假设答案：向量检索采用分层图结构组织数据。"
        return "stub"


def _runtime(tmp_path: Path, brain: ScriptedBrain, *, max_rounds=3, max_llm_calls=8):
    embedder = FakeEmbedder()
    store = GapStore(embedder)
    cfg = {
        "retrieval": {
            "dense_top_k": 5,
            "bm25_top_k": 5,
            "rrf_k": 60,
            "final_top_k": 3,
            "rerank": {"enabled": False},
        },
        "agent": {
            "enabled": True,
            "brain_llm": "cloud",
            "max_rounds": max_rounds,
            "max_subqueries_per_round": 3,
            "top_k_per_subquery": 3,
            "max_final_evidence": 6,
            "on_budget_exhausted": "partial",
            "analyze": {"mode": "brain", "max_retries": 0},
            "routing": {"keyword_bm25_first": True, "summary_evidence_bonus": 2},
            "sufficiency": {
                "tau_low": 0.45,
                "tau_high": 0.65,
                "weights": {
                    "claim_support": 0.45,
                    "coverage": 0.25,
                    "top_score": 0.20,
                    "diversity": 0.10,
                },
            },
            "hyde": {"enabled": True, "start_round": 2},
            "budget": {
                "max_llm_calls": max_llm_calls,
                "max_brain_calls": 8,
                "max_wall_ms": 30000,
                "novelty_epsilon": 0.10,
            },
            "trace": {"enabled": True, "dir": str(tmp_path)},
        },
    }
    answer = AnswerClient()
    pipeline = RagPipeline(
        cfg, store=store, embedder=embedder, llm=answer, reranker=FakeReranker()
    )
    llms = {"cloud": {"client": brain}, "ft": {"client": answer}}
    return AgentRuntime(cfg, pipeline, llms), store, embedder, answer


def _events(out, stage):
    return [e for e in out["trace"] if e["stage"] == stage]


def test_insufficient_then_rewrite_then_sufficient(tmp_path):
    brain = ScriptedBrain(
        assess_scores=[0, 1],
        rewrites=[
            {
                "sub_queries": [{"text": "缺口一 向量检索", "purpose": "补齐结构细节"}],
                "reason": "补充缺失方面",
            }
        ],
    )
    runtime, _, embedder, answer = _runtime(tmp_path, brain)
    out = runtime.ask("Milvus 支持什么索引", session_id="b2a", llm="ft", include_trace=True)

    assert out["status"] == "answered"
    assert out["metrics"]["rounds"] == 2
    assert out["metrics"]["rewrites"] == 1
    assert out["metrics"]["sufficiency_score"] >= 0.65
    assert len(_events(out, "assess")) == 2
    assert len(_events(out, "rewrite")) == 1
    assert answer.calls == 1

    # 第二轮起启用 HyDE，且假设答案只进入 dense 查询
    assert len(_events(out, "hyde")) == 1
    assert any("假设答案" in q for q in embedder.queries)

    # 证据跨轮累加且编号稳定
    chunks = _events(out, "merge")[-1]["payload"]["chunks"]
    ids = [c["evidence_id"] for c in chunks]
    assert len(ids) == len(set(ids))
    assert "E1" in ids
    assert any(c["round_found"] == 2 for c in chunks)


def test_novelty_stop_when_retrieval_saturates(tmp_path):
    brain = ScriptedBrain(
        assess_scores=[0, 0, 0],
        rewrites=[{"sub_queries": [{"text": "Milvus 支持什么索引 补充"}]}],
    )
    runtime, _, _, _ = _runtime(tmp_path, brain)
    out = runtime.ask("Milvus 支持什么索引", session_id="b2b", llm="ft", include_trace=True)

    assert out["status"] == "partial"
    assert out["metrics"]["rounds"] == 2
    assert out["metrics"]["stop_reason"] == "no_new_evidence"
    assert out["refused"] is False


def test_max_rounds_stops_with_partial_answer(tmp_path):
    brain = ScriptedBrain(
        assess_scores=[0, 0, 0],
        rewrites=[{"sub_queries": [{"text": "缺口一 向量检索"}]}],
    )
    runtime, _, _, answer = _runtime(tmp_path, brain, max_rounds=2)
    out = runtime.ask("Milvus 支持什么索引", session_id="b2c", llm="ft", include_trace=True)

    assert out["metrics"]["rounds"] == 2
    assert out["metrics"]["stop_reason"] == "max_rounds"
    assert out["status"] == "partial"
    assert answer.calls == 1


def test_budget_stops_loop_after_assessment(tmp_path):
    brain = ScriptedBrain(assess_scores=[0])
    runtime, _, _, answer = _runtime(tmp_path, brain, max_llm_calls=2)
    out = runtime.ask("Milvus 支持什么索引", session_id="b2d", llm="ft", include_trace=True)

    assert out["metrics"]["rounds"] == 1
    assert out["metrics"]["stop_reason"] == "budget"
    assert out["metrics"]["rewrites"] == 0
    assert out["status"] == "partial"
    assert answer.calls == 1


def test_conflicts_stop_loop_and_are_exposed_for_parallel_answer(tmp_path):
    """B4 起冲突直接终止循环并进入并列回答，不再改写重试。"""
    brain = ScriptedBrain(
        assess_scores=[0],
        conflicts=[{"claim": "维度存在 1024 与 768 两种说法", "evidence_ids": ["E1", "E2"]}],
    )
    runtime, _, _, _ = _runtime(tmp_path, brain, max_rounds=1)
    out = runtime.ask("Milvus 支持什么索引", session_id="b2e", llm="ft", include_trace=True)

    assert out["metrics"]["rounds"] == 1
    assert out["metrics"]["stop_reason"] == "conflict"
    assert out["metrics"]["rewrites"] == 0
    assert out["status"] == "answered"
    assert out["conflicts"][0]["evidence_ids"] == ["E1", "E2"]
    assess = _events(out, "assess")[0]["payload"]
    assert assess["decision"] == "conflict"
