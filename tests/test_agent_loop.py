"""B1 单轮闭环：五类意图分支、路由、过滤、块类型与降级。"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from core.agentic.retrieval import RetrievalAdapter
from core.agentic.runtime import AgentRuntime
from core.agentic.state import SubQuery
from core.pipeline import RagPipeline


class FakeEmbedder:
    def __init__(self):
        self.queries: list[str] = []

    def embed_query(self, query: str) -> list[float]:
        self.queries.append(query)
        return [0.1, 0.2, 0.3, 0.4]


class FakeStore:
    def __init__(self):
        self.rows = [
            {
                "id": 1,
                "text": "Milvus 是开源向量数据库，支持 HNSW 索引。",
                "source": "doc_a.md",
                "page": 1,
                "chunk_idx": 0,
            },
            {
                "id": 2,
                "text": "| 参数 | 取值 |\n| --- | --- |\n| 维度 | 1024 |",
                "source": "doc_a.md",
                "page": 2,
                "chunk_idx": 1,
            },
            {
                "id": 3,
                "text": "BM25 是稀疏检索算法，擅长关键词精确匹配。",
                "source": "doc_b.md",
                "page": 1,
                "chunk_idx": 0,
            },
            {
                "id": 4,
                "text": "RRF 将多路召回结果融合为一个排名。",
                "source": "doc_b.md",
                "page": 2,
                "chunk_idx": 1,
            },
            {
                "id": 5,
                "text": "交叉编码器重排提升上下文相关性。",
                "source": "doc_c.md",
                "page": 1,
                "chunk_idx": 0,
            },
        ]
        self.search_calls = 0
        self.last_filter = ""

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
            rows = [r for r in rows if r["page"] == int(match.group(1))]
        hits = []
        for i, row in enumerate(rows):
            hit = dict(row)
            hit["dense_score"] = round(1.0 - i * 0.15, 6)
            hits.append(hit)
        return hits[:top_k]


class FakeReranker:
    enabled = False


class AnswerClient:
    def __init__(self):
        self.calls = 0

    def chat(self, messages: list[dict]) -> str:
        self.calls += 1
        return "[answer] 根据证据回答。"


def _plan_for(question: str) -> dict:
    if "无关" in question:
        return {"intent": "out_of_scope", "sub_queries": []}
    if "对比" in question:
        return {
            "intent": "comparison",
            "sub_queries": [
                {"text": "Milvus 的索引类型", "purpose": "实体 A"},
                {"text": "BM25 的检索原理", "purpose": "实体 B"},
            ],
        }
    if "汇总" in question:
        return {
            "intent": "summary",
            "sub_queries": [{"text": "文档核心结论", "purpose": "汇总"}],
        }
    if "多跳" in question:
        return {
            "intent": "multihop",
            "sub_queries": [
                {"text": "HNSW 是什么", "purpose": "第一跳"},
                {"text": "交叉编码器如何重排", "purpose": "第二跳"},
            ],
        }
    return {
        "intent": "fact",
        "sub_queries": [{"text": question, "purpose": "主查询"}],
    }


class PlannerBrain:
    def __init__(self, broken_json: bool = False):
        self.broken_json = broken_json
        self.planner_calls = 0
        self.hyde_calls = 0

    def chat(self, messages: list[dict]) -> str:
        system = messages[0]["content"]
        question = messages[-1]["content"]
        if "查询改写助手" in system:
            return question
        if "查询规划器" in system:
            self.planner_calls += 1
            if self.broken_json:
                return "这不是 JSON"
            data = _plan_for(question)
            data.setdefault("needs_table", False)
            data.setdefault("needs_calculation", False)
            return json.dumps(data, ensure_ascii=False)
        if "充分性评估器" in system:
            return json.dumps(
                {
                    "support": [
                        {"sub_query": "q", "score": 1, "reason": "充分"}
                    ] * 3,
                    "missing_aspects": [],
                    "conflicts": [],
                },
                ensure_ascii=False,
            )
        if "领域专家" in system:
            self.hyde_calls += 1
            return "假设答案：Milvus 支持 HNSW 索引，维度为 1024。"
        return "stub"


@pytest.fixture()
def env(tmp_path: Path):
    store = FakeStore()
    embedder = FakeEmbedder()
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
            "max_rounds": 3,
            "max_subqueries_per_round": 3,
            "top_k_per_subquery": 3,
            "max_final_evidence": 3,
            "analyze": {"mode": "brain", "max_retries": 1},
            "routing": {"keyword_bm25_first": True, "summary_evidence_bonus": 2},
            "hyde": {"enabled": True, "start_round": 2},
            "trace": {"enabled": True, "dir": str(tmp_path)},
            "budget": {"max_llm_calls": 8, "max_brain_calls": 6, "max_wall_ms": 30000},
            "citations": {"tau_answer_confidence": 0.55},
        },
    }
    pipeline = RagPipeline(
        cfg,
        store=store,
        embedder=embedder,
        llm=AnswerClient(),
        reranker=FakeReranker(),
    )
    brain = PlannerBrain()
    answer = AnswerClient()
    llms = {"cloud": {"client": brain}, "ft": {"client": answer}}
    runtime = AgentRuntime(cfg, pipeline, llms)
    return runtime, store, embedder, brain, answer, cfg, pipeline


def _events(out: dict, stage: str) -> list[dict]:
    return [e for e in out["trace"] if e["stage"] == stage]


def test_fact_single_round_with_chunk_types(env):
    runtime, store, _, brain, answer, _, _ = env
    out = runtime.ask("Milvus 支持什么索引？", session_id="s1", llm="ft", include_trace=True)

    assert out["status"] == "answered"
    assert out["metrics"]["rounds"] == 1
    assert out["references"]
    assert all(ref.get("chunk_type") in {"prose", "table", "code", "enum", "formula"} for ref in out["references"])
    assert _events(out, "route")[0]["payload"]["decision"]["channels"] == ["bm25"]
    assert brain.planner_calls == 1
    assert answer.calls == 1
    assert store.search_calls == 0  # BM25 命中，无需 dense 兜底


def test_comparison_multi_subquery(env):
    runtime, _, _, _, _, _, _ = env
    out = runtime.ask("对比 Milvus 和 BM25 的特点", session_id="s2", llm="ft", include_trace=True)

    assert out["status"] == "answered"
    assert out["metrics"]["sub_queries"] == 2
    routes = _events(out, "route")
    assert len(routes) == 2
    assert {r["payload"]["query"] for r in routes} == {
        "Milvus 的索引类型",
        "BM25 的检索原理",
    }


def test_multihop_plan_retrieves_all_hops_in_b1(env):
    runtime, _, _, _, _, _, _ = env
    out = runtime.ask(
        "多跳问题：HNSW 是什么，交叉编码器如何重排",
        session_id="s2b",
        llm="ft",
        include_trace=True,
    )

    assert out["status"] == "answered"
    assert out["metrics"]["sub_queries"] == 2
    assert len(_events(out, "route")) == 2


def test_summary_applies_source_filter(env):
    runtime, store, _, _, _, _, _ = env
    out = runtime.ask(
        "请汇总 doc_a.md 这份文档的核心内容", session_id="s3", llm="ft", include_trace=True
    )

    assert out["status"] == "answered"
    assert out["references"]
    assert all(ref["source"] == "doc_a.md" for ref in out["references"])
    assert 'source == "doc_a.md"' in store.last_filter
    routes = _events(out, "route")
    assert routes[0]["payload"]["decision"]["reason"] == "filtered"
    assert routes[0]["payload"]["decision"]["evidence_bonus"] == 2


def test_out_of_scope_refuses_without_retrieval(env):
    runtime, store, _, _, answer, _, _ = env
    out = runtime.ask("今天天气如何，无关问题", session_id="s4", llm="ft", include_trace=True)

    assert out["refused"] is True
    assert out["metrics"]["rounds"] == 0
    assert store.search_calls == 0
    assert answer.calls == 0
    assert _events(out, "analyze")[0]["payload"]["intent"] == "out_of_scope"


def test_bm25_zero_hits_falls_back_to_dense(env):
    runtime, store, embedder, _, _, _, _ = env
    out = runtime.ask("ZZTOP", session_id="s5", llm="ft", include_trace=True)

    assert out["status"] == "answered"
    route = _events(out, "route")[0]["payload"]
    assert route["fallback_used"] is True
    assert route["decision"]["channels"] == ["bm25"]
    assert store.search_calls == 1
    assert embedder.queries  # dense 兜底确实执行


def test_degraded_planner_falls_back_to_rules(env):
    runtime, _, _, _, answer, cfg, pipeline = env
    broken = PlannerBrain(broken_json=True)
    runtime.llms["cloud"]["client"] = broken
    runtime.brain = broken
    runtime.analyzer.brain = broken

    out = runtime.ask("Milvus 是什么", session_id="s6", llm="ft", include_trace=True)

    assert out["status"] == "answered"
    analyze = _events(out, "analyze")[0]["payload"]
    assert analyze["source"] == "rules"
    assert analyze["degraded"] is True
    assert broken.planner_calls == 2  # 首次 + 重试 1 次
    assert answer.calls == 1


def test_hyde_not_used_on_first_round(env):
    runtime, _, _, brain, _, _, _ = env
    out = runtime.ask("HNSW 索引是什么", session_id="s7", llm="ft", include_trace=True)
    assert brain.hyde_calls == 0
    assert _events(out, "hyde") == []


def test_adapter_hyde_changes_only_dense_query(env):
    _, _, embedder, _, _, _, pipeline = env
    adapter = RetrievalAdapter.from_pipeline(pipeline)
    hits, _ = adapter.search(
        SubQuery(text="HNSW 索引", hyde_text="假设答案：Milvus 使用 HNSW。"),
        top_k=2,
    )
    assert hits
    assert embedder.queries[-1] == "假设答案：Milvus 使用 HNSW。"


def test_budget_exhaustion_reads_live_usage(env):
    runtime, _, _, brain, _, _, _ = env
    original_chat = brain.chat

    def low_sufficiency(messages: list[dict]) -> str:
        if "充分性评估器" in messages[0]["content"]:
            return json.dumps(
                {
                    "support": [{"sub_query": "q", "score": 0, "reason": "不足"}] * 3,
                    "missing_aspects": ["缺少细节"],
                    "conflicts": [],
                },
                ensure_ascii=False,
            )
        return original_chat(messages)

    brain.chat = low_sufficiency
    # 规划 + 充分性评估各一次，评估后预算即耗尽
    runtime.budget_template.max_llm_calls = 2
    runtime.citations_enabled = False

    out = runtime.ask(
        "Milvus 支持什么索引？", session_id="s8", llm="ft", include_trace=True
    )

    # 修复前 state.usage 恒为 0，会继续 rewrite 到第 2 轮
    assert out["metrics"]["rounds"] == 1
    assert out["metrics"]["stop_reason"] == "budget"
