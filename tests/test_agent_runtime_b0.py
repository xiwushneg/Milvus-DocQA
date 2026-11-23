"""B0 Runtime 单测：直答、单轮检索、指代消解、拒答、brain/answer 分账。"""
from __future__ import annotations

import json
from pathlib import Path

from core.agentic.runtime import AgentRuntime
from core.pipeline import RagPipeline


class FakeEmbedder:
    def embed_query(self, query: str) -> list[float]:
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
                "text": "RRF 融合多路召回结果，提升检索稳定性。",
                "source": "doc_a.md",
                "page": 2,
                "chunk_idx": 1,
            },
            {
                "id": 3,
                "text": "交叉编码器重排序提升上下文相关性。",
                "source": "doc_b.md",
                "page": 1,
                "chunk_idx": 0,
            },
        ]
        self.search_calls = 0

    def all_chunks(self, max_rows: int = 100000, wait_visible: bool = False) -> list[dict]:
        return [dict(r) for r in self.rows]

    def search_dense(
        self, query_vec: list[float], top_k: int = 20, filter_expr: str = ""
    ) -> list[dict]:
        self.search_calls += 1
        hits = []
        for i, row in enumerate(self.rows):
            hit = dict(row)
            hit["dense_score"] = round(1.0 - i * 0.2, 6)
            hits.append(hit)
        return hits[:top_k]


class FakeReranker:
    enabled = False


class FakeChat:
    def __init__(self, label: str):
        self.label = label
        self.calls = 0

    def chat(self, messages: list[dict]) -> str:
        self.calls += 1
        system = messages[0]["content"]
        if "查询改写助手" in system:
            return "HNSW 索引是什么"
        if "查询规划器" in system:
            question = messages[-1]["content"]
            return json.dumps(
                {
                    "intent": "fact",
                    "sub_queries": [
                        {
                            "text": question,
                            "channels": ["dense", "bm25"],
                            "source": None,
                            "page": None,
                            "purpose": "主查询",
                        }
                    ],
                    "needs_table": False,
                    "needs_calculation": False,
                },
                ensure_ascii=False,
            )
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
        return f"[{self.label}] 根据资料回答。"


def _runtime(tmp_path: Path) -> tuple[AgentRuntime, FakeStore, FakeChat, FakeChat]:
    store = FakeStore()
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
            "routing": {"keyword_bm25_first": False},
            "trace": {"enabled": True, "dir": str(tmp_path)},
            "budget": {"max_llm_calls": 8, "max_brain_calls": 6, "max_wall_ms": 30000},
            "citations": {"tau_answer_confidence": 0.55},
        },
    }
    pipeline = RagPipeline(
        cfg,
        store=store,
        embedder=FakeEmbedder(),
        llm=FakeChat("base"),
        reranker=FakeReranker(),
    )
    cloud = FakeChat("cloud")
    ft = FakeChat("ft")
    llms = {"cloud": {"client": cloud}, "ft": {"client": ft}}
    return AgentRuntime(cfg, pipeline, llms), store, cloud, ft


def test_chitchat_skips_retrieval(tmp_path):
    runtime, store, cloud, ft = _runtime(tmp_path)
    out = runtime.ask("你好", session_id="s1", llm="ft", include_trace=True)

    assert out["status"] == "answered"
    assert out["references"] == []
    assert out["metrics"]["rounds"] == 0
    assert store.search_calls == 0
    assert ft.calls == 1
    assert cloud.calls == 0
    assert out["answer"].startswith("[ft]")


def test_factual_question_uses_single_round(tmp_path):
    runtime, store, _, ft = _runtime(tmp_path)
    out = runtime.ask("HNSW 索引是什么", session_id="s2", llm="ft", include_trace=True)

    assert out["status"] == "answered"
    assert out["references"]
    assert all("evidence_id" in ref for ref in out["references"])
    assert out["metrics"]["rounds"] == 1
    assert store.search_calls == 1
    assert ft.calls == 1

    stages = {e["stage"] for e in out["trace"]}
    assert {"normalize", "analyze", "retrieve", "merge", "compose"} <= stages


def test_history_triggers_brain_rewrite(tmp_path):
    runtime, _, cloud, ft = _runtime(tmp_path)
    out = runtime.ask(
        "它是什么",
        session_id="s3",
        llm="ft",
        history=[
            {"role": "user", "content": "介绍一下 Milvus"},
            {"role": "assistant", "content": "Milvus 是向量数据库。"},
        ],
        include_trace=True,
    )

    assert out["status"] == "answered"
    assert cloud.calls == 3   # brain：指代消解 + 结构化分析 + 充分性评估
    assert ft.calls == 1      # answer 生成
    roles = {e.get("model_role") for e in out["trace"] if e["stage"] == "llm"}
    assert "brain" in roles and "answer" in roles
    assert out["metrics"]["brain_llm_calls"] == 3
    assert out["metrics"]["answer_llm_calls"] == 1


def test_empty_question_is_refused_without_llm(tmp_path):
    runtime, store, _, ft = _runtime(tmp_path)
    out = runtime.ask("", session_id="s4", llm="ft")

    assert out["refused"] is True
    assert "无法可靠回答" in out["answer"]
    assert out["metrics"]["rounds"] == 0
    assert store.search_calls == 0
    assert ft.calls == 0


def test_trace_is_written(tmp_path):
    runtime, _, _, _ = _runtime(tmp_path)
    out = runtime.ask("HNSW 索引是什么", session_id="s5", llm="ft")
    trace_file = tmp_path / f"{out['trace_id']}.json"
    assert trace_file.is_file()
    assert out["trace_id"] in trace_file.read_text(encoding="utf-8")
