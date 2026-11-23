"""B4 会话记忆：SessionStore、指代/页码检测、两轮焦点消解与过滤。"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from core.agentic.memory import SessionStore, detect_reference
from core.agentic.runtime import AgentRuntime
from core.pipeline import RagPipeline


# ==================== 单元：指代检测与 SessionStore ====================
def test_detect_reference_anaphora_and_page():
    assert detect_reference("它支持什么索引？") == {"anaphora": True, "page": None}
    assert detect_reference("这份文档讲了什么")["anaphora"] is True
    assert detect_reference("该表有哪些列")["anaphora"] is True

    ref = detect_reference("第 3 页的表格内容是什么")
    assert ref["anaphora"] is True
    assert ref["page"] == 3

    assert detect_reference("Milvus 支持什么索引？") == {
        "anaphora": False,
        "page": None,
    }


def test_session_store_history_truncation_and_focus():
    store = SessionStore(max_turns=2)
    store.append(
        "s1",
        "q1",
        "a1",
        sources=["doc_a.md"],
        evidence_ids=["E1", "E2"],
        focus_source="doc_a.md",
        focus_page=1,
    )
    store.append(
        "s1",
        "q2",
        "a2",
        sources=["doc_b.md"],
        evidence_ids=["E1"],
        focus_source="doc_b.md",
        focus_page=2,
    )
    # 空焦点不覆盖已有焦点
    store.append("s1", "q3", "a3")

    session = store.get("s1")
    assert [t.question for t in session.turns] == ["q2", "q3"]
    assert session.focus_source == "doc_b.md"
    assert session.focus_page == 2
    assert session.history(1) == [
        {"role": "user", "content": "q3"},
        {"role": "assistant", "content": "a3"},
    ]
    assert len(store) == 1


def test_session_store_ttl_expiry(monkeypatch):
    store = SessionStore(ttl_s=10)
    store.append("s1", "q", "a")
    assert store.get("s1") is not None

    real_time = time.time
    monkeypatch.setattr(
        "core.agentic.memory.time.time", lambda: real_time() + 11
    )
    assert store.get("s1") is None
    assert len(store) == 0


def test_session_store_persist_roundtrip(tmp_path: Path):
    path = tmp_path / "sessions.json"
    store = SessionStore(persist_path=path)
    store.append(
        "s1",
        "问题",
        "答案",
        sources=["doc_a.md"],
        evidence_ids=["E1"],
        focus_source="doc_a.md",
        focus_page=2,
    )
    assert path.is_file()

    reloaded = SessionStore(persist_path=path)
    session = reloaded.get("s1")
    assert session is not None
    assert session.focus_source == "doc_a.md"
    assert session.focus_page == 2
    assert session.turns[0].question == "问题"
    assert json.loads(path.read_text(encoding="utf-8"))["version"] == 1


# ==================== 端到端：两轮指代消解与焦点过滤 ====================
class FakeEmbedder:
    def embed_query(self, query: str) -> list[float]:
        return [0.1, 0.2, 0.3, 0.4]


class FocusStore:
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
        ]
        self.last_filter = ""

    def all_chunks(self, max_rows: int = 100000, wait_visible: bool = False) -> list[dict]:
        return [dict(r) for r in self.rows]

    def search_dense(
        self, query_vec: list[float], top_k: int = 20, filter_expr: str = ""
    ) -> list[dict]:
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
        self.answers: list[str] = []

    def chat(self, messages):
        self.answers.append(messages[-1]["content"])
        return "根据证据回答。"


class MemoryBrain:
    """固定 condense 结果，用于验证会话历史触发改写与焦点注入。"""

    def __init__(self, condensed: str):
        self.condensed = condensed
        self.condense_calls = 0
        self.planner_questions: list[str] = []

    def chat(self, messages):
        system = messages[0]["content"]
        question = messages[-1]["content"]
        if "查询改写助手" in system:
            self.condense_calls += 1
            return self.condensed
        if "查询规划器" in system:
            self.planner_questions.append(question)
            return json.dumps(
                {
                    "intent": "fact",
                    "sub_queries": [{"text": question, "purpose": "主查询"}],
                },
                ensure_ascii=False,
            )
        if "充分性评估器" in system:
            return json.dumps(
                {
                    "support": [{"sub_query": "q", "score": 1}] * 3,
                    "missing_aspects": [],
                    "conflicts": [],
                },
                ensure_ascii=False,
            )
        return "{}"


def _runtime(tmp_path: Path, brain: MemoryBrain) -> tuple[AgentRuntime, FocusStore]:
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
            "analyze": {"mode": "brain", "max_retries": 0},
            "routing": {"keyword_bm25_first": True, "summary_evidence_bonus": 2},
            "hyde": {"enabled": False},
            "budget": {
                "max_llm_calls": 8,
                "max_brain_calls": 8,
                "max_wall_ms": 30000,
                "novelty_epsilon": 0.10,
            },
            "memory": {"session_ttl_s": 7200, "max_turns": 10, "history_turns": 4},
            "trace": {"enabled": True, "dir": str(tmp_path)},
        },
    }
    store = FocusStore()
    answer = AnswerClient()
    pipeline = RagPipeline(
        cfg,
        store=store,
        embedder=FakeEmbedder(),
        llm=answer,
        reranker=FakeReranker(),
    )
    llms = {"cloud": {"client": brain}, "ft": {"client": answer}}
    return AgentRuntime(cfg, pipeline, llms), store


def _events(out: dict, stage: str) -> list[dict]:
    return [e for e in out["trace"] if e["stage"] == stage]


def test_second_turn_resolves_anaphora_with_focus_source(tmp_path: Path):
    brain = MemoryBrain("Milvus 支持什么索引？")
    runtime, _ = _runtime(tmp_path, brain)

    first = runtime.ask(
        "doc_a.md 这份文档支持什么索引？",
        session_id="mem-a",
        llm="ft",
        include_trace=True,
    )
    assert first["status"] == "answered"
    assert first["references"]
    assert all(r["source"] == "doc_a.md" for r in first["references"])
    assert brain.condense_calls == 0  # 首轮无历史，不做改写

    second = runtime.ask(
        "它支持什么索引？", session_id="mem-a", llm="ft", include_trace=True
    )
    assert second["status"] == "answered"
    assert brain.condense_calls == 1  # 会话历史触发指代消解
    assert brain.planner_questions[-1] == "Milvus 支持什么索引？"

    memory_event = _events(second, "memory")[0]["payload"]
    assert memory_event["turns"] == 1
    assert memory_event["history_used"] is True
    assert memory_event["anaphora"] is True
    assert memory_event["focus_source"] == "doc_a.md"

    # 焦点 source 进入检索过滤，引用全部来自同一文档
    route = _events(second, "route")[0]["payload"]
    assert 'source == "doc_a.md"' in route["filter"]
    assert all(r["source"] == "doc_a.md" for r in second["references"])

    session = runtime.sessions.get("mem-a")
    assert session is not None
    assert len(session.turns) == 2
    assert session.focus_source == "doc_a.md"


def test_second_turn_page_reference_combines_focus_source_and_page(tmp_path: Path):
    brain = MemoryBrain("第 2 页讲了什么？")
    runtime, store = _runtime(tmp_path, brain)

    runtime.ask("doc_a.md 支持什么索引？", session_id="mem-b", llm="ft")
    out = runtime.ask(
        "该文档第 2 页讲了什么？", session_id="mem-b", llm="ft", include_trace=True
    )

    memory_event = _events(out, "memory")[0]["payload"]
    assert memory_event["focus_source"] == "doc_a.md"
    assert memory_event["focus_page"] == 2

    route = _events(out, "route")[0]["payload"]
    assert 'source == "doc_a.md"' in route["filter"]
    assert "page == 2" in route["filter"]
    assert 'source == "doc_a.md" and page == 2' in store.last_filter
    assert out["references"]
    assert all(r["page"] == 2 for r in out["references"])


def test_new_topic_without_anaphora_does_not_apply_focus(tmp_path: Path):
    brain = MemoryBrain("BM25 是什么？")
    runtime, store = _runtime(tmp_path, brain)

    runtime.ask("doc_a.md 支持什么索引？", session_id="mem-c", llm="ft")
    store.last_filter = ""
    out = runtime.ask("BM25 是什么？", session_id="mem-c", llm="ft", include_trace=True)

    memory_event = _events(out, "memory")[0]["payload"]
    assert memory_event["anaphora"] is False
    assert memory_event["focus_source"] is None
    route = _events(out, "route")[0]["payload"]
    assert route["filter"] == ""
    assert store.last_filter == ""
