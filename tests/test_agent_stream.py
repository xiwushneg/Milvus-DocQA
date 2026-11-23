"""B5 流式输出：ask_stream 事件序、逐 token 增量、真实 usage 与轨迹回放。"""
from __future__ import annotations

import json
from pathlib import Path

from core.agentic.runtime import AgentRuntime
from core.agentic.trace import TraceRecorder
from core.pipeline import RagPipeline


# ==================== 基础装配 ====================
class FakeEmbedder:
    def embed_query(self, query: str) -> list[float]:
        return [0.1, 0.2, 0.3, 0.4]


class StreamStore:
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
                "text": "BM25 是稀疏检索算法，擅长关键词精确匹配。",
                "source": "doc_b.md",
                "page": 1,
                "chunk_idx": 0,
            },
        ]

    def all_chunks(self, max_rows: int = 100000, wait_visible: bool = False) -> list[dict]:
        return [dict(r) for r in self.rows]

    def search_dense(
        self, query_vec: list[float], top_k: int = 20, filter_expr: str = ""
    ) -> list[dict]:
        hits = []
        for i, row in enumerate(self.rows):
            hit = dict(row)
            hit["dense_score"] = round(1.0 - i * 0.1, 6)
            hits.append(hit)
        return hits[:top_k]


class FakeReranker:
    enabled = False


class StreamAnswer:
    """只会流式输出的回答模型。"""

    def __init__(self, pieces: list[str]):
        self.pieces = pieces
        self.chat_called = False

    def chat(self, messages):
        self.chat_called = True
        raise AssertionError("流式路径不应调用 chat()")

    def chat_stream(self, messages):
        yield from self.pieces


class UsageStreamAnswer(StreamAnswer):
    """模拟服务端在流末尾返回真实 usage。"""

    def chat_stream(self, messages, usage=None):
        for piece in self.pieces:
            yield piece
        if usage is not None:
            usage.update({"prompt_tokens": 11, "completion_tokens": 7})


class PlainAnswer:
    """不支持流式的回答模型，用于验证单条 delta 兜底。"""

    def __init__(self, text: str):
        self.text = text

    def chat(self, messages):
        return self.text


class CiteBrain:
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
            return json.dumps(
                {
                    "support": [{"sub_query": "q", "score": 1}] * 3,
                    "missing_aspects": [],
                    "conflicts": [],
                },
                ensure_ascii=False,
            )
        if "引用校验器" in system:
            return json.dumps(
                {"claims": [{"index": 1, "supported": True, "reason": "证据支持"}]},
                ensure_ascii=False,
            )
        return "{}"


def _runtime(tmp_path: Path, answer_client) -> AgentRuntime:
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
            "citations": {
                "enabled": True,
                "min_coverage": 0.60,
                "verify_claims": True,
                "tau_answer_confidence": 0.55,
            },
            "trace": {"enabled": True, "dir": str(tmp_path)},
        },
    }
    pipeline = RagPipeline(
        cfg,
        store=StreamStore(),
        embedder=FakeEmbedder(),
        llm=answer_client,
        reranker=FakeReranker(),
    )
    llms = {"cloud": {"client": CiteBrain()}, "ft": {"client": answer_client}}
    return AgentRuntime(cfg, pipeline, llms)


def test_ask_stream_emits_tokens_and_replayable_trace(tmp_path: Path):
    answer = StreamAnswer(["Milvus ", "支持 ", "HNSW 索引 [E1]。"])
    runtime = _runtime(tmp_path, answer)

    events = list(runtime.ask_stream("Milvus 支持什么索引", session_id="st-a", llm="ft"))
    types = [e["type"] for e in events]

    assert types[0] == "session"
    assert types[-1] == "done"
    assert types.count("delta") == 3
    assert types.index("delta") < types.index("refs") < types.index("done")
    assert answer.chat_called is False

    text = "".join(e["text"] for e in events if e["type"] == "delta")
    assert text == "Milvus 支持 HNSW 索引 [E1]。"

    session = events[0]
    done = events[-1]
    assert done["answer"] == text
    assert done["status"] == "answered"
    assert done["session_id"] == "st-a"
    assert done["trace_id"] == session["trace_id"]

    stages = {e["stage"] for e in events if e["type"] == "trace"}
    assert {"analyze", "retrieve", "compose", "verify", "finalize"}.issubset(stages)

    # 轨迹可回放：落盘文件包含完整事件与分账
    path = tmp_path / f"{done['trace_id']}.json"
    assert path.is_file()
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    assert snapshot["trace_id"] == done["trace_id"]
    assert snapshot["status"] == "answered"
    assert snapshot["usage"]["answer"]["calls"] == 1
    assert any(ev["stage"] == "finalize" for ev in snapshot["events"])


def test_ask_stream_falls_back_to_single_delta_without_stream_support(tmp_path: Path):
    runtime = _runtime(tmp_path, PlainAnswer("Milvus 支持 HNSW 索引 [E1]。"))

    events = list(runtime.ask_stream("Milvus 支持什么索引", session_id="st-b", llm="ft"))
    deltas = [e["text"] for e in events if e["type"] == "delta"]

    assert deltas == ["Milvus 支持 HNSW 索引 [E1]。"]
    assert events[-1]["type"] == "done"
    assert events[-1]["answer"] == deltas[0]


def test_ask_stream_uses_real_usage_when_provider_returns_it(tmp_path: Path):
    runtime = _runtime(tmp_path, UsageStreamAnswer(["Milvus 支持 HNSW 索引 [E1]。"]))

    events = list(runtime.ask_stream("Milvus 支持什么索引", session_id="st-c", llm="ft"))
    metrics = events[-1]["metrics"]

    # answer 侧至少包含服务端返回的 11/7，brain 侧仍为估算
    assert metrics["prompt_tokens"] >= 11
    assert metrics["completion_tokens"] >= 7


def test_trace_recorder_on_event_works_when_persistence_disabled():
    seen: list[dict] = []
    recorder = TraceRecorder(enabled=False, on_event=seen.append)

    recorder.emit("retrieve", payload={"round": 1})
    recorder.record_llm("answer", [{"role": "user", "content": "hi"}], "ok", 12)

    assert recorder.events == []
    assert [e["stage"] for e in seen] == ["retrieve", "llm"]
