"""B4 可信输出：逐句引用、非法引用、冲突并列与低置信拒答。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.agentic.assess import AssessOutcome
from core.agentic.compose import cited_messages
from core.agentic.runtime import AgentRuntime
from core.agentic.state import Decision, Status, Sufficiency
from core.agentic.verify import (
    Verifier,
    build_verifier,
    extract_citations,
    load_calibration,
    split_sentences,
)
from core.pipeline import RagPipeline
from evaluation.calibrate_thresholds import calibrate
from evaluation.export_dev_confidence import _raw_confidence

CONTEXTS = [
    {"evidence_id": "E1", "text": "Milvus 支持 HNSW 索引。", "source": "doc_a.md", "page": 1},
    {"evidence_id": "E2", "text": "BM25 依赖词频统计。", "source": "doc_b.md", "page": 1},
]


# ==================== 单元：切句与引用提取 ====================
def test_split_sentences_and_extract_citations():
    sentences = split_sentences(
        "## 结论\nMilvus 支持 HNSW 索引 [E1]。BM25 依赖词频 [E2][T1]！\n- 短\n"
    )
    # 过短的“结论”被过滤；句末标点保留在句内
    assert sentences == [
        "Milvus 支持 HNSW 索引 [E1]。",
        "BM25 依赖词频 [E2][T1]！",
    ]
    assert extract_citations(sentences[0]) == ["E1"]
    assert extract_citations(sentences[1]) == ["E2", "T1"]


# ==================== 单元：Verifier ====================
def test_verifier_valid_citations_get_high_confidence():
    verifier = Verifier(brain=None, verify_claims=False)
    outcome = verifier.verify(
        answer="Milvus 支持 HNSW 索引 [E1]。BM25 依赖词频 [E2]。",
        contexts=CONTEXTS,
        sufficiency=Sufficiency(score=1.0),
    )
    assert outcome.citation_coverage == 1.0
    assert outcome.confidence == pytest.approx(1.0)
    assert outcome.invalid_citations == []
    assert all(c["valid"] for c in outcome.citations)


def test_verifier_invalid_and_low_coverage_force_refusal():
    verifier = Verifier(brain=None, verify_claims=False, min_coverage=0.6, tau=0.55)
    outcome = verifier.verify(
        answer="Milvus 支持 HNSW 索引 [E1]。这里引用了不存在的来源 [E9]。",
        contexts=CONTEXTS,
        sufficiency=Sufficiency(score=0.9),
    )
    assert outcome.citation_coverage == 0.5
    assert outcome.invalid_citations == ["E9"]
    assert outcome.citations[1]["valid"] is False
    assert outcome.confidence < 0.55


def test_verifier_tool_output_counts_as_t_evidence():
    outcome = Verifier(brain=None, verify_claims=False).verify(
        answer="计算结果为 22.25 [T1]。",
        contexts=[],
        tool_results=[{"ok": True, "tool": "calculator", "output": "22.25"}],
    )
    assert outcome.citation_coverage == 1.0
    assert outcome.invalid_citations == []
    assert outcome.confidence >= 0.55


def test_verifier_tool_numbering_skips_failed_results():
    """compose 只展示成功工具；verify 的 T 编号必须与之一致。"""
    outcome = Verifier(brain=None, verify_claims=False).verify(
        answer="计算结果为 22.25 [T1]。",
        contexts=[],
        tool_results=[
            {"ok": False, "tool": "calculator", "output": "", "error": "boom"},
            {"ok": True, "tool": "calculator", "output": "22.25"},
        ],
    )
    assert outcome.citation_coverage == 1.0
    assert outcome.invalid_citations == []
    assert outcome.confidence >= 0.55


def test_verifier_claim_check_flags_unsupported_claim():
    class ClaimBrain:
        def chat(self, messages):
            assert "引用校验器" in messages[0]["content"]
            return json.dumps(
                {"claims": [{"index": 1, "supported": False, "reason": "证据不足"}]},
                ensure_ascii=False,
            )

    verifier = Verifier(brain=ClaimBrain(), min_coverage=0.6, tau=0.55)
    outcome = verifier.verify(
        answer="Milvus 支持 HNSW 索引 [E1]。",
        contexts=CONTEXTS,
        sufficiency=Sufficiency(score=0.8),
    )
    assert outcome.claim_support == 0.0
    assert outcome.degraded is False
    assert outcome.unsupported_claims
    assert outcome.llm_trace is not None


def test_verifier_degrades_when_claim_brain_fails():
    class BrokenBrain:
        def chat(self, messages):
            raise RuntimeError("network down")

    outcome = Verifier(brain=BrokenBrain()).verify(
        answer="Milvus 支持 HNSW 索引 [E1]。",
        contexts=CONTEXTS,
        sufficiency=Sufficiency(score=0.8),
    )
    assert outcome.degraded is True
    assert "llm_error" in outcome.error
    assert outcome.claim_support == outcome.citation_coverage


def test_conflicts_reduce_confidence_but_do_not_force_refusal():
    verifier = Verifier(brain=None, verify_claims=False)
    common = {
        "answer": "Milvus 支持 HNSW 索引 [E1]。",
        "contexts": CONTEXTS,
        "sufficiency": Sufficiency(score=1.0),
    }
    without = verifier.verify(**common)
    with_conflict = verifier.verify(
        **common, conflicts=[{"claim": "版本存在两种说法", "evidence_ids": ["E1", "E2"]}]
    )
    assert without.confidence - with_conflict.confidence == pytest.approx(0.10)
    assert with_conflict.confidence >= verifier.tau


def test_verifier_direct_answer_skips_checks():
    outcome = Verifier(brain=None).verify(
        answer="你好，我是文档问答助手。", contexts=[], direct=True
    )
    assert outcome.confidence == 0.9
    assert outcome.citation_coverage == 1.0
    assert outcome.invalid_citations == []


# ==================== 引用提示与冲突并列 ====================
def test_cited_messages_use_evidence_ids_and_conflict_block():
    messages = cited_messages(
        "维度是多少",
        [
            {"evidence_id": "E1", "source": "doc_a.md", "page": 1, "text": "维度为 1024。"},
            {"evidence_id": "E2", "source": "doc_b.md", "page": 2, "text": "维度为 768。"},
        ],
        conflicts=[{"claim": "维度存在两种说法", "evidence_ids": ["E1", "E2"]}],
    )
    body = messages[-1]["content"]
    assert "[E1] （来源: doc_a.md 第1页）" in body
    assert "[E2] （来源: doc_b.md 第2页）" in body
    assert "【冲突提示】" in body
    assert "并列展示" in body
    assert "[E1]" in messages[0]["content"]  # system prompt 要求逐句引用


# ==================== 端到端：runtime verify 分支 ====================
class FakeEmbedder:
    def embed_query(self, query: str) -> list[float]:
        return [0.1, 0.2, 0.3, 0.4]


class CiteStore:
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


class AnswerClient:
    def __init__(self, answer: str):
        self.answer = answer
        self.messages = None

    def chat(self, messages):
        self.messages = messages
        return self.answer


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


class ConflictAssessor:
    """强制 CONFLICT 决策，用于验证冲突直接进入并列回答。"""

    def assess(self, *, question, sub_queries, contexts, novelty):
        return AssessOutcome(
            sufficiency=Sufficiency(
                score=0.3,
                coverage=1.0,
                claim_support=0.2,
                decision=Decision.CONFLICT,
                conflicts=[
                    {
                        "claim": "维度存在 1024 与 768 两种说法",
                        "evidence_ids": ["E1", "E2"],
                    }
                ],
            )
        )


def _runtime(tmp_path: Path, answer_text: str) -> tuple[AgentRuntime, AnswerClient]:
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
    answer = AnswerClient(answer_text)
    pipeline = RagPipeline(
        cfg,
        store=CiteStore(),
        embedder=FakeEmbedder(),
        llm=answer,
        reranker=FakeReranker(),
    )
    brain = CiteBrain()
    llms = {"cloud": {"client": brain}, "ft": {"client": answer}}
    return AgentRuntime(cfg, pipeline, llms), answer


def _events(out: dict, stage: str) -> list[dict]:
    return [e for e in out["trace"] if e["stage"] == stage]


def test_runtime_valid_citations_answer_with_verify_trace(tmp_path: Path):
    runtime, answer = _runtime(
        tmp_path, "Milvus 支持 HNSW 索引 [E1]，属于向量检索 [E2]。"
    )
    out = runtime.ask("Milvus 支持什么索引", session_id="cit-a", llm="ft", include_trace=True)

    assert out["status"] == "answered"
    assert out["refused"] is False
    assert out["metrics"]["citation_coverage"] == 1.0
    assert out["metrics"]["invalid_citations"] == 0
    assert out["metrics"]["confidence"] >= 0.55
    assert out["citations"] and all(c["valid"] for c in out["citations"])
    verify = _events(out, "verify")[0]["payload"]
    assert verify["citation_coverage"] == 1.0
    # 逐句引用提示已生效
    assert "【参考资料】" in answer.messages[-1]["content"]
    assert "[E1]" in answer.messages[0]["content"]


def test_runtime_invalid_citation_triggers_low_confidence_refusal(tmp_path: Path):
    runtime, _ = _runtime(tmp_path, "Milvus 支持 HNSW 索引 [E99]。")
    out = runtime.ask("Milvus 支持什么索引", session_id="cit-b", llm="ft", include_trace=True)

    assert out["refused"] is True
    assert out["refusal_reason"] == "low_confidence"
    assert out["status"] == "refused"
    assert out["metrics"]["invalid_citations"] == 1
    assert out["metrics"]["citation_coverage"] == 0.0
    assert out["citations"] == []  # 拒答时清空引用
    assert "无法可靠回答" in out["answer"]


def test_runtime_missing_citation_triggers_refusal(tmp_path: Path):
    runtime, _ = _runtime(tmp_path, "Milvus 支持 HNSW 索引。")
    out = runtime.ask("Milvus 支持什么索引", session_id="cit-c", llm="ft", include_trace=True)

    assert out["refused"] is True
    assert out["refusal_reason"] == "low_confidence"
    assert out["metrics"]["citation_coverage"] == 0.0


def test_runtime_conflict_answers_in_parallel_without_refusal(tmp_path: Path):
    runtime, answer = _runtime(
        tmp_path, "一种说法是维度 1024 [E1]，另一种说法是 768 [E2]。"
    )
    runtime.assessor = ConflictAssessor()

    out = runtime.ask("维度是多少", session_id="cit-d", llm="ft", include_trace=True)

    assert out["status"] == "answered"
    assert out["refused"] is False
    assert out["metrics"]["rounds"] == 1
    assert out["metrics"]["rewrites"] == 0
    assert out["metrics"]["stop_reason"] == "conflict"
    assert out["conflicts"][0]["evidence_ids"] == ["E1", "E2"]
    assert _events(out, "assess")[0]["payload"]["decision"] == "conflict"
    assert "【冲突提示】" in answer.messages[-1]["content"]
    assert out["metrics"]["confidence"] >= 0.55


# ==================== 阈值标定 ====================
def test_load_calibration_overrides_verifier_thresholds(tmp_path: Path):
    path = tmp_path / "calibration.json"
    path.write_text(
        json.dumps({"min_coverage": 0.7, "tau_answer_confidence": 0.62}),
        encoding="utf-8",
    )
    cfg = {
        "agent": {
            "citations": {
                "enabled": True,
                "min_coverage": 0.60,
                "tau_answer_confidence": 0.55,
                "calibration_path": str(path),
            }
        }
    }
    verifier = build_verifier(cfg, brain=None)
    assert verifier.min_coverage == 0.7
    assert verifier.tau == 0.62
    assert load_calibration(tmp_path / "missing.json") == {}
    assert load_calibration("") == {}


def test_calibrate_picks_threshold_with_best_refusal_f1():
    pairs = [
        {"confidence": 0.90, "answerable": True},
        {"confidence": 0.80, "answerable": True},
        {"confidence": 0.30, "answerable": False},
        {"confidence": 0.20, "answerable": False},
    ]
    result = calibrate(pairs, grid_step=0.01, max_false_refusal=0.0)
    assert 0.30 < result["tau_answer_confidence"] <= 0.80
    assert result["refusal_f1"] == 1.0
    assert result["false_refusal_rate"] == 0.0
    assert result["n"] == 4


def test_calibrate_rejects_empty_dataset():
    with pytest.raises(ValueError):
        calibrate([])


def test_calibrate_joint_searches_min_coverage():
    pairs = [
        {"confidence": 0.90, "raw_confidence": 0.90, "coverage": 0.90, "answerable": True},
        {"confidence": 0.80, "raw_confidence": 0.80, "coverage": 0.80, "answerable": True},
        {"confidence": 0.89, "raw_confidence": 0.89, "coverage": 0.20, "answerable": False},
        {"confidence": 0.86, "raw_confidence": 0.86, "coverage": 0.10, "answerable": False},
    ]
    result = calibrate(pairs, grid_step=0.01, max_false_refusal=0.0)

    assert result["mode"] == "joint"
    assert result["min_coverage"] > 0.2, "低覆盖拒答题必须通过 min_coverage 钳位"
    assert result["refusal_f1"] == 1.0
    assert result["false_refusal_rate"] == 0.0


def test_calibrate_tau_only_has_no_min_coverage():
    result = calibrate(
        [
            {"confidence": 0.90, "answerable": True},
            {"confidence": 0.20, "answerable": False},
        ],
        grid_step=0.01,
        max_false_refusal=0.0,
    )
    assert result["mode"] == "tau_only"
    assert "min_coverage" not in result


# ==================== 标定数据导出：原始置信度 ====================
def test_raw_confidence_recovers_value_before_refuse_zeroing():
    out = {
        "confidence": 0.0,  # refuse 节点清零后的返回值
        "trace": [
            {"stage": "compose", "payload": {"confidence": 0.5}},
            {"stage": "verify", "payload": {"confidence": 0.42}},
            {"stage": "refuse", "payload": {"reason": "low_confidence"}},
        ],
    }
    assert _raw_confidence(out) == pytest.approx(0.42)


def test_raw_confidence_prefers_unclamped_field():
    out = {
        "confidence": 0.0,
        "trace": [
            {
                "stage": "verify",
                "payload": {
                    "confidence": 0.29,
                    "raw_confidence": 0.71,
                    "citation_coverage": 0.29,
                },
            }
        ],
    }
    assert _raw_confidence(out) == pytest.approx(0.71)


def test_raw_confidence_keeps_zero_for_early_refusal():
    out = {
        "confidence": 0.0,
        "trace": [{"stage": "refuse", "payload": {"reason": "out_of_scope"}}],
    }
    assert _raw_confidence(out) == 0.0
