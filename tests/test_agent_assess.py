"""充分性评估：特征计算、阈值决策与降级路径。"""
from __future__ import annotations

import json

from core.agentic.assess import Assessor
from core.agentic.state import Decision, SubQuery


class Brain:
    def __init__(self, reply):
        self.reply = reply
        self.calls = 0

    def chat(self, messages):
        self.calls += 1
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def _cfg(tau_low=0.45, tau_high=0.65):
    return {
        "agent": {
            "sufficiency": {
                "tau_low": tau_low,
                "tau_high": tau_high,
                "weights": {
                    "claim_support": 0.45,
                    "coverage": 0.25,
                    "top_score": 0.20,
                    "diversity": 0.10,
                },
            },
            "assess_max_evidence": 6,
            "assess_max_chars": 300,
        },
        "retrieval": {"rrf_k": 60},
    }


def _context(text="Milvus 支持 HNSW 索引，维度为 1024。", source="doc_a.md", rrf=1 / 61):
    return {
        "evidence_id": "E1",
        "text": text,
        "source": source,
        "page": 1,
        "rrf_score": rrf,
        "rerank_score": None,
    }


def test_sufficient_with_full_support_and_coverage():
    brain = Brain(
        json.dumps(
            {
                "support": [{"sub_query": "q", "score": 1}],
                "missing_aspects": [],
                "conflicts": [],
            }
        )
    )
    outcome = Assessor(brain, _cfg()).assess(
        question="Milvus 支持什么索引？",
        sub_queries=[SubQuery(text="Milvus 支持的索引类型")],
        contexts=[_context()],
        novelty=1.0,
    )
    assert outcome.sufficiency.decision == Decision.SUFFICIENT
    assert outcome.sufficiency.score >= 0.65
    assert outcome.sufficiency.novelty == 1.0
    assert not outcome.degraded


def test_insufficient_when_support_zero():
    brain = Brain(
        json.dumps(
            {
                "support": [{"sub_query": "q", "score": 0}],
                "missing_aspects": ["索引参数"],
                "conflicts": [],
            }
        )
    )
    outcome = Assessor(brain, _cfg()).assess(
        question="索引参数是多少？",
        sub_queries=[SubQuery(text="索引参数")],
        contexts=[_context("无关内容")],
        novelty=0.5,
    )
    assert outcome.sufficiency.decision == Decision.INSUFFICIENT
    assert outcome.sufficiency.missing_aspects == ["索引参数"]
    assert outcome.sufficiency.score < 0.65


def test_conflict_when_low_score_and_conflicting_evidence():
    brain = Brain(
        json.dumps(
            {
                "support": [{"sub_query": "q", "score": 0}],
                "missing_aspects": [],
                "conflicts": [
                    {"claim": "维度为 1024 或 768", "evidence_ids": ["E1", "E2"]}
                ],
            }
        )
    )
    outcome = Assessor(brain, _cfg(tau_low=0.5)).assess(
        question="维度是多少？",
        sub_queries=[SubQuery(text="向量维度")],
        contexts=[_context("维度是 1024"), _context("维度是 768", rrf=1 / 62)],
        novelty=0.2,
    )
    assert outcome.sufficiency.decision == Decision.CONFLICT
    assert outcome.sufficiency.conflicts[0]["evidence_ids"] == ["E1", "E2"]


def test_brain_failure_degrades_to_rules():
    outcome = Assessor(Brain(RuntimeError("down")), _cfg()).assess(
        question="Milvus 支持什么索引？",
        sub_queries=[SubQuery(text="完全不相关的方面")],
        contexts=[_context()],
        novelty=1.0,
    )
    assert outcome.degraded is True
    assert "llm_error" in outcome.error
    assert outcome.sufficiency.claim_support == outcome.sufficiency.coverage
    assert outcome.sufficiency.decision in {
        Decision.INSUFFICIENT,
        Decision.SUFFICIENT,
    }


def test_invalid_json_degrades():
    outcome = Assessor(Brain("不是 JSON"), _cfg()).assess(
        question="问题",
        sub_queries=[SubQuery(text="Milvus 索引")],
        contexts=[_context()],
        novelty=1.0,
    )
    assert outcome.degraded is True
    assert outcome.error == "json_parse_error"


def test_top_score_rrf_normalization():
    brain = Brain("{}")
    outcome = Assessor(brain, _cfg()).assess(
        question="q",
        sub_queries=[SubQuery(text="Milvus")],
        contexts=[_context(rrf=2 / 61)],
        novelty=1.0,
    )
    assert outcome.sufficiency.top_score == 1.0


def test_diversity_uses_distinct_sources():
    brain = Brain("{}")
    contexts = [
        _context("a", source="doc_a.md", rrf=1 / 61),
        _context("b", source="doc_b.md", rrf=1 / 62),
    ]
    outcome = Assessor(brain, _cfg()).assess(
        question="q",
        sub_queries=[SubQuery(text="对比 A 和 B")],
        contexts=contexts,
        novelty=1.0,
    )
    assert outcome.sufficiency.diversity == 1.0


def test_no_brain_marks_degraded():
    outcome = Assessor(None, _cfg()).assess(
        question="q",
        sub_queries=[SubQuery(text="Milvus")],
        contexts=[_context()],
        novelty=1.0,
    )
    assert outcome.degraded is True
    assert outcome.error == "no_brain"
