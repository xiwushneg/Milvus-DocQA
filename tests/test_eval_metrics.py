"""B6 指标单测：EM/F1、拒答、引用、检索与 RAGAS 适配。"""
from __future__ import annotations

import sys

from evaluation.metrics import (
    aggregate_generation,
    citation_metrics,
    exact_match,
    f1_score,
    is_gold_hit,
    normalize_answer,
    ragas_scores,
    refusal_metrics,
    retrieval_metrics,
)


def test_normalize_em_and_f1_strip_citations():
    assert normalize_answer("  HNSW [E1] 索引。 ") == "hnsw 索引"
    assert exact_match("HNSW [E1]", "HNSW") == 1.0
    assert exact_match("Milvus 支持 HNSW", "Milvus 支持 IVF") == 0.0
    assert f1_score("Milvus 支持 HNSW 索引", "Milvus 支持 HNSW") == 0.8
    assert f1_score("完全无关", "Milvus") == 0.0
    assert f1_score("", "Milvus") == 0.0


def test_refusal_metrics_with_imbalance():
    rows = [
        {"pred_refused": True, "gold_refused": True},
        {"pred_refused": True, "gold_refused": False},
        {"pred_refused": False, "gold_refused": True},
        {"pred_refused": False, "gold_refused": False},
    ]
    m = refusal_metrics(rows)
    assert m["precision"] == 0.5
    assert m["recall"] == 0.5
    assert m["f1"] == 0.5
    assert m["accuracy"] == 0.5
    assert m["tp"] == 1 and m["fp"] == 1 and m["fn"] == 1 and m["tn"] == 1
    assert "programmatic" in m["evaluator"]


def test_citation_metrics_coverage_and_hallucination():
    rows = [
        {
            "citation_coverage": 1.0,
            "citations": [{"valid": True}, {"valid": True}],
            "claims": 2,
            "unsupported_claims": 0,
        },
        {
            "citation_coverage": 0.5,
            "citations": [{"valid": False}, {"valid": True}],
            "claims": 1,
            "unsupported_claims": 1,
        },
    ]
    m = citation_metrics(rows)
    assert m["citation_coverage"] == 0.75
    assert m["invalid_citation_rate"] == 0.25
    assert m["hallucination_rate"] == round(1 / 3, 4)
    assert m["total_citations"] == 4


def test_hallucination_rate_never_exceeds_one():
    rows = [{"claims": 1, "unsupported_claims": 3, "citations": []}]
    m = citation_metrics(rows)
    assert m["hallucination_rate"] == 1.0
    assert m["hallucination_rate"] <= 1.0


def test_retrieval_metrics_hit_rules():
    gold = [{"source": "doc_a", "must_contain": "HNSW", "page": 2}]
    hits = [
        {"source": "doc_b", "text": "HNSW", "page": 2},
        {"source": "doc_a", "text": "无关内容", "page": 2},
        {"source": "doc_a", "text": "支持 HNSW 索引", "page": 2},
    ]
    assert is_gold_hit(hits[0], gold) is False
    assert is_gold_hit(hits[1], gold) is False
    assert is_gold_hit(hits[2], gold) is True

    m = retrieval_metrics([hits], [gold], k=3)
    assert m["recall_at_k"] == 1.0
    assert m["mrr"] == round(1 / 3, 4)
    assert m["ndcg_at_k"] == round(1 / 2, 4)  # 1/log2(3+1)


def test_retrieval_ndcg_never_exceeds_one_with_multi_gold_or_duplicates():
    gold = [{"source": "doc_a", "must_contain": "HNSW"}, {"source": "doc_a", "must_contain": "图"}]
    hits = [
        {"source": "doc_a", "text": "HNSW 是图索引"},
        {"source": "doc_a", "text": "HNSW 图索引说明"},
    ]
    m = retrieval_metrics([hits], [gold], k=2)
    assert m["ndcg_at_k"] == 1.0
    assert m["ndcg_at_k"] <= 1.0


def test_aggregate_generation_combines_all_metrics():
    rows = [
        {
            "prediction": "Milvus 支持 HNSW [E1]。",
            "gold": "Milvus 支持 HNSW",
            "gold_refused": False,
            "pred_refused": False,
            "citation_coverage": 1.0,
            "citations": [{"valid": True}],
            "claims": 1,
            "unsupported_claims": 0,
            "latency_ms": 100,
            "prompt_tokens": 10,
            "completion_tokens": 5,
        },
        {
            "prediction": "无法可靠回答",
            "gold": "",
            "gold_refused": True,
            "pred_refused": True,
            "citation_coverage": 0.0,
            "citations": [],
            "latency_ms": 300,
        },
    ]
    m = aggregate_generation(rows)
    assert m["answerable_n"] == 1
    assert m["refusal"]["f1"] == 1.0
    assert m["avg_latency_ms"] == 200.0
    assert m["prompt_tokens"] == 10 and m["completion_tokens"] == 5
    assert m["evaluator"]["exact_match"].startswith("programmatic")


def test_ragas_adapter_degrades_when_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "ragas", None)  # 强制 ImportError
    out = ragas_scores([{"question": "q", "answer": "a", "contexts": ["c"]}])
    assert out["available"] is False
    assert out["reason"] == "ragas_not_installed"
    assert out["n"] == 1
