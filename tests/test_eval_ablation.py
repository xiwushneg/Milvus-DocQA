"""B6 消融实验单测：配置累积、行构造、汇总与基线读取。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.ablation_agentic import (
    EVALUATOR_LEGEND,
    _make_row,
    _run_e0,
    compare_e0_to_baseline,
    load_retrieval_baseline,
    run_ablation,
    variant_config,
)

_BASE = {
    "retrieval": {"final_top_k": 5},
    "agent": {
        "enabled": True,
        "max_rounds": 2,
        "hyde": {"enabled": True, "model": "keep-me"},
        "tools": {"enabled": True, "db_path": "keep.db"},
        "citations": {"enabled": True, "max_claims": 7},
    },
}


def test_variant_config_accumulates_without_mutating_base():
    e0 = variant_config(_BASE, "E0")
    assert e0["agent"]["enabled"] is False
    assert e0["retrieval"]["rerank"]["enabled"] is True
    # 未覆盖字段保留
    assert e0["agent"]["hyde"]["enabled"] is True

    e1 = variant_config(_BASE, "E1")
    assert e1["retrieval"]["rerank"]["enabled"] is True
    assert e1["agent"]["enabled"] is True
    assert e1["agent"]["max_rounds"] == 1
    assert e1["agent"]["hyde"]["enabled"] is False
    assert e1["agent"]["hyde"]["model"] == "keep-me"
    assert e1["agent"]["tools"]["enabled"] is False
    assert e1["agent"]["tools"]["db_path"] == "keep.db"
    assert e1["agent"]["citations"]["enabled"] is False

    e2 = variant_config(_BASE, "E2")
    assert e2["agent"]["max_rounds"] == 3
    assert e2["agent"]["hyde"]["enabled"] is False

    e3 = variant_config(_BASE, "E3")
    assert e3["agent"]["tools"]["enabled"] is True

    e4 = variant_config(_BASE, "E4")
    assert e4["agent"]["citations"]["enabled"] is True
    assert e4["retrieval"]["final_top_k"] == 5

    # 原始配置不得被修改
    assert _BASE["agent"]["max_rounds"] == 2
    assert _BASE["agent"]["hyde"]["enabled"] is True


def test_variant_config_rejects_unknown_variant():
    with pytest.raises(ValueError):
        variant_config(_BASE, "E9")


def test_make_row_reads_trace_and_metrics():
    out = {
        "answer": "HNSW 是图索引 [E1]。",
        "references": [{"source": "doc_a", "text": "HNSW", "page": 1}],
        "metrics": {"latency_ms": 12.5, "prompt_tokens": 4, "completion_tokens": 2},
        "trace": [
            {"stage": "retrieve", "payload": {}},
            {
                "stage": "verify",
                "payload": {
                    "citations": 2,
                    "unsupported_claims": ["臆造"],
                    "citation_coverage": 0.5,
                },
            },
        ],
        "citations": [{"valid": True}],
        "refused": False,
        "confidence": 0.88,
    }
    item = {
        "question": "HNSW 是什么",
        "gold": [{"source": "doc_a", "must_contain": "HNSW"}],
    }
    row = _make_row(item, out, agent=True)
    assert row["claims"] == 2
    assert row["unsupported_claims"] == 1
    assert row["citation_coverage"] == 0.5
    assert row["latency_ms"] == 12.5
    assert row["prompt_tokens"] == 4
    assert row["confidence"] == 0.88
    assert row["gold_refused"] is False
    assert row["pred_refused"] is False


def test_pred_refused_uses_heuristic_for_e0_and_runtime_flag_later():
    item = {"question": "q"}
    e0 = _make_row(item, {"answer": "资料不足，无法回答该问题"}, agent=False)
    assert e0["pred_refused"] is True
    # E0 忽略 runtime refused 字段，只看文本
    e0b = _make_row(item, {"answer": "HNSW 是图索引", "refused": True}, agent=False)
    assert e0b["pred_refused"] is False
    # E1+ 只看 runtime refused
    e1 = _make_row(item, {"answer": "HNSW 是图索引", "refused": True}, agent=True)
    assert e1["pred_refused"] is True


def test_pred_refused_counts_empty_answer_for_both_variants():
    item = {"question": "q"}
    assert _make_row(item, {"answer": ""}, agent=False)["pred_refused"] is True
    assert _make_row(item, {"answer": "   "}, agent=True)["pred_refused"] is True
    assert _make_row(item, {"answer": None, "refused": False}, agent=True)["pred_refused"] is True


def _fake_factory(name: str, cfg: dict):
    def runner(question: str) -> dict:
        if "作者" in question:
            return {
                "answer": "无法可靠回答",
                "references": [],
                "metrics": {"latency_ms": 5, "stop_reason": "refuse"},
                "trace": [{"stage": "verify", "payload": {"citations": 0, "unsupported_claims": []}}],
                "citations": [],
                "refused": True,
                "confidence": 0.2,
            }
        return {
            "answer": "HNSW 是图索引 [E1]",
            "references": [{"source": "doc_a", "text": "HNSW 是图索引", "page": 1}],
            "metrics": {
                "latency_ms": 15,
                "prompt_tokens": 3,
                "completion_tokens": 2,
                "stop_reason": "sufficient",
                "citation_coverage": 1.0,
            },
            "trace": [
                {"stage": "verify", "payload": {"citations": 1, "unsupported_claims": []}}
            ],
            "citations": [{"valid": True}],
            "refused": False,
            "confidence": 0.9,
        }

    return runner


_BASELINE = {
    "available": True,
    "source": "evaluation/results_rerank_ablation.json",
    "results": [
        {"strategy": "dense", "recall_at_k": 0.9, "mrr": 0.789},
        {"strategy": "hybrid+rerank", "recall_at_k": 0.95, "mrr": 0.91},
    ],
}


def _variant(name: str, recall: float, mrr: float) -> dict:
    return {"variant": name, "metrics": {"retrieval": {"recall_at_k": recall, "mrr": mrr}}}


def test_compare_e0_to_baseline_accepts_tolerance():
    check = compare_e0_to_baseline([_variant("E0", 0.95, 0.9104)], _BASELINE)
    assert check["performed"] is True
    assert check["passed"] is True
    assert check["expected"] == {"recall_at_k": 0.95, "mrr": 0.91}


def test_compare_e0_to_baseline_detects_drift():
    check = compare_e0_to_baseline([_variant("E0", 1.0, 0.892)], _BASELINE)
    assert check["performed"] is True
    assert check["passed"] is False
    assert check["actual"] == {"recall_at_k": 1.0, "mrr": 0.892}


def test_compare_e0_to_baseline_requires_hybrid_rerank_row():
    check = compare_e0_to_baseline(
        [_variant("E0", 0.95, 0.91)],
        {"available": True, "source": "x.json", "results": []},
    )
    assert check["performed"] is False
    assert check["passed"] is None
    assert "hybrid+rerank" in check["reason"]


def test_run_ablation_with_injected_runners(tmp_path):
    items = [
        {
            "question": "HNSW 是什么",
            "gold": [{"source": "doc_a", "must_contain": "HNSW"}],
            "answer": "HNSW 是图索引",
        },
        {"question": "这篇文档的作者是谁", "refusal_expected": True},
    ]
    missing_baseline = tmp_path / "missing.json"
    result = run_ablation(
        items,
        _BASE,
        variants=["E0", "E1"],
        runner_factory=_fake_factory,
        baseline_path=missing_baseline,
    )
    assert result["eval_set_n"] == 2
    assert result["top_k"] == 5
    assert [v["variant"] for v in result["variants"]] == ["E0", "E1"]
    assert result["retrieval_baseline"]["available"] is False
    assert result["baseline_check"]["performed"] is False
    assert result["evaluator_legend"] == EVALUATOR_LEGEND

    for variant in result["variants"]:
        m = variant["metrics"]
        assert m["answerable_n"] == 1
        assert m["exact_match"] == 1.0
        assert m["refusal"]["f1"] == 1.0
        assert m["retrieval"]["recall_at_k"] == 1.0
        assert m["citations"]["citation_coverage"] == 1.0
        assert m["avg_confidence"] > 0
        assert m["stop_reasons"] == {"refuse": 1, "sufficient": 1}


def test_run_ablation_requires_runner_factory():
    with pytest.raises(ValueError):
        run_ablation([{"question": "q"}], _BASE, variants=["E0"])


def test_run_ablation_verify_baseline_aborts_when_unavailable(tmp_path):
    items = [
        {
            "question": "HNSW 是什么",
            "gold": [{"source": "doc_a", "must_contain": "HNSW"}],
            "answer": "HNSW 是图索引",
        }
    ]
    with pytest.raises(RuntimeError, match="E0"):
        run_ablation(
            items,
            _BASE,
            variants=["E0"],
            runner_factory=_fake_factory,
            baseline_path=tmp_path / "missing.json",
            verify_baseline=True,
        )


def test_run_ablation_verify_baseline_aborts_on_drift(tmp_path):
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(
        json.dumps(
            {
                "top_k": 5,
                "num_questions": 20,
                "results": [
                    {"strategy": "hybrid+rerank", "recall_at_k": 0.95, "mrr": 0.91}
                ],
            }
        ),
        encoding="utf-8",
    )
    items = [
        {
            "question": "HNSW 是什么",
            "gold": [{"source": "doc_a", "must_contain": "HNSW"}],
            "answer": "HNSW 是图索引",
        }
    ]
    with pytest.raises(RuntimeError, match="未复现"):
        run_ablation(
            items,
            _BASE,
            variants=["E0"],
            runner_factory=_fake_factory,
            baseline_path=baseline_path,
            verify_baseline=True,
        )


def test_load_retrieval_baseline_reads_rerank_ablation_file():
    path = Path(__file__).resolve().parents[1] / "evaluation/results_rerank_ablation.json"
    baseline = load_retrieval_baseline(path)
    assert baseline["available"] is True
    assert baseline["num_questions"] == 49
    assert baseline["top_k"] == 5
    strategies = {r["strategy"]: r for r in baseline["results"]}
    assert strategies["dense"]["recall_at_k"] == 1.0
    assert strategies["hybrid(RRF)"]["recall_at_k"] == 1.0
    assert strategies["hybrid+rerank"]["mrr"] == 0.939
    assert "eval_retrieval" in baseline["evaluator"]


def test_run_e0_forwards_top_k():
    class FakePipeline:
        def __init__(self):
            self.calls = []

        def ask(self, question, top_k=None, llm=None):
            self.calls.append((question, top_k))
            return {"answer": "a", "references": [], "total_ms": 1}

    pipeline = FakePipeline()
    row = _run_e0(pipeline, "问题", None, top_k=7)
    assert pipeline.calls == [("问题", 7)]
    assert row["metrics"]["latency_ms"] == 1


def test_run_ablation_allows_subset_without_e0(tmp_path):
    items = [
        {
            "question": "HNSW 是什么",
            "gold": [{"source": "doc_a", "must_contain": "HNSW"}],
            "answer": "HNSW 是图索引",
        }
    ]
    result = run_ablation(
        items,
        _BASE,
        variants=["E1"],
        runner_factory=_fake_factory,
        baseline_path=tmp_path / "missing.json",
        verify_baseline=True,
    )
    assert result["baseline_check"]["performed"] is False
    assert result["baseline_check"]["skipped"] is True
