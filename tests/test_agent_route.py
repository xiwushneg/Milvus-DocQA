"""路由决策测试：通道选择、过滤、HyDE 轮次与证据预算。"""
from __future__ import annotations

from core.agentic.route import (
    query_features,
    route_subquery,
    should_use_hyde,
)
from core.agentic.state import Intent, SubQuery


def _route(text: str, **kwargs):
    kwargs.setdefault("intent", Intent.FACT)
    kwargs.setdefault("round_no", 1)
    return route_subquery(SubQuery(text=text), **kwargs)


def test_default_uses_hybrid_channels():
    decision = _route("这份文档主要讲了什么")
    assert decision.channels == ("dense", "bm25")
    assert decision.fallback_channels == ()
    assert decision.reason == "default"


def test_proper_noun_prefers_bm25_with_dense_fallback():
    decision = _route("HNSW 索引是什么")
    assert decision.channels == ("bm25",)
    assert decision.fallback_channels == ("dense",)
    assert decision.reason == "keyword_bm25_first"


def test_version_and_code_prefer_bm25():
    assert _route("升级到 Milvus 2.4 要注意什么").channels == ("bm25",)
    assert _route("如何使用 SELECT 查询").channels == ("bm25",)


def test_filtered_query_keeps_hybrid_and_reason():
    decision = route_subquery(
        SubQuery(text="RRF 是什么", source="doc_a.md", page=2),
        intent=Intent.FACT,
        round_no=1,
    )
    assert decision.channels == ("dense", "bm25")
    assert decision.reason == "filtered"


def test_chitchat_has_no_channels():
    decision = _route("你好", intent=Intent.CHITCHAT)
    assert decision.channels == ()
    assert decision.reason == "no_retrieval"


def test_summary_gets_evidence_bonus():
    decision = _route("总结这份文档", intent=Intent.SUMMARY)
    assert decision.evidence_bonus == 2


def test_hyde_respects_start_round():
    cfg = {"enabled": True, "start_round": 2}
    assert should_use_hyde(1, cfg) is False
    assert should_use_hyde(2, cfg) is True
    assert should_use_hyde(3, {"enabled": False}) is False
    assert _route("HNSW 索引是什么", round_no=2, hyde_cfg=cfg).use_hyde is True


def test_keyword_switch_off_restores_hybrid():
    decision = _route("HNSW 索引是什么", keyword_bm25_first=False)
    assert decision.channels == ("dense", "bm25")
    assert decision.reason == "default"


def test_query_features_detect_aggregate_and_calculation():
    features = query_features("对比两款模型并计算平均值 3.5 + 2.5")
    assert features["wants_aggregate"] is True
    assert features["wants_calculation"] is True


def test_route_flags_table_and_calculation():
    decision = _route("汇总表中各类别的数量并计算平均值")
    assert decision.needs_table is True
    assert decision.needs_calculation is True
