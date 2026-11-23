"""缺口改写：brain 输出校验、去重、规则降级。"""
from __future__ import annotations

import json

from core.agentic.rewrite import Rewriter
from core.agentic.state import SubQuery, Sufficiency


class Brain:
    def __init__(self, reply):
        self.reply = reply
        self.calls = 0

    def chat(self, messages):
        self.calls += 1
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def _suff(missing=("索引参数",), score=0.3):
    return Sufficiency(score=score, missing_aspects=list(missing))


def _plan():
    return [SubQuery(text="Milvus 支持什么索引", source="doc_a.md", page=2)]


def _rewrite(brain, plan=None, question="Milvus 支持什么索引", missing=("索引参数",)):
    return Rewriter(brain).rewrite(
        question=question,
        plan=plan or _plan(),
        sufficiency=_suff(missing),
        contexts=[],
        sources=["doc_a.md"],
    )


def test_brain_rewrite_returns_new_queries():
    brain = Brain(
        json.dumps(
            {
                "sub_queries": [
                    {"text": "HNSW 的参数配置", "purpose": "补充参数"},
                    {"text": "Milvus 支持什么索引", "purpose": "重复"},
                ],
                "reason": "补充参数细节",
            },
            ensure_ascii=False,
        )
    )
    outcome = _rewrite(brain)
    assert outcome.source == "brain"
    assert [sq.text for sq in outcome.plan] == ["HNSW 的参数配置"]
    assert outcome.reason == "补充参数细节"
    assert outcome.llm_trace is not None


def test_duplicate_only_falls_back_to_rules():
    brain = Brain(
        json.dumps(
            {"sub_queries": [{"text": "Milvus 支持什么索引"}]},
            ensure_ascii=False,
        )
    )
    outcome = _rewrite(brain)
    assert outcome.source == "rules"
    assert outcome.plan[0].text == "索引参数"
    assert outcome.error == "empty_or_duplicate_plan"


def test_invalid_json_falls_back_to_missing_aspects():
    outcome = _rewrite(Brain("不是 JSON"))
    assert outcome.source == "rules"
    assert outcome.plan[0].text == "索引参数"
    assert outcome.plan[0].source == "doc_a.md"
    assert outcome.plan[0].page == 2


def test_no_missing_aspects_generates_generic_query():
    outcome = _rewrite(Brain("{}"), missing=())
    assert outcome.source == "rules"
    assert "具体细节" in outcome.plan[0].text


def test_brain_exception_falls_back():
    outcome = _rewrite(Brain(RuntimeError("down")))
    assert outcome.source == "rules"
    assert "llm_error" in outcome.error


def test_rewrite_caps_subquery_count():
    items = [{"text": f"补充问题 {i}"} for i in range(6)]
    brain = Brain(json.dumps({"sub_queries": items}, ensure_ascii=False))
    outcome = _rewrite(brain)
    assert len(outcome.plan) == 3


def test_rewrite_rejects_hallucinated_source():
    brain = Brain(
        json.dumps(
            {
                "sub_queries": [
                    {"text": "补充参数", "source": "fake.pdf", "page": 9}
                ]
            },
            ensure_ascii=False,
        )
    )
    outcome = _rewrite(brain)
    sq = outcome.plan[0]
    assert sq.source is None
    assert sq.page is None


def test_empty_plan_and_no_rule_candidates_returns_none():
    # 无缺失方面时规则会生成通用查询，因此这里通过已有计划占满去重路径
    brain = Brain(
        json.dumps(
            {"sub_queries": [{"text": "Milvus 支持什么索引"}]},
            ensure_ascii=False,
        )
    )
    outcome = Rewriter(brain).rewrite(
        question="Milvus 支持什么索引",
        plan=_plan(),
        sufficiency=_suff(missing=()),
        contexts=[],
    )
    # 通用规则查询与问题不同，仍应产生一个兜底查询
    assert outcome.plan
    assert outcome.source == "rules"
