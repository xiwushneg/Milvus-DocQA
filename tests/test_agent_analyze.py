"""brain 结构化分析与降级路径测试。"""
from __future__ import annotations

import json

import pytest

from core.agentic.analyze import (
    BrainAnalyzer,
    HydeGenerator,
    RuleAnalyzer,
    extract_filters,
)
from core.agentic.state import Intent


class FakeBrain:
    def __init__(self, replies: list):
        self.replies = list(replies)
        self.calls: list[list[dict]] = []

    def chat(self, messages: list[dict]) -> str:
        self.calls.append(messages)
        if not self.replies:
            return "{}"
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def _json(intent: str, plan: list[dict], **extra) -> str:
    data = {
        "intent": intent,
        "sub_queries": plan,
        "needs_table": False,
        "needs_calculation": False,
    }
    data.update(extra)
    return json.dumps(data, ensure_ascii=False)


def test_valid_json_plan():
    brain = FakeBrain(
        [
            _json(
                "comparison",
                [
                    {"text": "Milvus 的索引类型", "purpose": "实体 A"},
                    {"text": "BM25 的检索原理", "purpose": "实体 B"},
                ],
            )
        ]
    )
    analysis = BrainAnalyzer(brain).analyze("对比 Milvus 和 BM25")

    assert analysis.intent == Intent.COMPARISON
    assert analysis.source == "brain"
    assert not analysis.degraded
    assert [sq.text for sq in analysis.plan] == ["Milvus 的索引类型", "BM25 的检索原理"]
    assert len(analysis.llm_traces) == 1
    assert analysis.llm_traces[0]["messages"][0]["role"] == "system"


def test_fenced_json_is_accepted():
    raw = "```json\n" + _json("fact", [{"text": "HNSW 是什么"}]) + "\n```"
    analysis = BrainAnalyzer(FakeBrain([raw])).analyze("HNSW 是什么")
    assert analysis.intent == Intent.FACT
    assert analysis.plan[0].text == "HNSW 是什么"


def test_invalid_json_retries_once_then_succeeds():
    brain = FakeBrain(["不是 JSON", _json("fact", [{"text": "重试后的问题"}])])
    analysis = BrainAnalyzer(brain, max_retries=1).analyze("问题")

    assert len(brain.calls) == 2
    assert analysis.source == "brain"
    assert analysis.plan[0].text == "重试后的问题"
    assert len(analysis.llm_traces) == 2


def test_invalid_json_twice_falls_back_to_rules():
    brain = FakeBrain(["oops", "still oops"])
    analysis = BrainAnalyzer(brain, max_retries=1).analyze("Milvus 是什么")

    assert len(brain.calls) == 2
    assert analysis.degraded is True
    assert analysis.source == "rules"
    assert analysis.intent == Intent.FACT
    assert len(analysis.plan) == 1
    assert analysis.error == "json_parse_error"


def test_brain_exception_falls_back():
    brain = FakeBrain([RuntimeError("boom"), RuntimeError("boom")])
    analysis = BrainAnalyzer(brain, max_retries=1).analyze("Milvus 是什么")
    assert analysis.source == "rules"
    assert analysis.degraded is True
    assert "llm_error" in analysis.error


def test_chitchat_fast_path_skips_brain():
    brain = FakeBrain([])
    analysis = BrainAnalyzer(brain).analyze("你好")
    assert analysis.intent == Intent.CHITCHAT
    assert brain.calls == []


def test_empty_question_is_out_of_scope_without_brain():
    brain = FakeBrain([])
    analysis = BrainAnalyzer(brain).analyze("   ")
    assert analysis.intent == Intent.OUT_OF_SCOPE
    assert brain.calls == []


def test_out_of_scope_from_brain_has_empty_plan():
    brain = FakeBrain([_json("out_of_scope", [])])
    analysis = BrainAnalyzer(brain).analyze("今天天气如何")
    assert analysis.intent == Intent.OUT_OF_SCOPE
    assert analysis.plan == []


def test_hallucinated_source_and_page_are_dropped():
    brain = FakeBrain(
        [
            _json(
                "fact",
                [{"text": "第 2 页的结论", "source": "fake.pdf", "page": 9}],
            )
        ]
    )
    analysis = BrainAnalyzer(brain).analyze(
        "第 2 页的结论是什么", sources=["doc_a.md"]
    )
    sq = analysis.plan[0]
    assert sq.source is None          # 不在已入库清单中
    assert sq.page == 2               # 回退到问题中明确出现的页码


def test_explicit_filename_in_question_sets_source():
    brain = FakeBrain([_json("summary", [{"text": "文档核心结论"}])])
    analysis = BrainAnalyzer(brain).analyze(
        "请汇总 doc_a.md 第 3 页的核心结论", sources=["doc_a.md"]
    )
    sq = analysis.plan[0]
    assert sq.source == "doc_a.md"
    assert sq.page == 3


def test_invalid_channels_normalized():
    brain = FakeBrain(
        [_json("fact", [{"text": "问题", "channels": ["web", "bm25"]}])]
    )
    analysis = BrainAnalyzer(brain).analyze("问题")
    assert analysis.plan[0].channels == ("bm25",)


def test_rule_analyzer_applies_detected_page():
    analysis = RuleAnalyzer().analyze("第 4 页讲了什么")
    assert analysis.plan[0].page == 4


def test_extract_filters():
    detected = extract_filters("《doc_a.md》第 2 页", sources=["doc_a.md"])
    assert detected.source == "doc_a.md"
    assert detected.page == 2


def test_hyde_generator_strips_and_limits():
    class Brain:
        def chat(self, messages):
            return '"假设答案：Milvus 支持 HNSW 索引。"'

    text, trace = HydeGenerator(Brain(), max_chars=200).generate("Milvus 支持什么索引？")
    assert text.startswith("假设答案")
    assert trace["messages"][-1]["content"] == "Milvus 支持什么索引？"


def test_brain_client_error_is_wrapped():
    class Broken:
        def chat(self, messages):
            raise RuntimeError("down")

    with pytest.raises(RuntimeError):
        HydeGenerator(Broken()).generate("问题")
