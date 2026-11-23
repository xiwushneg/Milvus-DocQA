"""B6 评测集与多跳评测单测（离线、无 Milvus / LLM 依赖）。"""
from __future__ import annotations

import json
from pathlib import Path

from evaluation.build_qa_zh import (
    build_llm_dataset,
    build_template_dataset,
    pick_keyphrases,
    refusal_items,
    template_questions,
)
from evaluation.eval_multihop import evaluate, gold_titles, load_items, rank_contexts

_CHUNK = (
    "Milvus Lite 是轻量级部署形态，适合本地开发调试。\n"
    "Milvus Standalone 适合生产环境，单机即可支撑百万级向量。\n"
    "HNSW 是分层可导航小世界图索引，查询速度快、内存占用高。"
)


def test_pick_keyphrases_are_original_substrings():
    phrases = pick_keyphrases(_CHUNK, limit=3)
    assert phrases, "应至少抽到一条短语"
    assert len(phrases) <= 3
    for p in phrases:
        assert p in _CHUNK
        assert len(p) >= 10
    assert len(set(phrases)) == len(phrases)


def test_template_questions_have_gold_in_source_text():
    items = template_questions("milvus入门", _CHUNK, variants=2)
    assert len(items) == 2
    for item in items:
        gold = item["gold"][0]
        assert gold["source"] == "milvus入门"
        assert gold["must_contain"] in _CHUNK
        assert gold["must_contain"] in item["question"]
        assert item["refusal_expected"] is False


def test_template_questions_fallback_for_short_chunk():
    text = "短块内容：Milvus 支持向量检索。"
    items = template_questions("短文档", text, variants=2)
    assert len(items) >= 1
    for item in items:
        assert item["gold"][0]["must_contain"] in text


def test_refusal_items_are_unique_and_flagged():
    items = refusal_items(["doc_a", "doc_b", "doc_c"], limit=6)
    assert len(items) == 6
    assert len({i["question"] for i in items}) == 6
    assert all(i["refusal_expected"] is True for i in items)
    assert all(i["gold"] == [] for i in items)


def test_build_template_dataset_hits_target_and_dedups():
    chunks = [
        {"source": f"doc_{i}", "text": _CHUNK + f"\n第 {i} 篇文档补充说明，包含独特编号 {i}。"}
        for i in range(5)
    ]
    items = build_template_dataset(chunks, target=15, variants=4, refusal_ratio=0.1)
    assert len(items) == 15
    questions = [i["question"] for i in items]
    assert len(set(questions)) == len(questions)
    assert sum(1 for i in items if i["refusal_expected"]) == 3
    assert {i["gold"][0]["source"] for i in items if not i["refusal_expected"]} == {
        f"doc_{i}" for i in range(5)
    }


def test_build_template_dataset_never_exceeds_target():
    chunks = [{"source": "only", "text": _CHUNK}]
    items = build_template_dataset(chunks, target=50, variants=4)
    assert len(items) <= 50
    assert len({i["question"] for i in items}) == len(items)


def test_eval_set_answerable_items_have_reference_answers():
    path = Path(__file__).resolve().parents[1] / "evaluation/eval_set.json"
    items = json.loads(path.read_text(encoding="utf-8"))
    answerable = [i for i in items if not i.get("refusal_expected")]
    refusal = [i for i in items if i.get("refusal_expected")]

    assert len(answerable) == 44
    assert len(refusal) == 5
    for item in answerable:
        answer = item.get("answer", "").strip()
        assert answer, f"缺少参考答案: {item['question']}"
        for gold in item["gold"]:
            assert gold["must_contain"] in answer
    assert all("answer" not in item for item in refusal)


class _FakeClient:
    def __init__(self, replies: list[str]) -> None:
        self.replies = replies
        self.calls = 0

    def chat(self, messages, **kwargs) -> str:  # noqa: ANN001
        reply = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        return reply


def test_build_llm_dataset_retries_then_accepts():
    text = "Milvus Lite 适合本地开发调试。"
    chunks = [{"source": "doc", "text": text}]
    client = _FakeClient(
        [
            "这不是 JSON",
            json.dumps({"question": "Milvus Lite 适合什么场景？", "must_contain": "本地开发调试"}),
        ]
    )
    items = build_llm_dataset(chunks, client, target=10)
    assert len(items) == 1
    assert client.calls == 2
    assert items[0]["gold"][0]["must_contain"] == "本地开发调试"


def test_build_llm_dataset_skips_hallucinated_phrase():
    chunks = [{"source": "doc", "text": "原文只提到 HNSW。"}]
    client = _FakeClient(
        [json.dumps({"question": "问什么？", "must_contain": "原文中没有的短语"})] * 2
    )
    assert build_llm_dataset(chunks, client, target=10) == []


def test_load_items_supports_json_and_jsonl(tmp_path):
    rows = [{"question": "q1", "answer": "a1"}, {"question": "q2", "answer": "a2"}]
    json_path = tmp_path / "samples.json"
    json_path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    jsonl_path = tmp_path / "samples.jsonl"
    jsonl_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8"
    )
    assert load_items(json_path) == rows
    assert load_items(jsonl_path) == rows


def test_gold_titles_accepts_multiple_formats():
    assert gold_titles({"supporting_titles": ["A", "A", "B"]}) == ["A", "B"]
    assert gold_titles({"supporting_facts": [["A", 0], ["B", 1]]}) == ["A", "B"]
    assert gold_titles({"supporting_facts": [{"title": "C"}]}) == ["C"]
    assert gold_titles({"contexts": [{"title": "D", "is_supporting": True}]}) == ["D"]


def test_rank_contexts_puts_gold_title_first():
    contexts = [
        {"title": "天气", "text": "今天晴朗，适合郊游。"},
        {"title": "菜谱", "text": "番茄炒蛋需要先打散鸡蛋。"},
        {"title": "Milvus", "text": "HNSW 是分层可导航小世界图索引。"},
    ]
    ranked = rank_contexts("HNSW 索引是什么结构", contexts, top_k=2)
    assert ranked[0]["source"] == "Milvus"


def test_evaluate_retrieval_only():
    items = [
        {
            "question": "HNSW 索引是什么",
            "contexts": [
                {"title": "Milvus", "text": "HNSW 是分层可导航小世界图索引。"},
                {"title": "天气", "text": "今天晴朗，适合郊游。"},
            ],
            "supporting_titles": ["Milvus"],
        }
    ]
    out = evaluate(items, top_k=2)
    assert out["n"] == 1
    assert out["has_predictions"] is False
    assert out["retrieval"]["recall_at_k"] == 1.0
    assert out["retrieval"]["mrr"] == 1.0
    assert out["answer"]["exact_match"] is None


def test_evaluate_with_prediction_scores_em_f1():
    items = [
        {
            "question": "HNSW 索引是什么",
            "contexts": [{"title": "Milvus", "text": "HNSW 图索引"}],
            "supporting_titles": ["Milvus"],
            "answer": "HNSW 是分层可导航小世界图",
            "prediction": "HNSW 是分层可导航小世界图 [E1]",
        }
    ]
    out = evaluate(items, top_k=1)
    assert out["has_predictions"] is True
    assert out["answer"]["exact_match"] == 1.0
    assert out["answer"]["f1"] == 1.0
