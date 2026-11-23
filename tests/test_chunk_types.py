"""块类型规则判定与内存缓存测试（不触达 Milvus）。"""
from __future__ import annotations

from core.agentic.chunk_types import ChunkTypeIndex, classify, count_types


def test_markdown_table():
    text = "| 模型 | 参数量 |\n| --- | --- |\n| Qwen3 | 1.7B |"
    assert classify(text) == "table"


def test_tab_aligned_table():
    text = "模型\t参数量\nQwen3\t1.7B"
    assert classify(text) == "table"


def test_fenced_code():
    assert classify("```python\nprint('hi')\n```") == "code"


def test_dense_symbol_code():
    text = (
        "def build_index(rows, dim=8):\n"
        "    index = {i: r for i, r in enumerate(rows)}\n"
        "    return index.get(dim, None)\n"
    )
    assert classify(text) == "code"


def test_enum_list():
    text = "- 第一项说明\n- 第二项说明\n1. 第三项说明"
    assert classify(text) == "enum"


def test_formula():
    text = "召回率 = 命中数 / 总数，示例：0.950，误差 0.05，阈值 0.65"
    assert classify(text) == "formula"


def test_prose_default():
    assert classify("Milvus 是一款开源向量数据库，支持多种索引。") == "prose"


def test_cache_rebuilds_when_rows_change():
    rows = [{"id": 1, "text": "普通段落文本"}]
    index = ChunkTypeIndex(lambda: rows)
    assert index.get(1) == "prose"
    assert index.stats() == {"prose": 1}

    rows.append({"id": 2, "text": "```sh\npip install milvus\n```"})
    assert index.get(2) == "code"
    assert index.stats()["code"] == 1


def test_annotate_and_count():
    index = ChunkTypeIndex(lambda: [])
    hits = [
        {"id": 1, "text": "| a | b |\n| 1 | 2 |"},
        {"id": 2, "text": "普通文本"},
    ]
    assert index.annotate(hits) is hits
    assert [h["chunk_type"] for h in hits] == ["table", "prose"]
    assert count_types(hits) == {"table": 1, "prose": 1}
