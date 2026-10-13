"""分块模块单元测试。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.splitter import recursive_split, split_segments  # noqa: E402


def test_short_text_single_chunk():
    assert recursive_split("你好世界", 500, 80) == ["你好世界"]


def test_respects_chunk_size():
    text = "这是第一句话。这是第二句话。" * 200  # ~2800 字符
    chunks = recursive_split(text, chunk_size=200, chunk_overlap=30)
    assert all(len(c) <= 200 for c in chunks)
    assert len(chunks) >= 5


def test_no_text_loss_with_zero_overlap():
    text = "段落一。\n\n段落二。\n\n段落三。\n\n" * 30
    chunks = recursive_split(text, chunk_size=100, chunk_overlap=0)
    assert sum(len(c) for c in chunks) == len(text.strip("\n"))


def test_hard_split_overlap():
    text = "甲乙丙丁" * 300  # 无任何分隔符 → 滑窗硬切
    chunks = recursive_split(text, chunk_size=200, chunk_overlap=50)
    assert len(chunks) >= 3
    # 相邻块保留 overlap 重叠
    assert chunks[1][:50] == chunks[0][-50:]


def test_overlap_must_be_smaller_than_chunk():
    try:
        recursive_split("abc", 100, 100)
    except ValueError:
        return
    raise AssertionError("overlap >= chunk_size 应抛出 ValueError")


def test_split_segments_metadata():
    segs = [
        {"text": "第一页内容。" * 100, "page": 3},
        {"text": "第二页内容。" * 100, "page": 4},
    ]
    chunks = split_segments(segs, source="demo.md", chunk_size=120, chunk_overlap=20)
    assert all(c.source == "demo.md" for c in chunks)
    assert {c.page for c in chunks} == {3, 4}
    assert [c.chunk_idx for c in chunks] == list(range(len(chunks)))
