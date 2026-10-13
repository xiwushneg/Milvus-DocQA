"""RRF 融合模块单元测试。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.fusion import rrf_fuse  # noqa: E402


def test_rrf_math():
    a = [{"id": 1}, {"id": 2}, {"id": 3}]
    b = [{"id": 2}, {"id": 1}]
    fused = rrf_fuse([a, b], k=60)
    scores = {f["id"]: f["rrf_score"] for f in fused}
    assert abs(scores[1] - (1 / 61 + 1 / 62)) < 1e-9
    assert abs(scores[2] - (1 / 62 + 1 / 61)) < 1e-9
    assert abs(scores[3] - 1 / 63) < 1e-9
    # id=1 与 id=2 并列第一，id=3 最后
    assert fused[-1]["id"] == 3
    assert abs(fused[0]["rrf_score"] - fused[1]["rrf_score"]) < 1e-9


def test_channel_tracking():
    a = [{"id": 1}]
    b = [{"id": 1}, {"id": 9}]
    fused = rrf_fuse([a, b], k=60)
    by_id = {f["id"]: f for f in fused}
    assert by_id[1]["channels"] == ["dense", "bm25"]
    assert by_id[9]["channels"] == ["bm25"]


def test_order_and_coverage():
    a = [{"id": 10}, {"id": 11}, {"id": 12}]
    b = [{"id": 10}, {"id": 20}, {"id": 21}]
    fused = rrf_fuse([a, b], k=60)
    ids = [f["id"] for f in fused]
    assert set(ids) == {10, 11, 12, 20, 21}
    # 10 命中双通道应排第一；各名次分数单调不增
    assert fused[0]["id"] == 10
    assert fused[0]["rrf_score"] >= fused[1]["rrf_score"] > fused[-1]["rrf_score"]
