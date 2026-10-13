"""RRF（Reciprocal Rank Fusion）倒数排名融合。

公式: RRF(d) = Σ_i 1 / (k + rank_i(d))，k 经典取 60。
只使用排名、不使用原始分数，天然规避 BM25 分值与余弦相似度分值尺度不一致的问题，
是混合检索的工业标准融合方案。
"""
from __future__ import annotations

CHANNEL_NAMES = ["dense", "bm25"]


def rrf_fuse(
    result_lists: list[list[dict]], k: int = 60, id_key: str = "id"
) -> list[dict]:
    """融合多路检索结果。

    - result_lists: 各通道的有序结果列表（第一名 rank=1）
    - 同一文档以 id_key 字段聚合，RRF 分数累加
    - 返回按 RRF 分数降序的列表，附带 rrf_score 与 channels（命中通道）
    """
    scores: dict = {}
    items: dict = {}
    channels: dict = {}
    for ci, lst in enumerate(result_lists):
        name = CHANNEL_NAMES[ci] if ci < len(CHANNEL_NAMES) else f"ch{ci}"
        for rank, item in enumerate(lst, start=1):
            uid = item[id_key]
            scores[uid] = scores.get(uid, 0.0) + 1.0 / (k + rank)
            if uid not in items:
                items[uid] = dict(item)
                channels[uid] = []
            channels[uid].append(name)
    fused: list[dict] = []
    for uid, s in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        it = items[uid]
        it["rrf_score"] = s
        it["channels"] = channels[uid]
        fused.append(it)
    return fused
