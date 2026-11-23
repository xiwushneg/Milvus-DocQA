"""BM25 重建与查询并发回归：索引与 meta 必须是同一版本快照。"""
from __future__ import annotations

import threading
import time

from core.bm25 import BM25Index


def _rows(prefix: str, keyword: str, start: int, count: int) -> list[dict]:
    return [
        {
            "id": start + i,
            "text": f"{prefix} {keyword} 编号 {i}",
            "source": f"{prefix}.md",
            "page": 1,
            "chunk_idx": i,
        }
        for i in range(count)
    ]


ROWS_A = _rows("alpha", "苹果", 0, 60)
ROWS_B = _rows("beta", "香蕉", 1000, 90)


def test_bm25_concurrent_build_and_search():
    index = BM25Index()
    index.build(ROWS_A)
    stop = threading.Event()
    errors: list[str] = []

    def writer() -> None:
        try:
            for _ in range(50):
                index.build(ROWS_A)
                index.build(ROWS_B)
        finally:
            stop.set()

    def reader() -> None:
        while not stop.is_set():
            try:
                for hit in index.search("苹果", top_k=5):
                    assert hit["id"] < 1000, f"苹果命中混入 beta: {hit}"
                for hit in index.search("香蕉", top_k=5):
                    assert hit["id"] >= 1000, f"香蕉命中混入 alpha: {hit}"
            except Exception as exc:  # noqa: BLE001 - 并发缺陷需要聚合上报
                errors.append(repr(exc))
            time.sleep(0.001)

    readers = [threading.Thread(target=reader) for _ in range(4)]
    for t in readers:
        t.start()
    w = threading.Thread(target=writer)
    w.start()
    w.join(timeout=30)
    for t in readers:
        t.join(timeout=30)

    assert not w.is_alive()
    assert not errors, errors[:3]
