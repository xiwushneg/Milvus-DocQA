"""批量写入失败回滚：已插入批次的主键必须被删除，避免半入库。"""
from __future__ import annotations

import pytest

from core.vectorstore import MilvusStore


class FlakyClient:
    def __init__(self):
        self.insert_calls = 0
        self.delete_filters: list[str] = []

    def insert(self, collection_name: str, data: list[dict]) -> dict:
        self.insert_calls += 1
        if self.insert_calls == 2:
            raise RuntimeError("模拟第二批写入失败")
        ids = list(range(len(data)))
        return {"insert_count": len(data), "ids": ids}

    def delete(self, collection_name: str, filter: str) -> dict:
        self.delete_filters.append(filter)
        return {"delete_count": 500}

    def flush(self, collection_name: str) -> None:
        pass


def make_store(client) -> MilvusStore:
    store = MilvusStore.__new__(MilvusStore)
    store.collection = "c"
    store.dim = 8
    store.client = client
    return store


def test_insert_rolls_back_previous_batches_on_failure():
    client = FlakyClient()
    store = make_store(client)
    rows = [{"text": f"t{i}", "source": "a.md", "page": 1, "chunk_idx": i, "dense": [0.1]} for i in range(600)]

    with pytest.raises(RuntimeError):
        store.insert_chunks(rows)

    assert client.insert_calls == 2
    assert len(client.delete_filters) == 1
    assert client.delete_filters[0] == "id in [" + ",".join(str(i) for i in range(500)) + "]"


def test_insert_without_ids_cannot_rollback_but_reraises():
    class NoIdsClient(FlakyClient):
        def insert(self, collection_name: str, data: list[dict]) -> dict:
            self.insert_calls += 1
            if self.insert_calls == 2:
                raise RuntimeError("失败")
            return {"insert_count": len(data)}

    client = NoIdsClient()
    store = make_store(client)
    with pytest.raises(RuntimeError):
        store.insert_chunks([{"text": "x"}] * 600)
    assert client.delete_filters == []
