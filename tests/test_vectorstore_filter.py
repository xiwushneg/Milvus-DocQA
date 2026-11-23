"""Milvus filter 字符串转义：双引号与反斜杠都必须转义。"""
from __future__ import annotations

from core.vectorstore import MilvusStore


class FakeClient:
    def __init__(self):
        self.last_filter = ""

    def delete(self, collection_name: str, filter: str) -> dict:
        self.last_filter = filter
        return {"delete_count": 0}


def test_delete_by_source_escapes_backslash_and_quote():
    store = MilvusStore.__new__(MilvusStore)
    store.collection = "c"
    store.client = FakeClient()
    store._flush_safe = lambda: None

    assert store.delete_by_source('a\\b"c') == 0
    assert store.client.last_filter == 'source == "a\\\\b\\"c"'
