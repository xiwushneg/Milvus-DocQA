"""全量读取分页：query_iterator 突破 16384，缺失时回退 offset 分页。"""
from __future__ import annotations

from core.vectorstore import MilvusStore


def _row(i: int) -> dict:
    return {"id": i, "text": f"t{i}", "source": "s", "page": 1, "chunk_idx": i}


class FakeIterator:
    def __init__(self, client: "PagedClient", batch_size: int):
        self.client = client
        self.batch_size = batch_size
        self.pos = 0
        self.closed = False

    def next(self) -> list[dict]:
        if self.pos >= self.client.total:
            return []
        end = min(self.pos + self.batch_size, self.client.total)
        batch = [_row(i) for i in range(self.pos, end)]
        self.pos = end
        return batch

    def close(self) -> None:
        self.closed = True


class PagedClient:
    def __init__(self, total: int):
        self.total = total
        self.query_offsets: list[int] = []

    def query(
        self,
        collection_name: str,
        filter: str,
        output_fields: list[str],
        limit: int = 0,
        offset: int = 0,
    ):
        if output_fields == ["count(*)"]:
            return [{"count(*)": self.total}]
        self.query_offsets.append(offset)
        assert offset + limit <= 16384, "offset+limit 超出 Milvus 上限"
        return [_row(i) for i in range(offset, min(offset + limit, self.total))]

    def query_iterator(self, collection_name, filter, output_fields, batch_size):
        return FakeIterator(self, batch_size)


class NoIteratorClient(PagedClient):
    query_iterator = None


def make_store(client) -> MilvusStore:
    store = MilvusStore.__new__(MilvusStore)
    store.collection = "c"
    store.dim = 8
    store.client = client
    return store


def test_all_chunks_uses_iterator_beyond_offset_limit():
    client = PagedClient(20000)
    rows = make_store(client).all_chunks(max_rows=20000)

    assert len(rows) == 20000
    assert client.query_offsets == [], "有迭代器时不应走 offset 分页"


def test_all_chunks_falls_back_to_offset_pagination():
    client = NoIteratorClient(20000)
    rows = make_store(client).all_chunks(max_rows=20000)

    # 回退路径在 offset+limit 触达 16384 前停止（16000 行）
    assert len(rows) == 16000
    assert client.query_offsets[:2] == [0, 1000]


def test_list_sources_reads_all_rows_via_iterator():
    counts = make_store(PagedClient(20000)).list_sources()
    assert counts == {"s": 20000}
