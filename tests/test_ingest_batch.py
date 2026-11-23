"""批量入库：BM25 只重建一次 + sha256 去重清单生命周期。"""
from __future__ import annotations

import json

from core.embedder import MockEmbedder
from core.pipeline import RagPipeline

CONTENT = (
    "# 测试文档\n\n苹果和香蕉都是水果。Milvus 是向量数据库，"
    "支持 HNSW 索引与 RRF 融合检索。"
)


class FakeStore:
    collection = "test_collection"

    def __init__(self):
        self.rows: list[dict] = []

    def insert_chunks(self, rows: list[dict]) -> int:
        self.rows.extend(rows)
        return len(rows)

    def all_chunks(self, max_rows: int = 100000, wait_visible: bool = False):
        return list(self.rows)

    def count_source(self, source: str) -> int:
        return sum(1 for r in self.rows if r["source"] == source)

    def delete_by_source(self, source: str) -> int:
        before = len(self.rows)
        self.rows = [r for r in self.rows if r["source"] != source]
        return before - len(self.rows)


def make_pipeline(manifest_path: str = "") -> RagPipeline:
    cfg = {
        "milvus": {"uri": "http://stub", "collection": FakeStore.collection, "dense_dim": 8},
        "chunking": {"chunk_size": 60, "chunk_overlap": 15, "min_chunk_size": 10},
        "retrieval": {"rerank": {"enabled": False}},
        "embedding": {"provider": "mock"},
    }
    if manifest_path:
        cfg["ingest"] = {"manifest_path": manifest_path}
    return RagPipeline(
        cfg, store=FakeStore(), embedder=MockEmbedder(dim=8), llm=object()
    )


def test_ingest_paths_rebuilds_bm25_once(tmp_path, monkeypatch):
    pipeline = make_pipeline()
    calls: list[int] = []
    monkeypatch.setattr(
        pipeline, "refresh_bm25", lambda: calls.append(1) or len(pipeline.store.rows)
    )
    a = tmp_path / "a.md"
    b = tmp_path / "b.md"
    a.write_text(CONTENT, encoding="utf-8")
    b.write_text(CONTENT + "\n\n补充内容。", encoding="utf-8")

    stats = pipeline.ingest_paths([str(a), str(b)])

    assert len(stats) == 2
    assert all(s["chunks"] > 0 for s in stats)
    assert len(calls) == 1, "批量入库只允许重建一次 BM25"


def test_duplicate_sha_is_skipped(tmp_path):
    manifest = tmp_path / "ingest.json"
    pipeline = make_pipeline(str(manifest))

    first = pipeline.ingest_bytes("a.md", CONTENT.encode("utf-8"))
    second = pipeline.ingest_bytes("copy.md", CONTENT.encode("utf-8"))

    assert first["chunks"] > 0
    assert second["skipped"] is True
    assert second["chunks"] == 0
    assert second["duplicate_of"] == "a.md"
    assert len(pipeline.store.rows) == first["chunks"]

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["collection"] == FakeStore.collection
    assert first["sha256"] in payload["files"]


def test_forget_source_allows_reingest(tmp_path):
    manifest = tmp_path / "ingest.json"
    pipeline = make_pipeline(str(manifest))
    first = pipeline.ingest_bytes("a.md", CONTENT.encode("utf-8"))

    deleted = pipeline.forget_source("a.md")
    assert deleted == first["chunks"]
    assert not pipeline.store.rows

    again = pipeline.ingest_bytes("a.md", CONTENT.encode("utf-8"))
    assert not again.get("skipped")
    assert again["chunks"] > 0


def test_manifest_disabled_without_config():
    pipeline = make_pipeline()
    pipeline.ingest_bytes("a.md", CONTENT.encode("utf-8"))
    second = pipeline.ingest_bytes("b.md", CONTENT.encode("utf-8"))
    assert not second.get("skipped")
    assert len(pipeline.store.rows) == second["chunks"] * 2
