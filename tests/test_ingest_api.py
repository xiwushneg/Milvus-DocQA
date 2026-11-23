"""入库接口的错误码契约：非法类型 4xx、空文件显式 skipped。"""
from __future__ import annotations

import asyncio
import io

import pytest
from fastapi import HTTPException, UploadFile

from app.main import ingest


class FakePipeline:
    def __init__(self):
        self.refreshes = 0

    def ingest_bytes(self, name: str, data: bytes, rebuild_bm25: bool = True) -> dict:
        return {
            "source": name,
            "sha256": "x",
            "segments": 1,
            "chunks": 1,
            "split_ms": 0,
            "embed_ms": 0,
            "elapsed_ms": 0,
        }

    def refresh_bm25(self) -> int:
        self.refreshes += 1
        return 1


def _upload(name: str, data: bytes) -> UploadFile:
    return UploadFile(filename=name, file=io.BytesIO(data))


def test_ingest_rejects_unsupported_type():
    with pytest.raises(HTTPException) as exc:
        asyncio.run(ingest([_upload("a.exe", b"x")], pipeline=FakePipeline()))
    assert exc.value.status_code == 415


def test_ingest_all_empty_is_400_with_skipped():
    with pytest.raises(HTTPException) as exc:
        asyncio.run(ingest([_upload("a.md", b"")], pipeline=FakePipeline()))
    assert exc.value.status_code == 400
    assert exc.value.detail["skipped"] == [{"source": "a.md", "reason": "文件为空"}]


def test_ingest_reports_skipped_for_partial_empty():
    pipeline = FakePipeline()
    out = asyncio.run(
        ingest(
            [_upload("a.md", b"hello"), _upload("b.md", b"")],
            pipeline=pipeline,
        )
    )
    assert [d.source for d in out.documents] == ["a.md"]
    assert out.skipped == [{"source": "b.md", "reason": "文件为空"}]
    assert pipeline.refreshes == 1, "批量入库结束后只重建一次 BM25"


class DuplicatePipeline(FakePipeline):
    def ingest_bytes(self, name: str, data: bytes, rebuild_bm25: bool = True) -> dict:
        stats = super().ingest_bytes(name, data, rebuild_bm25)
        if name == "dup.md":
            stats.update(
                {
                    "skipped": True,
                    "reason": "duplicate",
                    "duplicate_of": "a.md",
                    "chunks": 0,
                }
            )
        return stats


def test_ingest_reports_duplicate_in_skipped():
    pipeline = DuplicatePipeline()
    out = asyncio.run(
        ingest(
            [_upload("a.md", b"hello"), _upload("dup.md", b"hello")],
            pipeline=pipeline,
        )
    )
    assert [d.source for d in out.documents] == ["a.md"]
    assert out.skipped[0]["source"] == "dup.md"
    assert out.skipped[0]["duplicate_of"] == "a.md"
    assert pipeline.refreshes == 1
