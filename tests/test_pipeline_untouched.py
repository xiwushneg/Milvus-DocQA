"""RagPipeline._retrieve 源码哈希冻结：约束「一行不改」的硬证据。"""
from __future__ import annotations

import hashlib
import inspect

from core.pipeline import RagPipeline

# B0 动工前采集，见 docs/agentic-rag-design.md §7 R2
EXPECTED_SHA256 = "7b49cc28247be9136467c147c275d9c9b1fb99c044bcf5da5b6550d02292f651"


def test_retrieve_source_is_frozen():
    source = inspect.getsource(RagPipeline._retrieve)
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    assert digest == EXPECTED_SHA256, (
        "RagPipeline._retrieve 被修改；Agent 升级不得改变固定流水线的检索实现。"
    )
