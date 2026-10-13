"""Milvus 向量存储封装。

基于 pymilvus 的 MilvusClient 统一 API，三种部署无缝切换：
- MilvusClient("./xx.db")           → Milvus Lite（免安装，本地文件即数据库）
- MilvusClient("http://host:19530") → Milvus Standalone（Docker 部署）

Schema: id(主键自增) / text / source / page / chunk_idx / dense(FLOAT_VECTOR)
索引: AUTOINDEX + COSINE（Milvus Lite 与 Standalone 均支持；Standalone 生产可改 HNSW）
"""
from __future__ import annotations

import time

from pymilvus import DataType, MilvusClient

OUTPUT_FIELDS = ["id", "text", "source", "page", "chunk_idx"]


class MilvusStore:
    def __init__(self, uri: str, collection: str = "docqa_chunks", dense_dim: int = 1024):
        self.client = MilvusClient(uri)
        self.collection = collection
        self.dim = dense_dim
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if self.client.has_collection(self.collection):
            return
        schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("text", DataType.VARCHAR, max_length=8192)
        schema.add_field("source", DataType.VARCHAR, max_length=1024)
        schema.add_field("page", DataType.INT64)
        schema.add_field("chunk_idx", DataType.INT64)
        schema.add_field("dense", DataType.FLOAT_VECTOR, dim=self.dim)
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="dense", index_type="AUTOINDEX", metric_type="COSINE"
        )
        self.client.create_collection(
            self.collection, schema=schema, index_params=index_params
        )

    # ---------- 写入 ----------
    def insert_chunks(self, rows: list[dict]) -> int:
        total = 0
        for i in range(0, len(rows), 500):
            batch = rows[i : i + 500]
            res = self.client.insert(collection_name=self.collection, data=batch)
            total += (
                int(res.get("insert_count", len(batch)))
                if isinstance(res, dict)
                else len(batch)
            )
        self._flush_safe()
        return total

    def _flush_safe(self) -> None:
        """写入后尽量触发 flush，使数据立刻可查；后端不支持时静默跳过。"""
        try:
            self.client.flush(collection_name=self.collection)
        except Exception:
            pass

    # ---------- 检索 ----------
    def search_dense(self, query_vec: list[float], top_k: int = 20) -> list[dict]:
        res = self.client.search(
            collection_name=self.collection,
            data=[query_vec],
            anns_field="dense",
            limit=top_k,
            filter="",
            output_fields=OUTPUT_FIELDS,
            search_params={"metric_type": "COSINE"},
        )
        hits: list[dict] = []
        for row in res[0]:
            e = row["entity"]
            hits.append(
                {
                    "id": row["id"],
                    "dense_score": float(row["distance"]),
                    "text": e["text"],
                    "source": e["source"],
                    "page": int(e["page"]),
                    "chunk_idx": int(e["chunk_idx"]),
                }
            )
        return hits

    # ---------- 全量读取（用于构建 BM25 内存索引） ----------
    def all_chunks(self, max_rows: int = 100000, wait_visible: bool = False) -> list[dict]:
        """offset 分页受 Milvus 16384 上限约束；更大规模请切换 Standalone
        并改用 query_iterator（README 常见问题）。

        Milvus 默认 Bounded 一致性下，新写入可能短暂不可见（且行数统计可能滞后）：
        - wait_visible=True：刚写入过数据，强制重试直到查询可见（约 3 秒上限）
        - 否则仅当统计行数 > 0 且查询为空时短暂重试
        """
        expected = self.count()
        rows: list[dict] = []
        for attempt in range(6):
            rows = []
            offset, page_size = 0, 1000
            while len(rows) < max_rows:
                batch = self.client.query(
                    collection_name=self.collection,
                    filter="id >= 0",
                    output_fields=OUTPUT_FIELDS,
                    limit=page_size,
                    offset=offset,
                )
                if not batch:
                    break
                rows.extend(batch)
                if len(batch) < page_size:
                    break
                offset += page_size
            if rows or (expected == 0 and not wait_visible):
                break
            time.sleep(0.5)
        return rows

    # ---------- 管理 ----------
    def delete_by_source(self, source: str) -> int:
        safe = source.replace('"', '\\"')
        res = self.client.delete(
            collection_name=self.collection, filter=f'source == "{safe}"'
        )
        self._flush_safe()  # 触发 flush，使删除立刻对后续查询/统计可见
        if isinstance(res, dict):
            return int(res.get("delete_count", 0))
        if isinstance(res, list):  # 部分后端（如 Milvus Lite）返回被删主键列表
            return len(res)
        return 0

    def list_sources(self) -> dict[str, int]:
        rows = self.client.query(
            collection_name=self.collection,
            filter="id >= 0",
            output_fields=["source"],
            limit=16384,
        )
        counts: dict[str, int] = {}
        for r in rows:
            counts[r["source"]] = counts.get(r["source"], 0) + 1
        return counts

    def drop(self) -> None:
        """删除整个 collection（测试隔离 / 重建索引用）。"""
        if self.client.has_collection(self.collection):
            self.client.drop_collection(self.collection)

    def count(self) -> int:
        # 用 query count(*) 而非 get_collection_stats：后者在 Standalone 上
        # 统计有延迟（写入/删除后不会立刻反映），query 结果实时且精确。
        res = self.client.query(
            collection_name=self.collection,
            filter="id >= 0",
            output_fields=["count(*)"],
        )
        if res and isinstance(res[0], dict):
            return int(res[0].get("count(*)", 0))
        return 0
