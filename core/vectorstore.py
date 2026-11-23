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

_PAGE_SIZE = 1000
# Milvus 要求 offset + limit ≤ 16384；迭代器不可用时回退分页只能读到这个上界
_MAX_PAGE_WINDOW = 16384


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
        inserted_ids: list[int] = []
        try:
            for i in range(0, len(rows), 500):
                batch = rows[i : i + 500]
                res = self.client.insert(collection_name=self.collection, data=batch)
                if isinstance(res, dict):
                    total += int(res.get("insert_count", len(batch)))
                    ids = res.get("ids")
                    if isinstance(ids, list):
                        inserted_ids.extend(ids)
                else:
                    total += len(batch)
        except Exception:
            # 任一批失败即回滚已写入的主键，避免留下半入库文档
            self._delete_ids(inserted_ids)
            raise
        self._flush_safe()
        return total

    def _delete_ids(self, ids: list) -> None:
        clean = [str(int(i)) for i in ids if str(i).lstrip("-").isdigit()]
        if not clean:
            return
        try:
            self.client.delete(
                collection_name=self.collection, filter=f"id in [{','.join(clean)}]"
            )
            self._flush_safe()
        except Exception:
            pass  # 回滚尽力而为，不掩盖原始写入异常

    def _flush_safe(self) -> None:
        """写入后尽量触发 flush，使数据立刻可查；后端不支持时静默跳过。"""
        try:
            self.client.flush(collection_name=self.collection)
        except Exception:
            pass

    # ---------- 检索 ----------
    def search_dense(
        self, query_vec: list[float], top_k: int = 20, filter_expr: str = ""
    ) -> list[dict]:
        res = self.client.search(
            collection_name=self.collection,
            data=[query_vec],
            anns_field="dense",
            limit=top_k,
            filter=filter_expr,
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

    # ---------- 全量读取 ----------
    def _query_all(self, output_fields: list[str], max_rows: int) -> list[dict]:
        """优先用 query_iterator 突破 offset 16384 上限，失败时回退 offset 分页。"""
        rows: list[dict] = []
        iterator_fn = getattr(self.client, "query_iterator", None)
        if iterator_fn is not None:
            try:
                iterator = iterator_fn(
                    collection_name=self.collection,
                    filter="id >= 0",
                    output_fields=output_fields,
                    batch_size=_PAGE_SIZE,
                )
                try:
                    while len(rows) < max_rows:
                        batch = iterator.next()
                        if not batch:
                            break
                        rows.extend(batch)
                finally:
                    iterator.close()
                return rows[:max_rows]
            except Exception:
                rows = []  # 迭代器中途失败则整段重来，避免半全量/重复

        offset = 0
        while len(rows) < max_rows:
            if offset + _PAGE_SIZE > _MAX_PAGE_WINDOW:
                break  # 后端不支持迭代器时的已知上界
            batch = self.client.query(
                collection_name=self.collection,
                filter="id >= 0",
                output_fields=output_fields,
                limit=_PAGE_SIZE,
                offset=offset,
            )
            if not batch:
                break
            rows.extend(batch)
            if len(batch) < _PAGE_SIZE:
                break
            offset += _PAGE_SIZE
        return rows[:max_rows]

    def all_chunks(self, max_rows: int = 100000, wait_visible: bool = False) -> list[dict]:
        """读取全量 chunk（用于构建 BM25 内存索引）。

        Milvus 默认 Bounded 一致性下，新写入可能短暂不可见（且行数统计可能滞后）：
        - wait_visible=True：刚写入过数据，强制重试直到查询可见（约 3 秒上限）
        - 否则仅当统计行数 > 0 且查询为空时短暂重试
        """
        expected = self.count()
        rows: list[dict] = []
        for _ in range(6):
            rows = self._query_all(OUTPUT_FIELDS, max_rows)
            if rows or (expected == 0 and not wait_visible):
                break
            time.sleep(0.5)
        return rows

    # ---------- 管理 ----------
    @staticmethod
    def _escape_filter_value(value: str) -> str:
        return str(value).replace("\\", "\\\\").replace('"', '\\"')

    def delete_by_source(self, source: str) -> int:
        safe = self._escape_filter_value(source)
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
        counts: dict[str, int] = {}
        for r in self._query_all(["source"], max_rows=10**9):
            counts[r["source"]] = counts.get(r["source"], 0) + 1
        return counts

    def drop(self) -> None:
        """删除整个 collection（测试隔离 / 重建索引用）。"""
        if self.client.has_collection(self.collection):
            self.client.drop_collection(self.collection)

    def count(self, filter_expr: str = "id >= 0") -> int:
        # 用 query count(*) 而非 get_collection_stats：后者在 Standalone 上
        # 统计有延迟（写入/删除后不会立刻反映），query 结果实时且精确。
        res = self.client.query(
            collection_name=self.collection,
            filter=filter_expr,
            output_fields=["count(*)"],
        )
        if res and isinstance(res[0], dict):
            return int(res[0].get("count(*)", 0))
        return 0

    def count_source(self, source: str) -> int:
        safe = self._escape_filter_value(source)
        return self.count(f'source == "{safe}"')
