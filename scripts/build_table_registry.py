"""预构建表格工件：扫描 Milvus 全量 chunk 的管道表格，写 CSV + registry.json。

用法:
    python scripts/build_table_registry.py
    python scripts/build_table_registry.py --out data/tables --max-rows 100000

不新增 collection/字段；仅读取现有 chunk 文本。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import load_config  # noqa: E402
from core.agentic.tables import TableRegistry  # noqa: E402
from core.vectorstore import MilvusStore  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="构建表格工件注册表")
    ap.add_argument("--config", default=None, help="配置文件路径（默认 config/config.yaml）")
    ap.add_argument("--out", default=None, help="输出目录（默认 agent.tools.tables_dir 或 data/tables）")
    ap.add_argument("--max-rows", type=int, default=100000, help="最多扫描的 chunk 数")
    args = ap.parse_args()

    cfg = load_config(args.config)
    mcfg = cfg.get("milvus", {}) or {}
    store = MilvusStore(
        uri=mcfg.get("uri", "http://localhost:19530"),
        collection=mcfg.get("collection", "docqa_chunks"),
        dense_dim=int(mcfg.get("dense_dim", 1024)),
    )
    rows = store.all_chunks(max_rows=args.max_rows)
    out_dir = (
        args.out
        or (cfg.get("agent", {}) or {}).get("tools", {}).get("tables_dir")
        or "data/tables"
    )
    registry = TableRegistry.build(rows, out_dir)
    print(f"扫描 {len(rows)} 个 chunk，识别 {len(registry)} 张表 → {out_dir}")
    for table in registry.all()[:10]:
        print(
            f"  - {table.table_id}: {table.source} 第{table.page}页 "
            f"{len(table.columns)} 列 × {len(table.rows)} 行"
        )


if __name__ == "__main__":
    main()
