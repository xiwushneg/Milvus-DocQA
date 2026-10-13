"""批量入库脚本。

用法:
    python scripts/ingest.py --path scripts/demo_data        # 目录
    python scripts/ingest.py --path /path/to/report.pdf      # 单文件
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import load_config  # noqa: E402
from core.pipeline import build_pipeline  # noqa: E402

SUFFIXES = {".pdf", ".docx", ".md", ".markdown", ".txt", ".text"}


def main() -> None:
    ap = argparse.ArgumentParser(description="文档批量入库")
    ap.add_argument("--path", required=True, help="文件或目录路径")
    ap.add_argument("--config", default=None, help="配置文件路径（默认 config/config.yaml）")
    args = ap.parse_args()

    p = Path(args.path)
    if p.is_dir():
        files = [f for f in sorted(p.rglob("*")) if f.suffix.lower() in SUFFIXES]
    else:
        files = [p]
    if not files:
        print("未找到可导入的文档")
        return

    cfg = load_config(args.config)
    pipeline = build_pipeline(cfg)
    print(f"共 {len(files)} 个文件待导入，embedding = {pipeline.embedder.name}")
    for f in files:
        stats = pipeline.ingest_bytes(f.name, f.read_bytes())
        print(
            f"  ✔ {stats['source']}: {stats['segments']} 段 → {stats['chunks']} 块，"
            f"耗时 {stats['elapsed_ms']}ms（含向量化 {stats['embed_ms']}ms）"
        )
    print(f"完成，Milvus 当前共 {pipeline.store.count()} 块")


if __name__ == "__main__":
    main()
