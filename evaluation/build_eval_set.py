"""构建检索评测集模板。

评测集 JSON 格式：
[
  {"question": "...", "gold": [{"source": "文档名关键词", "must_contain": "答案必须包含的关键句"}]}
]

标注方法：针对每篇文档写 5-10 个问题；must_contain 填写正确答案必须命中的
关键短语（须与文档原文一致）。内置示例对应 scripts/demo_data 的两篇文档。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEMO_EVAL_SET = [
    {
        "question": "Milvus 有哪几种部署形态？",
        "gold": [{"source": "milvus入门", "must_contain": "Milvus Lite"}],
    },
    {
        "question": "生产环境最常用的 Milvus 索引类型是什么？",
        "gold": [{"source": "milvus入门", "must_contain": "HNSW"}],
    },
    {
        "question": "文本向量检索最常用的距离度量是什么？",
        "gold": [{"source": "milvus入门", "must_contain": "COSINE"}],
    },
    {
        "question": "RAG 分为哪两个阶段？",
        "gold": [{"source": "rag原理", "must_contain": "离线索引"}],
    },
    {
        "question": "RRF 融合中常数 k 通常取多少？",
        "gold": [{"source": "rag原理", "must_contain": "60"}],
    },
    {
        "question": "为什么需要混合检索？",
        "gold": [{"source": "rag原理", "must_contain": "各有盲区"}],
    },
]


def main(out_path: str = "evaluation/eval_set.json") -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(DEMO_EVAL_SET, f, ensure_ascii=False, indent=2)
    print(f"已生成评测集模板: {out_path}（共 {len(DEMO_EVAL_SET)} 条，可自行扩充标注）")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "evaluation/eval_set.json")
